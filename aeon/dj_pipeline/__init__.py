"""DataJoint pipeline for Aeon."""

import json
import logging
import os
from typing import cast

import pymysql.converters

import datajoint as dj
import pandas as pd


# ---------------------------------------------------------------------------
# MariaDB compatibility patch
#
# On MariaDB 10.x, the JSON type aliases to longtext. DataJoint's codec
# columns (<filepath@dj_store>, <blob@dj_store>) produce Python dicts that
# DataJoint expects to serialize via json.dumps when attr.json is True.
# On MariaDB, attr.json is False (the column reports as longtext, not json),
# so the raw dict reaches pymysql, which raises:
#     TypeError: dict can not be used as parameter
#
# This patch teaches pymysql to serialize dicts as JSON strings. It must be
# applied before any DataJoint connection is created (Connection.__init__
# copies from converters.conversions at creation time).
#
# On MySQL 5.7+, this patch is inert: DataJoint serializes dicts to strings
# before they reach pymysql, so the dict encoder is never invoked.
#
# Remove when: migrated to MySQL, or DataJoint fixes external store codec
# handling on MariaDB (DJ 2.2.2 PR #1443 only fixes plain json columns).
# ---------------------------------------------------------------------------
def _escape_dict_as_json(val, charset, mapping=None):
    return pymysql.converters.escape_str(json.dumps(val, default=str), charset)


pymysql.converters.escape_dict = _escape_dict_as_json
pymysql.converters.encoders[dict] = _escape_dict_as_json
pymysql.converters.conversions[dict] = _escape_dict_as_json

# Register Aeon codecs BEFORE any schema activation
from aeon.dj_pipeline.utils.codec import (  # pyright: ignore[reportUnusedImport]
    AeonStreamCodec,
    OnixStreamCodec,
)

logger = dj.logger

_default_database_prefix = "aeon_"
_default_repository_config = {"ceph_aeon": "/ceph/aeon"}

os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

db_prefix = (
    dj.config.database.database_prefix or os.getenv("DJ_DATABASE_PREFIX") or _default_database_prefix
)

repository_config = (
    json.loads(os.environ["DJ_REPOSITORY_CONFIG"])
    if "DJ_REPOSITORY_CONFIG" in os.environ
    else _default_repository_config
)


def get_schema_name(name) -> str:
    """Return a schema name."""
    return db_prefix + name


def fetch_stream(query, drop_pk=True, round_microseconds=True):
    """Fetches data from a Stream table and returns it as a DataFrame.

    Fetches stream_df (decoded via AeonStreamCodec) from each matching row,
    concatenates into one DataFrame indexed by "time".

    Args:
        query (datajoint.Query): A query object containing data from a Stream table.
        drop_pk (bool, optional): Drop primary key columns. Defaults to True.
        round_microseconds (bool, optional): Round timestamps to microseconds. Defaults to True.
            (this is important as timestamps in mysql is only accurate to microseconds)
    """
    rows = (query & "sample_count > 0").to_dicts()
    if not rows:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for row in rows:
        stream_df = row["stream_df"]
        if not drop_pk:
            for pk_col in query.primary_key:
                if pk_col in row:
                    stream_df[pk_col] = row[pk_col]
        dfs.append(stream_df)

    df = pd.concat(dfs)
    df.index.name = "time"
    df.sort_index(inplace=True)

    if not df.empty and round_microseconds:
        logging.warning(
            "Rounding timestamps to microseconds is now enabled by default."
            " To disable, set round_microseconds=False."
        )
        df.index = cast(pd.DatetimeIndex, df.index).round("us")
    return df


try:
    from . import streams  # pyright: ignore[reportUnusedImport]
except ImportError:
    try:
        from .utils import streams_maker

        streams = dj.VirtualModule("streams", streams_maker.schema_name)
    except Exception as e:
        logger.debug(f"Could not import streams module: {e}")


# Activate downstream analysis schemas that depend on dynamically-generated
# stream tables. Each module's activate() is a no-op (logged warning) if its
# upstream streams aren't present in this experiment.
try:
    from . import analysis_feeder  # pyright: ignore[reportUnusedImport]

    analysis_feeder.activate()
except Exception as e:
    logger.debug(f"Could not activate analysis_feeder: {e}")
