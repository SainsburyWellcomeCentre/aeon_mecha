"""DataJoint pipeline for Aeon."""

import json
import logging
import os
from typing import cast

import datajoint as dj
import pandas as pd

# Register AeonStreamCodec BEFORE any schema activation
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
