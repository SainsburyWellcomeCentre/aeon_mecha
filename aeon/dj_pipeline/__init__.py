"""DataJoint pipeline for Aeon."""

import json
import logging
import os

import datajoint as dj

# Register AeonStreamCodec BEFORE any schema activation
from aeon.dj_pipeline.utils.codec import AeonStreamCodec  # noqa: F401

# Import dict_to_uuid from utils for backward compatibility
from aeon.dj_pipeline.utils import dict_to_uuid

logger = dj.logger

_default_database_prefix = "aeon_"
_default_repository_config = {"ceph_aeon": "/ceph/aeon"}

db_prefix = dj.config.database.database_prefix or os.getenv("DJ_DB_PREFIX") or _default_database_prefix

repository_config = (
    json.loads(os.environ["DJ_REPOSITORY_CONFIG"])
    if "DJ_REPOSITORY_CONFIG" in os.environ
    else _default_repository_config
)


def get_schema_name(name) -> str:
    """Return a schema name."""
    return db_prefix + name


def fetch_stream(query, drop_pk=True, round_microseconds=True):
    """Fetches data from a Stream table based on a query and returns it as a DataFrame.

    Provided a query containing data from a Stream table,
    fetch and aggregate the data into one DataFrame indexed by "time"

    Supports both codec-based tables (stream_df column) and legacy blob-based tables.

    Args:
        query (datajoint.Query): A query object containing data from a Stream table
        drop_pk (bool, optional): Drop primary key columns. Defaults to True.
        round_microseconds (bool, optional): Round timestamps to microseconds. Defaults to True.
            (this is important as timestamps in mysql is only accurate to microseconds)
    """
    import pandas as pd

    filtered = query & "sample_count > 0"

    if "stream_df" in query.heading.secondary_attributes:
        # Codec path: stream_df auto-decodes to DataFrame via AeonStreamCodec
        rows = filtered.to_dicts()
        if not rows:
            return pd.DataFrame()
        dfs = []
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
    else:
        # Legacy blob path
        df = filtered.to_pandas().reset_index()
        cols2explode = [
            c for c in query.heading.secondary_attributes if query.heading.attributes[c].is_blob
        ]
        df = df.explode(column=cols2explode)
        cols2drop = ["sample_count"] + (query.primary_key if drop_pk else [])
        df.drop(columns=cols2drop, inplace=True, errors="ignore")
        df.rename(columns={"timestamps": "time"}, inplace=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df = df.convert_dtypes(
            convert_string=False,
            convert_integer=False,
            convert_boolean=False,
            convert_floating=False,
        )

    if not df.empty and round_microseconds:
        logging.warning(
            "Rounding timestamps to microseconds is now enabled by default."
            " To disable, set round_microseconds=False."
        )
        df.index = df.index.round("us")
    return df


try:
    from . import streams
except ImportError:
    try:
        from .utils import streams_maker

        streams = dj.VirtualModule("streams", streams_maker.schema_name)
    except Exception as e:
        logger.debug(f"Could not import streams module: {e}")
