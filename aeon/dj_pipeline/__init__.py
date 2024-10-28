""" DataJoint pipeline for Aeon. """

import hashlib
import os
import uuid

import datajoint as dj

_default_database_prefix = os.getenv("DJ_DB_PREFIX") or "aeon_"
_default_repository_config = {"ceph_aeon": "/ceph/aeon"}

# safe-guard in case `custom` is not provided
if "custom" not in dj.config:
    dj.config["custom"] = {}

db_prefix = dj.config["custom"].get("database.prefix", _default_database_prefix)

repository_config = dj.config["custom"].get(
    "repository_config", _default_repository_config
)


def get_schema_name(name) -> str:
    """Return a schema name."""
    return db_prefix + name


def dict_to_uuid(key) -> uuid.UUID:
    """Given a dictionary `key`, returns a hash string as UUID."""
    hashed = hashlib.sha256()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest()[:32])


def fetch_stream(query, drop_pk=True):
    """Fetches data from a Stream table based on a query and returns it as a DataFrame.

    Provided a query containing data from a Stream table,
    fetch and aggregate the data into one DataFrame indexed by "time"
    """
    df = (query & "sample_count > 0").fetch(format="frame").reset_index()
    cols2explode = [
        c
        for c in query.heading.secondary_attributes
        if query.heading.attributes[c].type == "longblob"
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
    return df


try:
    from . import streams
except ImportError:
    try:
        from .utils import streams_maker

        streams = dj.VirtualModule("streams", streams_maker.schema_name)
    except:
        pass
