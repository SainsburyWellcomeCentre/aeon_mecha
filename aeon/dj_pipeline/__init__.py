import os

import datajoint as dj
import hashlib
import uuid

_default_database_prefix = os.getenv("DJ_DB_PREFIX") or "aeon_"
_default_repository_config = {"ceph_aeon": "/ceph/aeon"}

# safe-guard in case `custom` is not provided
if "custom" not in dj.config:
    dj.config["custom"] = {}

db_prefix = dj.config["custom"].get("database.prefix", _default_database_prefix)

repository_config = dj.config['custom'].get('repository_config',
                                            _default_repository_config)

try:
    from .utils import streams_maker
    streams = dj.VirtualModule("streams", streams_maker.STREAMS_MODULE_NAME)
except:
    pass


def get_schema_name(name):
    return db_prefix + name


def dict_to_uuid(key):
    """
    Given a dictionary `key`, returns a hash string as UUID
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest())
