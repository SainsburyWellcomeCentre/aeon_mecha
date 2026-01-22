"""Helper functions and utilities for the Aeon project."""

import hashlib
import uuid


def dict_to_uuid(key) -> uuid.UUID:
    """Given a dictionary `key`, returns a hash string as UUID."""
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(k).encode())
        hashed.update(str(v).encode())
    return uuid.UUID(hex=hashed.hexdigest())
