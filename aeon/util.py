"""Utility functions."""
from __future__ import annotations

from typing import Any


def find_nested_key(obj: dict | list, key: str) -> Any:
    """Returns the value of the first found nested key."""
    if isinstance(obj, dict):
        if v := obj.get(key):  # found it!
            return v
        for v in obj.values():
            found = find_nested_key(v, key)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_nested_key(item, key)
            if found:
                return found
    return None
