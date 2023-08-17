"""Utility functions."""

from typing import Union


def find_nested_key(obj: Union[dict, list], key: str):
    """Returns the value of the first found nested key."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:  # Found it!
                return v
            found = find_nested_key(v, key)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_nested_key(item, key)
            if found:
                return found
    return None
