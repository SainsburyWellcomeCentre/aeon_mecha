"""Manual JSON serialization for MariaDB json columns.

MariaDB 10.3 aliases ``json`` to ``longtext``. When DataJoint reads the column
type back from the database, it sees ``longtext`` instead of ``json`` and sets
``attr.json = False``. This means DataJoint skips the automatic
``json.dumps()`` on insert and ``json.loads()`` on fetch that it normally does
for json columns. The result is a TypeError on insert (dict passed where string
expected) and raw JSON strings returned on fetch.

This module provides a helper that serializes values to JSON strings only when
DataJoint won't do it itself. It checks ``attr.json`` at runtime, so the same
code works on both MariaDB (manual serialization) and MySQL (DataJoint handles
it), and will stop serializing automatically if DataJoint fixes the underlying
issue.

See: https://github.com/datajoint/datajoint-python/issues/1438
"""

import json


def json_serialized(value, needs_serialize):
    """Serialize a value to a JSON string only when DataJoint won't.

    On MariaDB, ``attr.json`` is ``False`` for json columns (reported as
    ``longtext``), so DataJoint skips serialization. On MySQL (or after a
    future DataJoint fix), ``attr.json`` is ``True`` and DataJoint handles it.
    This function adapts automatically based on the caller's check.

    Parameters
    ----------
    value : dict, list, or str
        The value to serialize. If already a string, returned as-is.
    needs_serialize : bool
        ``True`` when DataJoint won't serialize (``attr.json`` is ``False``).
        Callers should compute this once per table via::

            needs_serialize = not self.heading.attributes['some_json_col'].json

    Returns
    -------
    dict, list, or str
        The original value when DataJoint will handle serialization,
        or a JSON string when it won't.
    """
    if not needs_serialize or isinstance(value, str):
        return value
    return json.dumps(value)
