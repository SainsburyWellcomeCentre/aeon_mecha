"""Unit tests for time utilities.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection — before any fixtures run —
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

from datetime import datetime

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "input_time",
    ["2026-04-15T09-03-01", "2026-04-15T090301Z"],
    ids=["old hyphenated", "new compact ISO 8601"],
)
def test_parse_epoch_timestamp(input_time):
    """Test parsing of new and old epoch string formats."""
    from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

    assert parse_epoch_timestamp(input_time) == datetime(2026, 4, 15, 9, 3, 1)
