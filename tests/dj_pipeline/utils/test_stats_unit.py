"""Unit tests for stats.py - pure functions, no database required.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection — before any fixtures run —
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestColumnStats:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                {"dtype": "float64", "count": 5, "min": 1.0, "max": 5.0, "mean": 3.0},
                id="numeric array",
            ),
            pytest.param(
                np.array([10, 20, 30], dtype=np.int32),
                {"dtype": "int32", "count": 3, "min": 10.0, "max": 30.0, "mean": 20.0},
                id="integer array",
            ),
            pytest.param(
                np.array([], dtype=np.float64),
                {"dtype": "float64", "count": 0},
                id="empty array",
            ),
            pytest.param(
                np.array(["a", "b", "c"]),
                {"dtype": "<U1", "count": 3},
                id="non-numeric array",
            ),
            pytest.param(
                np.array([1.0, np.nan, 3.0]),
                {"dtype": "float64", "count": 3, "min": 1.0, "max": 3.0, "mean": 2.0},
                id="partial NaN",
            ),
            pytest.param(
                np.array([np.nan, np.nan, np.nan]),
                {"dtype": "float64", "count": 3},
                id="all NaN",
            ),
            pytest.param(
                np.array([1.0, np.inf, 3.0]),
                {"dtype": "float64", "count": 3, "min": 1.0, "max": 3.0, "mean": 2.0},
                id="partial Inf",
            ),
            pytest.param(
                np.array([np.inf, -np.inf]),
                {"dtype": "float64", "count": 2},
                id="all Inf",
            ),
        ],
    )
    def test_column_stats(self, values, expected):
        """Test column_stats output for various input arrays, including edge cases."""
        from aeon.dj_pipeline.utils.stats import column_stats

        assert column_stats(values) == expected

    @pytest.mark.parametrize(
        "values",
        [
            np.array([np.nan]),
            np.array([np.inf]),
            np.array([1.0, np.nan, 3.0]),
            np.array([1.0, np.inf, 3.0]),
        ],
    )
    def test_column_stats_json_serializable(self, values):
        """min/max/mean must never be nan or inf (not JSON-serializable)."""
        from aeon.dj_pipeline.utils.stats import column_stats

        json.dumps(column_stats(values))  # must not raise


class TestTimestampStats:
    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            pytest.param(
                pd.date_range("2025-01-01", periods=100, freq="2ms"),
                {
                    "min": "2025-01-01 00:00:00",
                    "max": "2025-01-01 00:00:00.198000",
                    "count": 100,
                    "sampling_rate_hz": 500.0,
                },
                id="regular timestamps",
            ),
            pytest.param(
                pd.DatetimeIndex(["2025-01-01"]),
                {
                    "min": "2025-01-01 00:00:00",
                    "max": "2025-01-01 00:00:00",
                    "count": 1,
                },
                id="single timestamp",
            ),
            pytest.param(
                pd.DatetimeIndex([]),
                {"count": 0},
                id="empty index",
            ),
            pytest.param(
                pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:01", "2025-01-01 00:00:05"]),
                {
                    "min": "2025-01-01 00:00:00",
                    "max": "2025-01-01 00:00:05",
                    "count": 3,
                    "sampling_rate_hz": 0.4,
                },
                id="irregular timestamps",
            ),
        ],
    )
    def test_timestamp_stats(self, index, expected):
        """Test timestamp_stats output for various datetime indices, including edge cases."""
        from aeon.dj_pipeline.utils.stats import timestamp_stats

        assert timestamp_stats(index) == expected

    @pytest.mark.parametrize(
        "index",
        [
            pd.date_range("2025-01-01", periods=100, freq="2ms"),
            pd.DatetimeIndex(["2025-01-01"]),
            pd.DatetimeIndex([]),
        ],
    )
    def test_timestamp_stats_json_serializable(self, index):
        """All stats output must be JSON-serializable."""
        from aeon.dj_pipeline.utils.stats import timestamp_stats

        json.dumps(timestamp_stats(index))  # must not raise
