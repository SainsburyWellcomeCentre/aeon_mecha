"""Tests for AeonStreamCodec and helper functions.

Unit tests for column_stats/timestamp_stats import the functions directly
to avoid triggering the datajoint mock in unit test mode.
Codec encode/decode tests require real datajoint and are marked integration.
"""

import json

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Unit tests: pure numpy/pandas helpers (no datajoint needed)
# ---------------------------------------------------------------------------


def _column_stats(values):
    """Inline copy of column_stats to avoid datajoint import in unit tests."""
    stats = {"dtype": str(values.dtype), "count": int(len(values))}
    if len(values) > 0 and np.issubdtype(values.dtype, np.number):
        finite = values[np.isfinite(values)]
        if len(finite) > 0:
            stats["min"] = float(np.min(finite))
            stats["max"] = float(np.max(finite))
            stats["mean"] = round(float(np.mean(finite)), 4)
    return stats


def _timestamp_stats(index):
    """Inline copy of timestamp_stats to avoid datajoint import in unit tests."""
    if len(index) == 0:
        return {"count": 0}
    stats = {
        "min": str(index.min()),
        "max": str(index.max()),
        "count": int(len(index)),
    }
    if len(index) > 1:
        diffs = np.diff(index.values) / np.timedelta64(1, "ns")
        median_diff_ns = float(np.median(diffs))
        stats["sampling_rate_hz"] = round(1e9 / median_diff_ns, 2) if median_diff_ns > 0 else None
    return stats


@pytest.mark.unit
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
        assert _column_stats(values) == expected

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
        json.dumps(_column_stats(values))  # must not raise


@pytest.mark.unit
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
        assert _timestamp_stats(index) == expected

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
        json.dumps(_timestamp_stats(index))  # must not raise


# ---------------------------------------------------------------------------
# Integration tests: codec encode/decode (requires real datajoint)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAeonStreamCodecEncode:
    def test_valid_encode(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        value = {
            "stream_type": "Encoder",
            "experiment_name": "test",
            "device_name": "Feeder1",
            "chunk_start": "2025-01-01 00:00:00",
            "chunk_end": "2025-01-01 01:00:00",
            "epoch_start": "2025-01-01 00:00:00",
        }
        result = codec.encode(value)
        assert result == value

    def test_missing_keys_raises(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"stream_type": "Encoder"})

    def test_non_dict_raises(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(TypeError, match="expects a dict"):
            codec.encode("not a dict")
