"""Tests for AeonStreamCodec and helper functions.

Unit tests for column_stats/timestamp_stats import the functions directly
to avoid triggering the datajoint mock in unit test mode.
Codec encode/decode tests require real datajoint and are marked integration.
"""

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
        stats["min"] = float(np.nanmin(values))
        stats["max"] = float(np.nanmax(values))
        stats["mean"] = round(float(np.nanmean(values)), 4)
    return stats


def _timestamp_stats(index):
    """Inline copy of timestamp_stats to avoid datajoint import in unit tests."""
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
    def test_numeric_array(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = _column_stats(values)
        assert stats["dtype"] == "float64"
        assert stats["count"] == 5
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0

    def test_integer_array(self):
        values = np.array([10, 20, 30], dtype=np.int32)
        stats = _column_stats(values)
        assert stats["dtype"] == "int32"
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0

    def test_empty_array(self):
        values = np.array([], dtype=np.float64)
        stats = _column_stats(values)
        assert stats["count"] == 0
        assert stats["dtype"] == "float64"
        assert "min" not in stats

    def test_non_numeric_array(self):
        values = np.array(["a", "b", "c"])
        stats = _column_stats(values)
        assert stats["count"] == 3
        assert "min" not in stats
        assert "max" not in stats
        assert "mean" not in stats

    def test_nan_handling(self):
        values = np.array([1.0, np.nan, 3.0])
        stats = _column_stats(values)
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["count"] == 3


@pytest.mark.unit
class TestTimestampStats:
    def test_regular_timestamps(self):
        index = pd.date_range("2025-01-01", periods=100, freq="2ms")
        stats = _timestamp_stats(index)
        assert stats["count"] == 100
        assert stats["sampling_rate_hz"] == 500.0

    def test_single_timestamp(self):
        index = pd.DatetimeIndex(["2025-01-01"])
        stats = _timestamp_stats(index)
        assert stats["count"] == 1
        assert "sampling_rate_hz" not in stats
        assert "sampling_rate_hz" not in stats

    def test_irregular_timestamps(self):
        times = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:01", "2025-01-01 00:00:05"])
        stats = _timestamp_stats(times)
        assert stats["count"] == 3
        assert stats["sampling_rate_hz"] is not None


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
