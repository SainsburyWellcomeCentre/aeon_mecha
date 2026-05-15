"""Unit tests for Environment stream row building in acquisition.py.

Tests target the pure `_environment_row` builder. The thin DB-coupled
`_make_environment_stream` wrapper is exercised by integration tests.
"""

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def chunk_window():
    """(chunk_start, chunk_end, epoch_start) -- the same triple Chunk.fetch1 returns."""
    return (
        pd.Timestamp("2025-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 01:00:00"),
        pd.Timestamp("2025-01-01 00:00:00"),
    )


@pytest.fixture
def chunk_key():
    return {"experiment_name": "exp", "chunk_start": pd.Timestamp("2025-01-01 00:00:00")}


class TestEnvironmentRow:
    def test_builds_full_row_when_dataframe_has_data(self, chunk_key, chunk_window):
        import aeon.dj_pipeline.acquisition as acquisition  # noqa: PLR0402

        df = pd.DataFrame(
            {"state": ["A", "B"]},
            index=pd.DatetimeIndex(["2025-01-01 00:00:01", "2025-01-01 00:30:00"]),
        )
        row = acquisition._environment_row(
            df,
            key=chunk_key,
            chunk_window=chunk_window,
            stream_type="EnvironmentState",
        )
        assert row["sample_count"] == 2
        assert isinstance(row["timestamps"], dict)
        assert row["timestamps"].get("count") == 2
        assert row["stream_df"] == {
            "stream_type": "EnvironmentState",
            "experiment_name": "exp",
            "device_name": "Environment",
            "chunk_start": "2025-01-01 00:00:00",
            "chunk_end": "2025-01-01 01:00:00",
            "epoch_start": "2025-01-01 00:00:00",
        }

    def test_builds_empty_stats_when_df_is_none(self, chunk_key, chunk_window):
        """df=None means no reader was resolved (e.g., LightEvents on non-foragingABC).

        The row carries sample_count=0, empty timestamps, AND stream_df=None.
        """
        import aeon.dj_pipeline.acquisition as acquisition  # noqa: PLR0402

        row = acquisition._environment_row(
            None,
            key=chunk_key,
            chunk_window=chunk_window,
            stream_type="LightEvents",
        )
        assert row["sample_count"] == 0
        assert row["timestamps"] == {}
        assert row["stream_df"] is None

    def test_builds_empty_stats_when_df_is_empty(self, chunk_key, chunk_window):
        """Reader resolved but the chunk window contains no rows -> sample_count=0.

        Empty timestamps dict, but stream_df IS populated (decode will yield empty DataFrame).
        """
        import aeon.dj_pipeline.acquisition as acquisition  # noqa: PLR0402

        empty_df = pd.DataFrame({"state": []}, index=pd.DatetimeIndex([]))
        row = acquisition._environment_row(
            empty_df,
            key=chunk_key,
            chunk_window=chunk_window,
            stream_type="EnvironmentState",
        )
        assert row["sample_count"] == 0
        assert row["timestamps"] == {}
        assert row["stream_df"]["stream_type"] == "EnvironmentState"
        assert row["stream_df"]["device_name"] == "Environment"

    def test_includes_pk_fields_from_key(self, chunk_key, chunk_window):
        import aeon.dj_pipeline.acquisition as acquisition  # noqa: PLR0402

        row = acquisition._environment_row(
            None,
            key=chunk_key,
            chunk_window=chunk_window,
            stream_type="MessageLog",
        )
        assert row["experiment_name"] == "exp"
        assert row["chunk_start"] == chunk_key["chunk_start"]
