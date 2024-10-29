"""Tests for the aeon API."""

from pathlib import Path

import pandas as pd
import pytest

import aeon
from aeon.schema.schemas import exp02

nonmonotonic_path = Path(__file__).parent.parent / "data" / "nonmonotonic"
monotonic_path = Path(__file__).parent.parent / "data" / "monotonic"


@pytest.mark.api
def test_load_start_only():
    data = aeon.load(
        nonmonotonic_path,
        exp02.Patch2.Encoder,
        start=pd.Timestamp("2022-06-06T13:00:49"),
        downsample=None,
    )
    if len(data) <= 0:
        raise AssertionError("Loaded data is empty. Expected non-empty data.")


@pytest.mark.api
def test_load_end_only():
    data = aeon.load(
        nonmonotonic_path,
        exp02.Patch2.Encoder,
        end=pd.Timestamp("2022-06-06T13:00:49"),
        downsample=None,
    )
    if len(data) <= 0:
        raise AssertionError("Loaded data is empty. Expected non-empty data.")


@pytest.mark.api
def test_load_filter_nonchunked():
    data = aeon.load(
        nonmonotonic_path, exp02.Metadata, start=pd.Timestamp("2022-06-06T09:00:00")
    )
    if len(data) <= 0:
        raise AssertionError("Loaded data is empty. Expected non-empty data.")


@pytest.mark.api
def test_load_monotonic():
    data = aeon.load(monotonic_path, exp02.Patch2.Encoder, downsample=None)
    if len(data) <= 0:
        raise AssertionError("Loaded data is empty. Expected non-empty data.")

    if not data.index.is_monotonic_increasing:
        raise AssertionError("Data index is not monotonic increasing.")


@pytest.mark.api
def test_load_nonmonotonic():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder, downsample=None)
    if data.index.is_monotonic_increasing:
        raise AssertionError(
            "Data index is monotonic increasing, but it should not be."
        )


@pytest.mark.api
def test_load_encoder_with_downsampling():
    DOWNSAMPLE_PERIOD = 0.02
    data = aeon.load(monotonic_path, exp02.Patch2.Encoder, downsample=True)
    raw_data = aeon.load(monotonic_path, exp02.Patch2.Encoder, downsample=None)

    # Check that the length of the downsampled data is less than the raw data
    if len(data) >= len(raw_data):
        raise AssertionError(
            "Downsampled data length should be less than raw data length."
        )

    # Check that the first timestamp of the downsampled data is within 20ms of the raw data
    if abs(data.index[0] - raw_data.index[0]).total_seconds() > DOWNSAMPLE_PERIOD:
        raise AssertionError(
            "The first timestamp of downsampled data is not within 20ms of raw data."
        )

    # Check that the last timestamp of the downsampled data is within 20ms of the raw data
    if abs(data.index[-1] - raw_data.index[-1]).total_seconds() > DOWNSAMPLE_PERIOD:
        raise AssertionError(
            f"The last timestamp of downsampled data is not within {DOWNSAMPLE_PERIOD*1000} ms of raw data."
        )

    # Check that the minimum difference between consecutive timestamps in the downsampled data
    # is at least 20ms (50Hz)
    min_diff = data.index.to_series().diff().dt.total_seconds().min()
    if min_diff < DOWNSAMPLE_PERIOD:
        raise AssertionError(
            f"Minimum difference between consecutive timestamps is less than {DOWNSAMPLE_PERIOD} seconds."
        )

    # Check that the timestamps in the downsampled data are strictly increasing
    if not data.index.is_monotonic_increasing:
        raise AssertionError(
            "Timestamps in downsampled data are not strictly increasing."
        )


if __name__ == "__main__":
    pytest.main()
