from pathlib import Path

import pandas as pd
import pytest
from pytest import mark

import aeon
from aeon.schema.schemas import exp02
from aeon.schema.ingestion_schemas import social03

nonmonotonic_path = Path(__file__).parent.parent / "data" / "nonmonotonic"
monotonic_path = Path(__file__).parent.parent / "data" / "monotonic"


@mark.api
def test_load_start_only():
    data = aeon.load(
        nonmonotonic_path, exp02.Patch2.Encoder, start=pd.Timestamp("2022-06-06T13:00:49")
    )
    assert len(data) > 0


@mark.api
def test_load_end_only():
    data = aeon.load(
        nonmonotonic_path, exp02.Patch2.Encoder, end=pd.Timestamp("2022-06-06T13:00:49")
    )
    assert len(data) > 0


@mark.api
def test_load_filter_nonchunked():
    data = aeon.load(nonmonotonic_path, exp02.Metadata, start=pd.Timestamp("2022-06-06T09:00:00"))
    assert len(data) > 0


@mark.api
def test_load_monotonic():
    data = aeon.load(monotonic_path, exp02.Patch2.Encoder)
    assert len(data) > 0
    assert data.index.is_monotonic_increasing


@mark.api
def test_load_nonmonotonic():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder)
    assert not data.index.is_monotonic_increasing


@mark.api
def test_load_encoder_with_downsampling():
    DOWNSAMPLE_PERIOD = 0.02
    data = aeon.load(monotonic_path, social03.Patch2.Encoder)
    raw_data = aeon.load(monotonic_path, exp02.Patch2.Encoder)

    # Check that the length of the downsampled data is less than the raw data
    assert len(data) < len(raw_data)

    # Check that the first timestamp of the downsampled data is within 20ms of the raw data
    assert abs(data.index[0] - raw_data.index[0]).total_seconds() <= DOWNSAMPLE_PERIOD

    # Check that the last timestamp of the downsampled data is within 20ms of the raw data
    assert abs(data.index[-1] - raw_data.index[-1]).total_seconds() <= DOWNSAMPLE_PERIOD

    # Check that the minimum difference between consecutive timestamps in the downsampled data
    # is at least 20ms (50Hz)
    assert data.index.to_series().diff().dt.total_seconds().min() >= DOWNSAMPLE_PERIOD

    # Check that the timestamps in the downsampled data are strictly increasing
    assert data.index.is_monotonic_increasing


if __name__ == "__main__":
    pytest.main()
