from pathlib import Path

import pandas as pd
import pytest
from pytest import mark

import aeon
from aeon.schema.schemas import exp02

nonmonotonic_path = Path(__file__).parent.parent / "data" / "nonmonotonic"
monotonic_path = Path(__file__).parent.parent / "data" / "monotonic"


@mark.api
def test_load_start_only():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder, start=pd.Timestamp("2022-06-06T13:00:49"))
    assert len(data) > 0


@mark.api
def test_load_end_only():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder, end=pd.Timestamp("2022-06-06T13:00:49"))
    assert len(data) > 0


@mark.api
def test_load_filter_nonchunked():
    data = aeon.load(nonmonotonic_path, exp02.Metadata, start=pd.Timestamp("2022-06-06T09:00:00"))
    assert len(data) > 0


@mark.api
def test_load_monotonic():
    data = aeon.load(monotonic_path, exp02.Patch2.Encoder)
    assert len(data) > 0 and data.index.is_monotonic_increasing


@mark.api
def test_load_nonmonotonic():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder)
    assert not data.index.is_monotonic_increasing


if __name__ == "__main__":
    pytest.main()
