<<<<<<< HEAD
from pathlib import Path

=======
>>>>>>> b9a1e3f... Blackened and ruffed
import pandas as pd
import pytest
from pytest import mark

<<<<<<< HEAD
import aeon
from aeon.schema.dataset import exp02

nonmonotonic_path = Path(__file__).parent.parent / "data" / "nonmonotonic"
monotonic_path = Path(__file__).parent.parent / "data" / "monotonic"


@mark.api
def test_load_start_only():
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder, start=pd.Timestamp("2022-06-06T13:00:49"))
=======
import aeon.io.api as aeon
from aeon.schema.dataset import exp02


@mark.api
def test_load_start_only():
    data = aeon.load(
        "./tests/data/nonmonotonic", exp02.Patch2.Encoder, start=pd.Timestamp("2022-06-06T13:00:49")
    )
>>>>>>> b9a1e3f... Blackened and ruffed
    assert len(data) > 0


@mark.api
def test_load_end_only():
    data = aeon.load(
<<<<<<< HEAD
        nonmonotonic_path, exp02.Patch2.Encoder, end=pd.Timestamp("2022-06-06T13:00:49")
=======
        "./tests/data/nonmonotonic", exp02.Patch2.Encoder, end=pd.Timestamp("2022-06-06T13:00:49")
>>>>>>> b9a1e3f... Blackened and ruffed
    )
    assert len(data) > 0


@mark.api
def test_load_filter_nonchunked():
<<<<<<< HEAD
    data = aeon.load(
        nonmonotonic_path, exp02.Metadata, start=pd.Timestamp("2022-06-06T09:00:00")
    )
=======
    data = aeon.load("./tests/data/nonmonotonic", exp02.Metadata, start=pd.Timestamp("2022-06-06T09:00:00"))
>>>>>>> b9a1e3f... Blackened and ruffed
    assert len(data) > 0


@mark.api
def test_load_monotonic():
<<<<<<< HEAD
    data = aeon.load(monotonic_path, exp02.Patch2.Encoder)
    assert len(data) > 0 and data.index.is_monotonic_increasing

=======
    data = aeon.load("./tests/data/monotonic", exp02.Patch2.Encoder)
    assert data.index.is_monotonic_increasing
>>>>>>> b9a1e3f... Blackened and ruffed


@mark.api
def test_load_nonmonotonic():
<<<<<<< HEAD
    data = aeon.load(nonmonotonic_path, exp02.Patch2.Encoder)
=======
    data = aeon.load("./tests/data/nonmonotonic", exp02.Patch2.Encoder)
>>>>>>> b9a1e3f... Blackened and ruffed
    assert not data.index.is_monotonic_increasing


if __name__ == "__main__":
    pytest.main()
