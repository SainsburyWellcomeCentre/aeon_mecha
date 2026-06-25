"""Integration tests for XArrayNetCDFCodec against a real DataJoint table.

Runs on testcontainers MySQL (see ``mysql_container`` / ``dj_config_integration``).
A throwaway schema with a single ``<xarray@xarray_store>`` column is backed by a
temp-dir file store, so no real CEPH path is touched.

Note: ``datajoint.json`` is untracked, so CI has no ``xarray_store`` on disk — the
``xarray_store`` fixture registers it in ``dj.config`` (pointing at ``tmp_path``)
rather than relying on the file.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.integration


def make_mock_dataset(n=1000, n_ch=4):
    return xr.Dataset(
        {
            "signal": (("time", "channel"), np.random.randn(n, n_ch).astype("float32")),
            "flag": (("time",), np.random.rand(n) > 0.5),
        },
        coords={
            "time": pd.date_range("2026-06-03", periods=n, freq="20ms"),
            "channel": np.arange(n_ch),
        },
        attrs={"fs": 50.0, "note": "mock"},
    )


@pytest.fixture
def xarray_store(dj_config_integration, tmp_path):
    """Point the ``xarray_store`` file store at a temp dir (created up-front).

    ``_get_backend`` raises ``FileNotFoundError`` if the location doesn't already
    exist, so the directory is made before any insert. The previous store entry
    (if any) is restored on teardown.
    """
    import datajoint as dj

    location = tmp_path / "xarray_store"
    location.mkdir()
    stores = dj.config["stores"]
    saved = stores.get("xarray_store")
    stores["xarray_store"] = {"protocol": "file", "location": str(location)}
    yield location
    if saved is None:
        stores.pop("xarray_store", None)
    else:
        stores["xarray_store"] = saved


@pytest.fixture
def mock_xarray_table(xarray_store):
    """Throwaway schema + table with one ``<xarray@xarray_store>`` column."""
    import datajoint as dj

    from aeon.dj_pipeline import get_schema_name  # importing also registers the codec

    schema = dj.Schema(get_schema_name("test_xarray_codec"))

    @schema
    class MockXArray(dj.Manual):
        definition = """
        rec_id : int
        ---
        data : <xarray@xarray_store>
        """

    yield MockXArray, schema, xarray_store
    schema.drop()


class TestValidate:
    def test_accepts_dataset_rejects_others(self, dj_config_integration):
        from datajoint.errors import DataJointError

        from aeon.dj_pipeline.utils.xarray_codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
        mock = make_mock_dataset(n=5)
        codec.validate(mock)  # a Dataset must not raise

        with pytest.raises(DataJointError, match="requires an xarray.Dataset"):
            codec.validate([1, 2, 3])

        # a DataArray is rejected — caller must convert (avoids store-DataArray/get-Dataset)
        with pytest.raises(DataJointError, match=r"got DataArray.*to_dataset"):
            codec.validate(mock["signal"])


class TestDBRoundTrip:
    def test_round_trip_returns_lazy_equal_dataset(self, mock_xarray_table):
        mock_xarray, _schema, _loc = mock_xarray_table
        mock = make_mock_dataset()

        mock_xarray.insert1({"rec_id": 1, "data": mock})
        got = (mock_xarray & {"rec_id": 1}).fetch1("data")

        assert isinstance(got, xr.Dataset)
        assert got["signal"].variable._in_memory is False  # lazy until accessed
        xr.testing.assert_equal(got.load(), mock)

    def test_real_file_at_schema_addressed_tokened_path(self, mock_xarray_table):
        mock_xarray, _schema, loc = mock_xarray_table

        mock_xarray.insert1({"rec_id": 7, "data": make_mock_dataset(n=20)})

        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 1
        assert files[0].is_file()
        assert "rec_id=7" in files[0].as_posix()  # path mirrors schema structure
        assert files[0].name != "data.nc"  # filename carries a random token

    def test_fetch_is_lazy_without_dask(self, mock_xarray_table):
        mock_xarray, _schema, _loc = mock_xarray_table

        mock_xarray.insert1({"rec_id": 1, "data": make_mock_dataset()})
        got = (mock_xarray & {"rec_id": 1}).fetch1("data")

        assert got["signal"].chunks is None  # chunks=None, no dask
        assert got["signal"].variable._in_memory is False
        got["signal"].load()
        assert got["signal"].variable._in_memory is True

    def test_two_rows_two_files(self, mock_xarray_table):
        mock_xarray, _schema, loc = mock_xarray_table
        ds1, ds2 = make_mock_dataset(n=100), make_mock_dataset(n=200)

        mock_xarray.insert([{"rec_id": 1, "data": ds1}, {"rec_id": 2, "data": ds2}])

        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 2
        assert len({f.name for f in files}) == 2  # distinct tokened files
        xr.testing.assert_equal((mock_xarray & {"rec_id": 1}).fetch1("data").load(), ds1)
        xr.testing.assert_equal((mock_xarray & {"rec_id": 2}).fetch1("data").load(), ds2)
