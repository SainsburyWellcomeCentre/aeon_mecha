"""Unit tests for the xarray NetCDF codec's on-disk serialization contract.

These exercise the pure xarray ⇄ NetCDF round-trip the codec relies on
(``to_netcdf`` / ``open_dataset`` with the h5netcdf engine and ``chunks=None``),
with no datajoint and no database. Codec-level tests that need a real
``SchemaCodec`` (validate, DB round-trip, GC) live in the integration module —
the unit harness mocks ``datajoint``, so a ``SchemaCodec`` subclass can't be
imported here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.unit


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


class TestNetCDFRoundTrip:
    def test_on_disk_round_trip_preserves_data(self, tmp_path):
        mock = make_mock_dataset(n=200)
        path = tmp_path / "mock.nc"
        mock.to_netcdf(path, engine="h5netcdf")

        got = xr.open_dataset(path, engine="h5netcdf", chunks=None)

        assert dict(got.sizes) == {"time": 200, "channel": 4}
        assert got["signal"].dtype == np.float32
        assert got["flag"].dtype == bool
        assert got["time"].dtype == np.dtype("datetime64[ns]")
        assert got.attrs["fs"] == 50.0
        xr.testing.assert_equal(got.load(), mock)

    def test_open_is_lazy_without_dask(self, tmp_path):
        path = tmp_path / "mock.nc"
        make_mock_dataset(n=100).to_netcdf(path, engine="h5netcdf")

        got = xr.open_dataset(path, engine="h5netcdf", chunks=None)

        # chunks=None => xarray's own lazy indexing, not dask
        assert got["signal"].chunks is None
        assert got["signal"].variable._in_memory is False
        got["signal"].load()
        assert got["signal"].variable._in_memory is True
