"""Unit tests for ``XArrayNetCDFCodec`` - no database required."""

from contextlib import nullcontext

import pytest
import xarray as xr
from datajoint.errors import DataJointError
from datajoint.settings import Config

pytestmark = pytest.mark.unit


@pytest.fixture
def codec():
    """Return an XArrayNetCDFCodec instance."""
    from aeon.dj_pipeline.utils.xarray_codec import XArrayNetCDFCodec

    return XArrayNetCDFCodec()


@pytest.fixture
def dj_config(tmp_path):
    """Real ``dj.settings.Config`` with an ``xarray_store`` file store under tmp_path."""
    config = Config()
    config.stores = {"xarray_store": {"protocol": "file", "location": str(tmp_path)}}
    return config


class TestValidate:
    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (lambda ds: ds, None),
            (lambda ds: [1, 2, 3], "requires an xarray.Dataset"),
            (lambda ds: ds["signal"], r"got DataArray.*to_dataset"),
        ],
        ids=["valid: dataset", "invalid: list", "invalid: dataarray - caller must convert"],
    )
    def test_validate(self, codec, mock_xarray_dataset, value, match):
        """Test that validate accepts an xarray.Dataset and rejects other types."""
        expectation = nullcontext() if match is None else pytest.raises(DataJointError, match=match)
        with expectation:
            codec.validate(value(mock_xarray_dataset))


class TestEncodeDecode:
    def test_encode_writes_schema_addressed_nc_file(self, codec, dj_config, mock_xarray_dataset, tmp_path):
        """Test that encode writes one tokened ``.nc`` file under a schema-addressed path."""
        key = {"_schema": "test_schema", "_table": "test_table", "rec_id": 1, "_config": dj_config}
        stored = codec.encode(mock_xarray_dataset, key=key, store_name="xarray_store")
        assert stored["store"] == "xarray_store"
        assert stored["dims"] == {"time": 20, "channel": 4}
        assert set(stored["data_vars"]) == {"signal", "flag"}
        files = list(tmp_path.rglob("data_*.nc"))
        assert len(files) == 1
        assert "rec_id=1" in files[0].as_posix()

    def test_decode_returns_lazy_equal_dataset(self, codec, dj_config, mock_xarray_dataset):
        """Test that decode reopens lazily but equal once loaded."""
        key = {"_schema": "test_schema", "_table": "test_table", "rec_id": 1, "_config": dj_config}
        stored = codec.encode(mock_xarray_dataset, key=key, store_name="xarray_store")
        decoded = codec.decode(stored, key={"_config": dj_config})
        assert decoded["signal"].chunks is None  # xarray's own lazy indexing, not dask
        assert decoded["signal"].variable._in_memory is False
        xr.testing.assert_equal(decoded.load(), mock_xarray_dataset)


class TestLocalPath:
    def test_rejects_non_file_protocol(self, codec, dj_config):
        """Test that a non-``file`` store protocol is rejected."""
        dj_config.stores = {
            "s3_store": {
                "protocol": "s3",
                "endpoint": "endpoint",
                "bucket": "bucket",
                "access_key": "key",
                "secret_key": "secret",
                "location": "loc",
            }
        }
        with pytest.raises(DataJointError, match="protocol: file"):
            codec._local_path("some/path.nc", "s3_store", dj_config)
