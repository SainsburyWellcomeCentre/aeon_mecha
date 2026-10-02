"""Unit tests for codec.py — no database required."""

from contextlib import nullcontext

import pytest
import xarray as xr
from datajoint.errors import DataJointError
from datajoint.settings import Config

pytestmark = pytest.mark.unit


class TestOnixStreamCodecEncode:
    """Direct ``encode()`` calls against OnixStreamCodec, no DB round trip."""

    def test_encodes_valid_dict(self):
        """Test that encode returns the input dict unchanged for a valid reference."""
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        ref = {
            "experiment_name": "exp01",
            "epoch_start": "2024-06-04 10:24:07",
            "sync_start": "2024-06-04 11:00:00",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055",
        }
        encoded = codec.encode(ref)
        assert encoded == ref

    def test_rejects_non_dict(self):
        """Test that encode raises TypeError for non-dict input."""
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(TypeError, match="OnixStreamCodec expects a dict"):
            codec.encode("not-a-dict")

    def test_rejects_dict_missing_keys(self):
        """Test that encode raises ValueError when required keys are missing."""
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"experiment_name": "exp01"})

    def test_codec_name(self):
        """Test that the codec's registered name is ``aeon_onix_stream``."""
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        assert OnixStreamCodec.name == "aeon_onix_stream"


@pytest.fixture
def dj_config(tmp_path):
    """Real ``dj.settings.Config`` with an ``xarray_store`` file store under tmp_path."""
    config = Config()
    config.stores = {"xarray_store": {"protocol": "file", "location": str(tmp_path)}}
    return config


class TestXArrayNetCDFCodec:
    """``validate()``, ``encode()``/``decode()``, and ``_local_path()`` on XArrayNetCDFCodec."""

    @pytest.mark.parametrize(
        ("value", "match"),
        [
            (lambda ds: ds, None),
            (lambda ds: [1, 2, 3], "requires an xarray.Dataset"),
            (lambda ds: ds["signal"], r"got DataArray.*to_dataset"),
        ],
        ids=["valid: dataset", "invalid: list", "invalid: dataarray - caller must convert"],
    )
    def test_validate(self, mock_xarray_dataset, value, match):
        """Test that validate accepts an xarray.Dataset and rejects other types."""
        from aeon.dj_pipeline.utils.codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
        expectation = nullcontext() if match is None else pytest.raises(DataJointError, match=match)
        with expectation:
            codec.validate(value(mock_xarray_dataset))

    def test_encode_writes_schema_addressed_nc_file(self, dj_config, mock_xarray_dataset, tmp_path):
        """Test that encode writes one tokened ``.nc`` file under a schema-addressed path."""
        from aeon.dj_pipeline.utils.codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
        key = {"_schema": "test_schema", "_table": "test_table", "rec_id": 1, "_config": dj_config}
        stored = codec.encode(mock_xarray_dataset, key=key, store_name="xarray_store")
        assert stored["store"] == "xarray_store"
        assert stored["dims"] == {"time": 20, "channel": 4}
        assert set(stored["data_vars"]) == {"signal", "flag"}
        files = list(tmp_path.rglob("data_*.nc"))
        assert len(files) == 1
        assert "rec_id=1" in files[0].as_posix()

    def test_decode_returns_lazy_equal_dataset(self, dj_config, mock_xarray_dataset):
        """Test that decode reopens lazily but equal once loaded."""
        from aeon.dj_pipeline.utils.codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
        key = {"_schema": "test_schema", "_table": "test_table", "rec_id": 1, "_config": dj_config}
        stored = codec.encode(mock_xarray_dataset, key=key, store_name="xarray_store")
        decoded = codec.decode(stored, key={"_config": dj_config})
        assert decoded["signal"].chunks is None  # xarray's own lazy indexing, not dask
        assert decoded["signal"].variable._in_memory is False
        xr.testing.assert_equal(decoded.load(), mock_xarray_dataset)

    def test_rejects_non_file_protocol(self, dj_config):
        """Test that a non-``file`` store protocol is rejected."""
        from aeon.dj_pipeline.utils.codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
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
