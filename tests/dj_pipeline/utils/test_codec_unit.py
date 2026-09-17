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


class TestPynappleCodecRegistration:
    """The codec's name, and that importing the module registers it."""

    def test_codec_name(self):
        """Test that the codec's registered name is ``pynapple``."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        assert PynappleCodec.name == "pynapple"

    def test_importing_the_module_registers_the_codec(self):
        """Test that ``<pynapple@…>`` resolves, i.e. the codec is in DataJoint's registry.

        Registration is an import side effect of ``Codec.__init_subclass__``; nothing
        calls a decorator. If this fails, no table can declare the column type.
        """
        import sys

        import aeon.dj_pipeline.utils.codec  # noqa: F401  (import registers)

        # `import datajoint.codecs` would resolve through the mocked `datajoint`
        # package attribute (see `mock_dj_for_unit`); the real submodule is kept in
        # sys.modules by _REAL_DJ_SUBMODULES, so read the registry from there.
        assert "pynapple" in sys.modules["datajoint.codecs"]._codec_registry


@pytest.fixture
def dj_config_nap(tmp_path):
    """Real ``dj.settings.Config`` with a ``pynapple_store`` file store under tmp_path."""
    config = Config()
    config.stores = {"pynapple_store": {"protocol": "file", "location": str(tmp_path)}}
    return config


class TestPynappleCodecValidate:
    """``validate()`` accepts pynapple objects and rejects everything else."""

    def test_accepts_all_six_pynapple_types(self, mock_tsgroup, mock_intervalset):
        """Test that every pynapple container the spec names validates."""
        import numpy as np
        import pynapple as nap

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        t = np.arange(5.0)
        for value in (
            nap.Ts(t=t),
            nap.Tsd(t=t, d=np.arange(5)),
            nap.TsdFrame(t=t, d=np.zeros((5, 2))),
            nap.TsdTensor(t=t, d=np.zeros((5, 2, 2))),
            mock_intervalset,
            mock_tsgroup,
        ):
            PynappleCodec().validate(value)

    @pytest.mark.parametrize(
        "value", [[1, 2, 3], "not-pynapple", 42, None], ids=["list", "str", "int", "none"]
    )
    def test_rejects_non_pynapple(self, value):
        """Test that non-pynapple values are rejected by type name."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        with pytest.raises(DataJointError, match="requires a pynapple object"):
            PynappleCodec().validate(value)


class TestPynappleCodecDtype:
    """The store-only guard, which needs a concrete class to instantiate."""

    def test_requires_store_modifier(self):
        """Test that ``<pynapple>`` without ``@`` is rejected with a usable message."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        with pytest.raises(DataJointError, match=r"<pynapple> requires @"):
            PynappleCodec().get_dtype(is_store=False)

    def test_store_modifier_yields_json(self):
        """Test that the store form stores JSON metadata in the column."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        assert PynappleCodec().get_dtype(is_store=True) == "json"


class TestPynappleCodecEncodeDecode:
    """``encode``/``decode`` against a real file store, no DB."""

    def test_encode_writes_schema_addressed_npz(self, dj_config_nap, mock_tsgroup, tmp_path):
        """Test that encode writes one tokened ``.npz`` under a schema-addressed path."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        key = {"_schema": "test_schema", "_table": "test_table", "rec_id": 1, "_config": dj_config_nap}
        stored = PynappleCodec().encode(mock_tsgroup, key=key, store_name="pynapple_store")

        assert stored["store"] == "pynapple_store"
        assert stored["kind"] == "TsGroup"
        assert stored["n_rows"] == 3  # units, including the empty one
        assert stored["t_start"] > 3.0e9  # Harp epoch survived
        files = list(tmp_path.rglob("data_*.npz"))
        assert len(files) == 1
        assert "rec_id=1" in files[0].as_posix()

    def test_decode_round_trips_tsgroup(self, dj_config_nap, mock_tsgroup):
        """Test that decode returns an equal TsGroup: keys, times, support, metadata."""
        import numpy as np

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        codec = PynappleCodec()
        key = {"_schema": "s", "_table": "t", "rec_id": 1, "_config": dj_config_nap}
        decoded = codec.decode(
            codec.encode(mock_tsgroup, key=key, store_name="pynapple_store"),
            key={"_config": dj_config_nap},
        )

        assert list(decoded.index) == list(mock_tsgroup.index)  # incl. non-contiguous
        assert len(decoded[6]) == 0  # empty unit survives
        for unit in mock_tsgroup.index:
            np.testing.assert_array_equal(decoded[unit].t, mock_tsgroup[unit].t)
        np.testing.assert_array_equal(
            decoded.time_support.values,
            mock_tsgroup.time_support.values,  # two intervals
        )
        np.testing.assert_array_equal(
            decoded.get_info("covered_seconds"), mock_tsgroup.get_info("covered_seconds")
        )

    def test_decode_round_trips_intervalset(self, dj_config_nap, mock_intervalset):
        """Test that a non-TsGroup type round-trips through the generic path."""
        import numpy as np

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        codec = PynappleCodec()
        key = {"_schema": "s", "_table": "t", "rec_id": 2, "_config": dj_config_nap}
        decoded = codec.decode(
            codec.encode(mock_intervalset, key=key, store_name="pynapple_store"),
            key={"_config": dj_config_nap},
        )
        np.testing.assert_array_equal(decoded.values, mock_intervalset.values)
        np.testing.assert_array_equal(decoded.get_info("tag"), mock_intervalset.get_info("tag"))

    def test_timestamps_are_bit_exact(self, dj_config_nap, mock_tsgroup):
        """Test that Harp-magnitude float64 timestamps survive bit-for-bit.

        At 3.87e9 s the float64 ULP is 477 ns. A round trip that quantises here
        would silently shift spikes by a fraction of a sample.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        codec = PynappleCodec()
        key = {"_schema": "s", "_table": "t", "rec_id": 3, "_config": dj_config_nap}
        decoded = codec.decode(
            codec.encode(mock_tsgroup, key=key, store_name="pynapple_store"),
            key={"_config": dj_config_nap},
        )
        assert decoded[0].t.dtype == np.float64
        assert (decoded[0].t == mock_tsgroup[0].t).all()  # exact, not allclose

    def test_rejects_non_file_protocol(self, dj_config_nap):
        """Test that a non-``file`` store protocol is rejected."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        dj_config_nap.stores = {
            "s3_store": {
                "protocol": "s3",
                "endpoint": "e",
                "bucket": "b",
                "access_key": "k",
                "secret_key": "s",
                "location": "loc",
            }
        }
        with pytest.raises(DataJointError, match="protocol: file"):
            PynappleCodec()._local_path("some/path.npz", "s3_store", dj_config_nap)
