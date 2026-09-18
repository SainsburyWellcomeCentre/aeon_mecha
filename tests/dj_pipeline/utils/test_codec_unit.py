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


@pytest.fixture
def dj_config_nap(tmp_path):
    """Real ``dj.settings.Config`` with a ``pynapple_store`` file store under tmp_path."""
    config = Config()
    config.stores = {"pynapple_store": {"protocol": "file", "location": str(tmp_path)}}
    return config


def _sample_objects():
    """One small instance of each pynapple container, for round-trip coverage."""
    import numpy as np
    import pynapple as nap

    t = np.arange(6.0) + 3.87e9  # Harp magnitude, where the float64 gap is 477 ns
    return {
        "Ts": nap.Ts(t=t),
        "Tsd": nap.Tsd(t=t, d=np.arange(6, dtype="int64")),
        # TsdFrame and IntervalSet carry metadata, like TsGroup — and metadata is the
        # pickled half of the npz, the part whose encoding broke at pynapple 0.9.
        "TsdFrame": nap.TsdFrame(
            t=t,
            d=np.zeros((6, 3)),
            columns=["a", "b", "c"],
            metadata={"region": np.array(["ca1", "ca3", "dg"])},
        ),
        "TsdTensor": nap.TsdTensor(t=t, d=np.zeros((6, 2, 2))),
        "IntervalSet": nap.IntervalSet(
            start=[0.0, 20.0], end=[10.0, 30.0], metadata={"tag": np.array(["wake", "sleep"])}
        ),
    }


class TestPynappleCodecRoundTrip:
    """``validate`` / ``encode`` / ``decode`` against a real file store, no DB."""

    def test_accepts_all_six_pynapple_types(self, mock_tsgroup, mock_intervalset):
        """Test that every pynapple container the codec claims to take validates."""
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

    def test_rejects_non_pynapple(self):
        """Test that a non-pynapple value is rejected by type name."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        with pytest.raises(DataJointError, match="requires a pynapple object"):
            PynappleCodec().validate([1, 2, 3])

    def test_tsgroup_round_trips_bit_exactly(self, dj_config_nap, mock_tsgroup):
        """Test that keys, times, support and metadata all survive, exactly.

        Equality rather than allclose: the float64 ULP at Harp magnitude is 477 ns,
        so a quantising round trip would shift spikes within a sample and pass any
        tolerance. The fixture is deliberately awkward — non-contiguous keys, a unit
        with zero spikes, a two-interval support.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        codec = PynappleCodec()
        key = {"_schema": "s", "_table": "t", "rec_id": 1, "_config": dj_config_nap}
        stored = codec.encode(mock_tsgroup, key=key, store_name="pynapple_store")
        decoded = codec.decode(stored, key={"_config": dj_config_nap})

        assert list(decoded.index) == list(mock_tsgroup.index)
        assert len(decoded[6]) == 0
        for unit in mock_tsgroup.index:
            assert (decoded[unit].t == mock_tsgroup[unit].t).all()
        np.testing.assert_array_equal(decoded.time_support.values, mock_tsgroup.time_support.values)
        np.testing.assert_array_equal(
            decoded.get_info("covered_seconds"), mock_tsgroup.get_info("covered_seconds")
        )
        assert (stored["kind"], stored["n_rows"]) == ("TsGroup", 3)
        assert stored["t_start"] > 3.0e9

    @pytest.mark.parametrize("kind", ["Ts", "Tsd", "TsdFrame", "TsdTensor", "IntervalSet"])
    def test_every_other_type_round_trips_with_its_summary(self, dj_config_nap, kind):
        """Test the ``nap.load_file`` branch for each type that takes it.

        The codec is domain-agnostic and claims all six pynapple containers, so all
        six need round-tripping — not just the two this pipeline happens to use.
        Each case also checks the type-specific summary, which is what makes the
        stored JSON worth querying without opening the file.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        obj = _sample_objects()[kind]
        codec = PynappleCodec()
        key = {"_schema": "s", "_table": "t", "rec_id": 2, "_config": dj_config_nap}
        stored = codec.encode(obj, key=key, store_name="pynapple_store")
        decoded = codec.decode(stored, key={"_config": dj_config_nap})

        assert type(decoded).__name__ == kind
        assert stored["kind"] == kind
        assert stored["n_rows"] == len(obj)

        # Ts carries times and no values; IntervalSet carries values and no times.
        if hasattr(obj, "t"):
            np.testing.assert_array_equal(decoded.t, obj.t)
        if hasattr(obj, "values"):
            np.testing.assert_array_equal(decoded.values, obj.values)

        # metadata is pickled inside the npz, so it round-trips on its own path
        if hasattr(obj, "metadata"):
            assert sorted(decoded.metadata.columns) == sorted(obj.metadata.columns)
            for col in obj.metadata.columns:
                np.testing.assert_array_equal(
                    np.asarray(decoded.get_info(col)), np.asarray(obj.get_info(col))
                )

        expected_extras = {
            "Ts": set(),
            "Tsd": {"dtype"},
            "TsdFrame": {"dtype", "n_columns", "columns"},
            "TsdTensor": {"dtype", "shape"},
            "IntervalSet": {"total_seconds"},
        }[kind]
        assert expected_extras <= set(stored)

    def test_tsdframe_summary_names_its_columns(self, dj_config_nap):
        """Test that a TsdFrame's shape is legible from the stored JSON alone.

        ``n_rows`` counts samples, which says nothing about width — the same gap
        ``<xarray@store>`` fills with ``dims`` and ``data_vars``.
        """
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        key = {"_schema": "s", "_table": "t", "rec_id": 3, "_config": dj_config_nap}
        stored = PynappleCodec().encode(_sample_objects()["TsdFrame"], key=key, store_name="pynapple_store")
        assert stored["n_columns"] == 3
        assert stored["columns"] == ["a", "b", "c"]

    def test_tsgroup_summary_counts_events_not_just_units(self, dj_config_nap, mock_tsgroup):
        """Test that a TsGroup reports total events, so a caller can size a query."""
        from aeon.dj_pipeline.utils.codec import PynappleCodec

        key = {"_schema": "s", "_table": "t", "rec_id": 4, "_config": dj_config_nap}
        stored = PynappleCodec().encode(mock_tsgroup, key=key, store_name="pynapple_store")
        assert stored["n_rows"] == 3  # units
        assert stored["n_events"] == sum(len(mock_tsgroup[u]) for u in mock_tsgroup.index)

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


def _tsgroup_shapes():
    """TsGroup shapes that each break the fast path in a different way."""
    import numpy as np
    import pynapple as nap

    rng = np.random.default_rng(1)
    spikes = lambda n: np.sort(rng.uniform(0, 10, n))  # noqa: E731
    t = np.arange(6.0)
    return {
        # an empty member, non-contiguous keys, and a support with a gap
        "gaps_and_empty": nap.TsGroup(
            {
                0: nap.Ts(t=np.array([1.0, 2.0, 21.0])),
                6: nap.Ts(t=np.array([], dtype=float)),
                9: nap.Ts(t=np.array([22.0, 23.0])),
            },
            time_support=nap.IntervalSet(start=[0, 20], end=[10, 30]),
        ),
        # unit ids past int16, forcing the narrowing guard's upper bound
        "keys_beyond_int16": nap.TsGroup({k: nap.Ts(t=spikes(5)) for k in (0, 40_000, 70_000)}),
        # negative unit ids, which pynapple permits and which wrap if only max is
        # checked — a wrong sort order with no error
        "negative_keys": nap.TsGroup({k: nap.Ts(t=spikes(6)) for k in (-40_000, -1, 0, 7)}),
        # members carrying values, which take the `d` branch
        "tsd_members": nap.TsGroup({0: nap.Tsd(t=t, d=t * 2), 1: nap.Tsd(t=t + 0.5, d=t * 3)}),
    }


class TestPynappleFastPath:
    """The TsGroup fast path must equal ``nap.load_file`` exactly.

    This is the only place the codec depends on pynapple's private npz layout, so
    these tests are what make that dependency acceptable: a format change fails
    here rather than silently returning different data.
    """

    @staticmethod
    def _assert_same(a, b):
        """Assert two TsGroups are indistinguishable."""
        import numpy as np

        assert list(a.index) == list(b.index)
        for unit in a.index:
            np.testing.assert_array_equal(a[unit].t, b[unit].t)
            if hasattr(a[unit], "d"):
                np.testing.assert_array_equal(a[unit].d, b[unit].d)
        np.testing.assert_array_equal(a.time_support.values, b.time_support.values)
        assert sorted(a.metadata.columns) == sorted(b.metadata.columns)
        for col in a.metadata.columns:
            np.testing.assert_array_equal(np.asarray(a.get_info(col)), np.asarray(b.get_info(col)))

    @pytest.mark.parametrize("shape", list(_tsgroup_shapes()), ids=list(_tsgroup_shapes()))
    def test_fast_path_equals_stock_loader(self, shape, tmp_path):
        """Test equivalence on each shape that stresses a different part of the rebuild."""
        import pynapple as nap

        from aeon.dj_pipeline.utils.codec import _tsgroup_from_npz

        tg = _tsgroup_shapes()[shape]
        path = tmp_path / f"{shape}.npz"
        tg.save(str(path))
        self._assert_same(nap.load_file(str(path)), _tsgroup_from_npz(str(path)))

    def test_fast_path_preserves_rate(self, mock_tsgroup, tmp_path):
        """Test that ``rate`` matches stock.

        ``rate`` is n_events / tot_length(time_support). Building members without
        the group support and then passing ``bypass_check=True`` computes it from
        each member's own support instead — wrong, and silent.
        """
        import numpy as np
        import pynapple as nap

        from aeon.dj_pipeline.utils.codec import _tsgroup_from_npz

        path = tmp_path / "tg.npz"
        mock_tsgroup.save(str(path))
        np.testing.assert_allclose(
            np.asarray(_tsgroup_from_npz(str(path)).rate),
            np.asarray(nap.load_file(str(path)).rate),
        )


class TestPynappleMemberParity:
    """``_to_members`` must match what pynapple's own ``save()`` writes.

    It reads ``_metadata``, a private attribute, and mirrors save-path logic that
    upstream is free to change. This test is the tripwire: it writes a real .npz
    and compares, so the day pynapple alters its layout, this fails rather than
    the rows silently becoming unreadable by ``nap.load_file``.
    """

    @pytest.mark.parametrize("kind", ["Ts", "Tsd", "TsdFrame", "TsdTensor", "IntervalSet", "TsGroup"])
    def test_members_match_pynapple_savez_output(self, kind, tmp_path, mock_tsgroup):
        """Test that member keys and values equal what ``obj.save()`` produces."""
        import numpy as np

        from aeon.dj_pipeline.utils.codec import _to_members

        obj = mock_tsgroup if kind == "TsGroup" else _sample_objects()[kind]
        path = tmp_path / "ref.npz"
        obj.save(path.as_posix())

        with np.load(path, allow_pickle=True) as npz:
            reference = {name: npz[name] for name in npz.files}
        produced = _to_members(obj)

        assert set(produced) == set(reference), f"member keys differ for {kind}"
        for name, expected in reference.items():
            actual = produced[name]
            if expected.dtype == object and expected.shape == ():
                # `_metadata` is a dict of numpy arrays, so compare it entry by entry.
                produced_meta, expected_meta = actual.item(), expected.item()
                assert set(produced_meta) == set(expected_meta), f"{kind}.{name} keys"
                for col, values in expected_meta.items():
                    np.testing.assert_array_equal(
                        np.asarray(produced_meta[col]), np.asarray(values), err_msg=f"{kind}.{name}.{col}"
                    )
            elif expected.dtype == object:
                assert list(actual) == list(expected), f"{kind}.{name}"
            elif expected.dtype.kind == "U":
                assert list(np.asarray(actual).ravel()) == list(expected.ravel()), f"{kind}.{name}"
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=f"{kind}.{name}")

    def test_float_members_are_bitwise_identical(self, mock_tsgroup, tmp_path):
        """Test that float64 members match bit-for-bit, not merely within tolerance.

        The float64 gap at Harp magnitude is 477 ns, so a quantising path would
        shift spikes within a sample and still pass allclose.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.codec import _to_members

        path = tmp_path / "ref.npz"
        mock_tsgroup.save(path.as_posix())
        with np.load(path, allow_pickle=True) as npz:
            reference = {name: npz[name] for name in npz.files}
        produced = _to_members(mock_tsgroup)

        for name, expected in reference.items():
            if expected.dtype.kind == "f":
                assert np.array_equal(
                    np.ascontiguousarray(produced[name]).view("u8"),
                    np.ascontiguousarray(expected).view("u8"),
                ), f"{name} is not bitwise identical"
