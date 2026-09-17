"""Integration tests for codecs.py.

Codec encode/decode tests require real datajoint and run on testcontainers MySQL
(see ``mysql_container`` / ``dj_config_integration``).

The XArrayNetCDFCodec tests additionally use a throwaway schema with a single
``<xarray@xarray_store>`` column, backed by a temp-dir file store, so no real CEPH
path is touched.

Note: ``datajoint.json`` is untracked, so CI has no ``xarray_store`` on disk — the
``xarray_store`` fixture registers it in ``dj.config`` (pointing at ``tmp_path``)
rather than relying on the file.
"""

import numpy as np
import pytest
import xarray as xr
from ephys_factories import (
    make_synthetic_bno055_data,
    make_synthetic_ephys_epoch,
    register_synthetic_experiment,
)

pytestmark = pytest.mark.integration


class TestAeonStreamCodecEncode:
    """Direct ``encode()`` calls against AeonStreamCodec, no DB round trip."""

    def test_valid_encode(self, dj_config_integration):
        """Test that encode returns the input dict unchanged when all required keys are present."""
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        value = {
            "stream_type": "Encoder",
            "experiment_name": "test",
            "device_name": "Feeder1",
            "chunk_start": "2025-01-01 00:00:00",
            "chunk_end": "2025-01-01 01:00:00",
            "epoch_start": "2025-01-01 00:00:00",
        }
        result = codec.encode(value)
        assert result == value

    def test_missing_keys_raises(self, dj_config_integration):
        """Test that encode raises ValueError when required keys are missing."""
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"stream_type": "Encoder"})

    def test_non_dict_raises(self, dj_config_integration):
        """Test that encode raises TypeError for non-dict input."""
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(TypeError, match="expects a dict"):
            codec.encode("not a dict")


class TestOnixStreamCodecRoundTrip:
    """Insert through the codec via a real ephys pipeline, fetch back, and check what comes out."""

    def test_round_trip_returns_onix_indexed_dataframe(self, dj_config_integration, tmp_path):
        """Test that fetch returns a DataFrame merged and indexed by ONIX clock, not HARP datetimes."""
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        experiment_name = "test_codec_round_trip"
        epoch_dir_name = "2024-06-11T10-24-07"
        device_name = "NeuropixelsV2Beta"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

        key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        assert df.index.dtype == np.uint64  # ONIX-indexed (uint64), not HARP datetimes
        # Bno055 chunks are staggered against HarpSync windows in the synthetic
        # factory; the only sync window owns all of the chunk's 100 samples,
        # including those outside its HarpSync rows.
        assert len(df) == 100

    def test_round_trip_no_data_returns_empty_dataframe(self, dj_config_integration, tmp_path):
        """Test that fetch returns an empty IMU_COLUMNS DataFrame when no IMU data exists."""
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        experiment_name = "test_codec_no_data"
        epoch_dir_name = "2024-06-12T10-24-07"
        device_name = "NeuropixelsV2Beta"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        # NO make_synthetic_bno055_data — no IMU files
        register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

        key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        assert len(df) == 0


@pytest.fixture
def xarray_store(dj_config_integration, tmp_path, monkeypatch):
    """Point the ``xarray_store`` file store at a temp dir (created up-front).

    ``_get_backend`` raises ``FileNotFoundError`` if the location doesn't already
    exist, so the directory is made before any insert.
    """
    import datajoint as dj

    location = tmp_path / "xarray_store"
    location.mkdir()
    entry = {"protocol": "file", "location": str(location)}
    monkeypatch.setitem(dj.config["stores"], "xarray_store", entry)
    return location


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


class TestXArrayNetCDFCodecRoundTrip:
    """Insert through the codec, fetch back, and check what comes out."""

    def test_round_trip_returns_lazy_equal_dataset(self, mock_xarray_table, mock_xarray_dataset):
        """Test that fetch is lazy (xarray indexing, no dask) but equal once loaded."""
        mock_xarray, _schema, _loc = mock_xarray_table
        mock_xarray.insert1({"rec_id": 1, "data": mock_xarray_dataset})
        fetched = (mock_xarray & {"rec_id": 1}).fetch1("data")
        assert fetched["signal"].chunks is None  # chunks=None, no dask
        assert fetched["signal"].variable._in_memory is False  # lazy until accessed
        xr.testing.assert_equal(fetched.load(), mock_xarray_dataset)
        assert fetched["signal"].variable._in_memory is True

    def test_real_file_at_schema_addressed_tokened_path(self, mock_xarray_table, mock_xarray_dataset):
        """Test that insert writes one tokened ``.nc`` file under a schema-addressed path."""
        mock_xarray, _schema, loc = mock_xarray_table
        mock_xarray.insert1({"rec_id": 7, "data": mock_xarray_dataset})
        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 1
        assert "rec_id=7" in files[0].as_posix()  # path mirrors schema structure

    def test_two_rows_two_files(self, mock_xarray_table, mock_xarray_dataset):
        """Test that two inserts produce two distinct, tokened files."""
        mock_xarray, _schema, loc = mock_xarray_table
        mock_xarray.insert(
            [{"rec_id": 1, "data": mock_xarray_dataset}, {"rec_id": 2, "data": mock_xarray_dataset}]
        )
        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 2
        assert len({f.name for f in files}) == 2  # distinct tokened files


def collector(schema):
    """A GarbageCollector scoped to `schema` and the temp-dir xarray_store."""
    import datajoint as dj

    return dj.gc.GarbageCollector(schema, store="xarray_store")


class TestXArrayNetCDFCodecGarbageCollection:
    """``dj.gc`` over ``<xarray@>`` columns.

    Deleting a row leaves its ``.nc`` on disk; garbage collection is what reclaims
    it. ``collect()`` diffs the files stored under the schema's section of the
    store against the paths live rows reference — the latter discovered through
    the codec's own ``SchemaCodec.referenced_paths``, which reads the stored JSON
    metadata without opening any file.

    Since a wrong answer here means deleting live data, every test that runs a
    real collect asserts the surviving row is still *fetchable* afterwards, not
    merely that some file is still on disk.
    """

    def test_live_rows_leave_nothing_orphaned(self, mock_xarray_table, mock_xarray_dataset):
        """Test that every file backing a live row is discovered as referenced."""
        mock_xarray, schema, _loc = mock_xarray_table
        mock_xarray.insert(
            [{"rec_id": 1, "data": mock_xarray_dataset}, {"rec_id": 2, "data": mock_xarray_dataset}]
        )
        stats = collector(schema).collect(dry_run=True)
        assert stats["schema_paths_referenced"] == 2
        assert stats["schema_paths_orphaned"] == 0

    @pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "real_run"])
    def test_collect_orphan(self, mock_xarray_table, mock_xarray_dataset, dry_run):
        """Test that ``dry_run`` reports the orphan without deleting; a real run reclaims only it."""
        mock_xarray, schema, loc = mock_xarray_table
        mock_xarray.insert(
            [{"rec_id": 1, "data": mock_xarray_dataset}, {"rec_id": 2, "data": mock_xarray_dataset}]
        )
        (mock_xarray & {"rec_id": 2}).delete()
        stats = collector(schema).collect(dry_run=dry_run)
        remaining = list(loc.rglob("data_*.nc"))
        assert stats["schema_paths_deleted"] == (0 if dry_run else 1)
        assert len(remaining) == (2 if dry_run else 1)  # nothing removed on a dry run
        assert stats["errors"] == 0
        if dry_run:
            assert (stats["schema_paths_referenced"], stats["schema_paths_orphaned"]) == (1, 1)
            assert stats["bytes_freed"] == 0
            # the deleted row's file, not the live one
            assert "rec_id=2" in stats["orphaned_schema_paths"][0]
        else:
            assert stats["bytes_freed"] > 0
            assert "rec_id=1" in remaining[0].as_posix()
            # the live row must still be readable — not merely present on disk
            fetched = (mock_xarray & {"rec_id": 1}).fetch1("data")
            xr.testing.assert_equal(fetched.load(), mock_xarray_dataset)

    def test_collect_is_idempotent(self, mock_xarray_table, mock_xarray_dataset):
        """Test that a second pass finds nothing: the first reclaimed exactly the orphans."""
        mock_xarray, schema, _loc = mock_xarray_table
        mock_xarray.insert1({"rec_id": 1, "data": mock_xarray_dataset})
        mock_xarray.insert1({"rec_id": 2, "data": mock_xarray_dataset})
        (mock_xarray & {"rec_id": 2}).delete()
        collector(schema).collect(dry_run=False)

        stats = collector(schema).collect(dry_run=False)

        assert stats["schema_paths_orphaned"] == 0
        assert stats["schema_paths_deleted"] == 0
        fetched = (mock_xarray & {"rec_id": 1}).fetch1("data")
        xr.testing.assert_equal(fetched.load(), mock_xarray_dataset)


@pytest.fixture
def pynapple_store(dj_config_integration, tmp_path, monkeypatch):
    """Point a ``pynapple_store`` file store at a temp dir (created up-front).

    ``_get_backend`` raises ``FileNotFoundError`` if the location doesn't already
    exist, so the directory is made before any insert.
    """
    import datajoint as dj

    location = tmp_path / "pynapple_store"
    location.mkdir()
    monkeypatch.setitem(
        dj.config["stores"], "pynapple_store", {"protocol": "file", "location": str(location)}
    )
    return location


@pytest.fixture
def mock_pynapple_table(pynapple_store):
    """Throwaway schema + table with one ``<pynapple@pynapple_store>`` column."""
    import datajoint as dj

    from aeon.dj_pipeline import get_schema_name  # importing also registers the codec

    schema = dj.Schema(get_schema_name("test_pynapple_codec"))

    @schema
    class MockPynapple(dj.Manual):
        definition = """
        rec_id : int
        ---
        data : <pynapple@pynapple_store>
        """

    yield MockPynapple, schema, pynapple_store
    schema.drop()


def assert_tsgroup_equal(a, b):
    """Assert two TsGroups are indistinguishable: keys, times, support, metadata."""
    assert list(a.index) == list(b.index)
    for unit in a.index:
        np.testing.assert_array_equal(a[unit].t, b[unit].t)
    np.testing.assert_array_equal(a.time_support.values, b.time_support.values)
    for col in a.metadata.columns:
        np.testing.assert_array_equal(np.asarray(a.get_info(col)), np.asarray(b.get_info(col)))


class TestPynappleCodecRoundTrip:
    """Insert through the codec, fetch back, and check what comes out."""

    def test_round_trip_returns_equal_tsgroup(self, mock_pynapple_table, mock_tsgroup):
        """Test that a DB round trip preserves keys, times, support and metadata."""
        table, _schema, _loc = mock_pynapple_table
        table.insert1({"rec_id": 1, "data": mock_tsgroup})
        assert_tsgroup_equal((table & {"rec_id": 1}).fetch1("data"), mock_tsgroup)

    def test_real_file_at_schema_addressed_tokened_path(self, mock_pynapple_table, mock_tsgroup):
        """Test that insert writes one tokened ``.npz`` under a schema-addressed path."""
        table, _schema, loc = mock_pynapple_table
        table.insert1({"rec_id": 7, "data": mock_tsgroup})
        files = list(loc.rglob("data_*.npz"))
        assert len(files) == 1
        assert "rec_id=7" in files[0].as_posix()

    def test_json_summary_is_queryable_without_opening_the_file(self, mock_pynapple_table, mock_tsgroup):
        """Test that the stored JSON carries the summary the spec promises.

        On MariaDB the JSON type aliases to longtext, so this also exercises the
        dict-to-JSON pymysql patch in ``aeon/dj_pipeline/__init__.py``.
        """
        import json

        table, _schema, _loc = mock_pynapple_table
        table.insert1({"rec_id": 1, "data": mock_tsgroup})

        # Read the column straight from SQL. Any DataJoint fetch — including
        # ``proj()`` — runs the codec and hands back a TsGroup, so this is the only
        # way to assert what is actually stored.
        cursor = table.connection.query(
            # S608: the table name is generated by DataJoint, not user input, and a
            # table name cannot be a bound parameter.
            f"SELECT `data` FROM {table.full_table_name} WHERE rec_id = 1"  # noqa: S608
        )
        raw = cursor.fetchone()[0]
        meta = json.loads(raw) if isinstance(raw, str | bytes | bytearray) else raw

        assert meta["kind"] == "TsGroup"
        assert meta["n_rows"] == 3
        assert meta["t_start"] > 3.0e9
        assert meta["path"].endswith(".npz")

    def test_insert_rejects_non_pynapple(self, mock_pynapple_table):
        """Test that ``validate`` fires on insert, not at fetch time."""
        import datajoint as dj

        table, _schema, _loc = mock_pynapple_table
        with pytest.raises(dj.errors.DataJointError, match="requires a pynapple object"):
            table.insert1({"rec_id": 99, "data": [1, 2, 3]})


def pynapple_collector(schema):
    """A GarbageCollector scoped to `schema` and the temp-dir pynapple_store."""
    import datajoint as dj

    return dj.gc.GarbageCollector(schema, store="pynapple_store")


class TestPynappleCodecGarbageCollection:
    """``dj.gc`` over ``<pynapple@>`` columns.

    ``SchemaCodec.referenced_paths`` finds the files live rows depend on by reading
    the stored JSON, which only works because ``encode`` returns ``path`` and
    ``store`` at the top level. These tests are what prove that.

    Every test that runs a real collect re-fetches the surviving row, because the
    failure that matters is silent deletion of live data — not a missing file.
    """

    def test_live_rows_leave_nothing_orphaned(self, mock_pynapple_table, mock_tsgroup):
        """Test that every file backing a live row is discovered as referenced."""
        table, schema, _loc = mock_pynapple_table
        table.insert([{"rec_id": 1, "data": mock_tsgroup}, {"rec_id": 2, "data": mock_tsgroup}])
        stats = pynapple_collector(schema).collect(dry_run=True)
        assert stats["schema_paths_referenced"] == 2
        assert stats["schema_paths_orphaned"] == 0

    @pytest.mark.parametrize("dry_run", [True, False], ids=["dry_run", "real_run"])
    def test_collect_orphan(self, mock_pynapple_table, mock_tsgroup, dry_run):
        """Test that a dry run reports without deleting; a real run reclaims only the orphan."""
        table, schema, loc = mock_pynapple_table
        table.insert([{"rec_id": 1, "data": mock_tsgroup}, {"rec_id": 2, "data": mock_tsgroup}])
        (table & {"rec_id": 2}).delete()

        stats = pynapple_collector(schema).collect(dry_run=dry_run)
        remaining = list(loc.rglob("data_*.npz"))

        assert stats["schema_paths_deleted"] == (0 if dry_run else 1)
        assert len(remaining) == (2 if dry_run else 1)  # nothing removed on a dry run
        assert stats["errors"] == 0
        if dry_run:
            assert (stats["schema_paths_referenced"], stats["schema_paths_orphaned"]) == (1, 1)
            assert stats["bytes_freed"] == 0
            assert "rec_id=2" in stats["orphaned_schema_paths"][0]  # the deleted row's file
        else:
            assert stats["bytes_freed"] > 0
            assert "rec_id=1" in remaining[0].as_posix()
            # the live row must still be readable — not merely present on disk
            assert_tsgroup_equal((table & {"rec_id": 1}).fetch1("data"), mock_tsgroup)

    def test_collect_is_idempotent(self, mock_pynapple_table, mock_tsgroup):
        """Test that a second pass finds nothing and the survivor is still readable."""
        table, schema, _loc = mock_pynapple_table
        table.insert([{"rec_id": 1, "data": mock_tsgroup}, {"rec_id": 2, "data": mock_tsgroup}])
        (table & {"rec_id": 2}).delete()
        pynapple_collector(schema).collect(dry_run=False)

        stats = pynapple_collector(schema).collect(dry_run=False)

        assert stats["schema_paths_orphaned"] == 0
        assert stats["schema_paths_deleted"] == 0
        assert_tsgroup_equal((table & {"rec_id": 1}).fetch1("data"), mock_tsgroup)
