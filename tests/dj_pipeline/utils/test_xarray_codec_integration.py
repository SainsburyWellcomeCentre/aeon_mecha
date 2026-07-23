"""Integration tests for XArrayNetCDFCodec against a real DataJoint table.

Runs on testcontainers MySQL (see ``mysql_container`` / ``dj_config_integration``).
A throwaway schema with a single ``<xarray@xarray_store>`` column is backed by a
temp-dir file store, so no real CEPH path is touched.

Note: ``datajoint.json`` is untracked, so CI has no ``xarray_store`` on disk — the
``xarray_store`` fixture registers it in ``dj.config`` (pointing at ``tmp_path``)
rather than relying on the file.
"""

import pytest
import xarray as xr

pytestmark = pytest.mark.integration


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


class TestValidate:
    def test_accepts_dataset_rejects_others(self, dj_config_integration, mock_xarray_dataset):
        from datajoint.errors import DataJointError

        from aeon.dj_pipeline.utils.xarray_codec import XArrayNetCDFCodec

        codec = XArrayNetCDFCodec()
        mock = mock_xarray_dataset
        codec.validate(mock)  # a Dataset must not raise

        with pytest.raises(DataJointError, match="requires an xarray.Dataset"):
            codec.validate([1, 2, 3])

        # a DataArray is rejected — caller must convert (avoids store-DataArray/get-Dataset)
        with pytest.raises(DataJointError, match=r"got DataArray.*to_dataset"):
            codec.validate(mock["signal"])


class TestDBRoundTrip:
    def test_round_trip_returns_lazy_equal_dataset(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, _schema, _loc = mock_xarray_table
        mock = mock_xarray_dataset

        mock_xarray.insert1({"rec_id": 1, "data": mock})
        got = (mock_xarray & {"rec_id": 1}).fetch1("data")

        assert isinstance(got, xr.Dataset)
        assert got["signal"].variable._in_memory is False  # lazy until accessed
        xr.testing.assert_equal(got.load(), mock)

    def test_real_file_at_schema_addressed_tokened_path(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, _schema, loc = mock_xarray_table

        mock_xarray.insert1({"rec_id": 7, "data": mock_xarray_dataset})

        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 1
        assert files[0].is_file()
        assert "rec_id=7" in files[0].as_posix()  # path mirrors schema structure
        assert files[0].name != "data.nc"  # filename carries a random token

    def test_fetch_is_lazy_without_dask(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, _schema, _loc = mock_xarray_table

        mock_xarray.insert1({"rec_id": 1, "data": mock_xarray_dataset})
        got = (mock_xarray & {"rec_id": 1}).fetch1("data")

        assert got["signal"].chunks is None  # chunks=None, no dask
        assert got["signal"].variable._in_memory is False
        got["signal"].load()
        assert got["signal"].variable._in_memory is True

    def test_two_rows_two_files(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, _schema, loc = mock_xarray_table
        ds1 = ds2 = mock_xarray_dataset

        mock_xarray.insert([{"rec_id": 1, "data": ds1}, {"rec_id": 2, "data": ds2}])

        files = list(loc.rglob("data_*.nc"))
        assert len(files) == 2
        assert len({f.name for f in files}) == 2  # distinct tokened files
        xr.testing.assert_equal((mock_xarray & {"rec_id": 1}).fetch1("data").load(), ds1)
        xr.testing.assert_equal((mock_xarray & {"rec_id": 2}).fetch1("data").load(), ds2)


def collector(schema):
    """A GarbageCollector scoped to `schema` and the temp-dir xarray_store."""
    import datajoint as dj

    return dj.gc.GarbageCollector(schema, store="xarray_store")


class TestGarbageCollection:
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
        """Every file backing a live row is discovered as referenced."""
        mock_xarray, schema, _loc = mock_xarray_table
        mock_xarray.insert(
            [{"rec_id": 1, "data": mock_xarray_dataset}, {"rec_id": 2, "data": mock_xarray_dataset}]
        )

        stats = collector(schema).collect(dry_run=True)

        assert stats["schema_paths_referenced"] == 2
        assert stats["schema_paths_orphaned"] == 0

    def test_dry_run_reports_the_orphan_but_deletes_nothing(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, schema, loc = mock_xarray_table
        mock_xarray.insert(
            [{"rec_id": 1, "data": mock_xarray_dataset}, {"rec_id": 2, "data": mock_xarray_dataset}]
        )
        (mock_xarray & {"rec_id": 2}).delete()
        assert len(list(loc.rglob("data_*.nc"))) == 2  # row delete leaves the file on disk

        stats = collector(schema).collect(dry_run=True)

        assert (stats["schema_paths_referenced"], stats["schema_paths_orphaned"]) == (1, 1)
        assert stats["dry_run"] is True
        assert stats["schema_paths_deleted"] == 0
        assert stats["bytes_freed"] == 0
        assert "rec_id=2" in stats["orphaned_schema_paths"][0]  # the deleted row's file, not the live one
        assert len(list(loc.rglob("data_*.nc"))) == 2  # nothing removed

    def test_collect_reclaims_only_the_orphaned_file(self, mock_xarray_table, mock_xarray_dataset):
        mock_xarray, schema, loc = mock_xarray_table
        ds1 = mock_xarray_dataset
        mock_xarray.insert([{"rec_id": 1, "data": ds1}, {"rec_id": 2, "data": mock_xarray_dataset}])
        (mock_xarray & {"rec_id": 2}).delete()

        stats = collector(schema).collect(dry_run=False)

        assert stats["schema_paths_deleted"] == 1
        assert stats["errors"] == 0
        assert stats["bytes_freed"] > 0
        survivors = list(loc.rglob("data_*.nc"))
        assert len(survivors) == 1
        assert "rec_id=1" in survivors[0].as_posix()
        # the live row must still be readable — not merely present on disk
        xr.testing.assert_equal((mock_xarray & {"rec_id": 1}).fetch1("data").load(), ds1)

    def test_collect_is_idempotent(self, mock_xarray_table, mock_xarray_dataset):
        """A second pass finds nothing: the first reclaimed exactly the orphans."""
        mock_xarray, schema, _loc = mock_xarray_table
        ds1 = mock_xarray_dataset
        mock_xarray.insert1({"rec_id": 1, "data": ds1})
        mock_xarray.insert1({"rec_id": 2, "data": mock_xarray_dataset})
        (mock_xarray & {"rec_id": 2}).delete()
        collector(schema).collect(dry_run=False)

        stats = collector(schema).collect(dry_run=False)

        assert stats["schema_paths_orphaned"] == 0
        assert stats["schema_paths_deleted"] == 0
        xr.testing.assert_equal((mock_xarray & {"rec_id": 1}).fetch1("data").load(), ds1)
