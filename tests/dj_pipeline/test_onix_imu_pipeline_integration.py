"""Integration tests for the ONIX IMU pipeline (EphysSyncModel, EphysChunk, OnixImuChunk)."""

import logging

import pytest
from _synthetic_ephys_fixtures import (
    _make_synthetic_amplifier_data,
    _make_synthetic_bno055_data,
    _make_synthetic_ephys_epoch,
    _register_synthetic_experiment,
    _register_synthetic_probe_insertion,
)

logger = logging.getLogger(__name__)

# Auto-marked as integration via fixture usage; explicit marker for clarity:
pytestmark = pytest.mark.integration


# ============================================================================
# Task 4: Table existence
# ============================================================================


def test_ephys_sync_model_table_exists(dj_config_integration):
    """EphysSyncModel can be activated against an integration DB."""
    from aeon.dj_pipeline import ephys

    assert hasattr(ephys, "EphysSyncModel")
    table = ephys.EphysSyncModel()
    assert "experiment_name" in table.primary_key
    assert "epoch_start" in table.primary_key
    assert "sync_start" in table.primary_key

    attrs = set(table.heading.attributes.keys())
    expected_attrs = {"sync_end", "onix_ts_start", "onix_ts_end", "sync_model", "r2", "n_samples"}
    assert expected_attrs <= attrs, f"Missing: {expected_attrs - attrs}"


# ============================================================================
# Task 5: ingest classmethod
# ============================================================================


def test_ephys_sync_model_ingest_inserts_one_row_per_csv(dj_config_integration, tmp_path):
    """ingest() walks HarpSync CSVs and inserts one row per CSV."""
    from aeon.dj_pipeline import ephys

    experiment_name = "test_ephys_sync_ingest"
    epoch_dir_name = "2024-06-04T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=3)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)

    # Project only non-attach columns to avoid triggering attachment extraction to cwd
    query = (ephys.EphysSyncModel & {"experiment_name": experiment_name}).proj(
        "n_samples", "r2", "onix_ts_start", "onix_ts_end", "sync_start", "sync_end"
    )
    rows = query.to_dicts()
    assert len(rows) == 3
    for row in rows:
        assert row["n_samples"] == 60
        assert 0.0 <= row["r2"] <= 1.0
        assert row["onix_ts_end"] > row["onix_ts_start"]
        assert row["sync_end"] > row["sync_start"]


def test_ephys_sync_model_ingest_is_idempotent(dj_config_integration, tmp_path):
    """Re-running ingest doesn't insert duplicate rows."""
    from aeon.dj_pipeline import ephys

    experiment_name = "test_ephys_sync_idempotent"
    epoch_dir_name = "2024-06-05T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    initial_count = len(ephys.EphysSyncModel & {"experiment_name": experiment_name})
    assert initial_count == 2

    ephys.EphysSyncModel.ingest(experiment_name)
    assert len(ephys.EphysSyncModel & {"experiment_name": experiment_name}) == initial_count


# ============================================================================
# Task 6: EphysChunk.SyncModel Part is link-only FK
# ============================================================================


def test_ephys_chunk_sync_model_part_is_link_only(dj_config_integration):
    """EphysChunk.SyncModel exposes only FK columns, no attach or onix bounds."""
    from aeon.dj_pipeline import ephys

    sync_part = ephys.EphysChunk.SyncModel()
    attrs = set(sync_part.heading.attributes.keys())

    # FK to master (EphysChunk PK) + EphysSyncModel
    expected = {
        "experiment_name",
        "subject",
        "insertion_number",
        "chunk_start",
        "epoch_start",
        "sync_start",
    }
    assert expected <= attrs, f"Missing FK columns: {expected - attrs}"

    # No more attach / onix bounds / harp_start
    assert "sync_model" not in attrs
    assert "onix_ts_start" not in attrs
    assert "onix_ts_end" not in attrs
    assert "harp_start" not in attrs


# ============================================================================
# Task 7: EphysChunk.ingest_chunks uses DB-backed EphysSyncModel lookup
# ============================================================================


def test_ephys_chunk_ingest_uses_sync_model_from_db(dj_config_integration, tmp_path):
    """EphysChunk.ingest_chunks creates link Part rows referencing EphysSyncModel rows."""
    from aeon.dj_pipeline import acquisition, ephys
    from aeon.dj_pipeline import subject as subj_mod

    experiment_name = "test_ephys_chunk_db_lookup"
    epoch_dir_name = "2024-06-06T10-24-07"
    device_name = "NeuropixelsV2Beta"
    probe_label = "ProbeA"
    subject = "test-mouse-ephys"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Write HarpSync CSVs and AmplifierData + Clock binaries
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=3)
    _make_synthetic_amplifier_data(raw_dir, epoch_dir_name, device_name, probe_label, n_chunks=3)

    # Register experiment (inserts Lab, Arena, DevicesSchema, Experiment, Epoch, EphysEpoch)
    epoch_start = _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    # Register subject and link to experiment
    subj_mod.Subject.insert1(
        {"subject": subject, "sex": "U", "subject_birth_date": "2024-01-01"},
        skip_duplicates=True,
    )
    acquisition.Experiment.Subject.insert1(
        {"experiment_name": experiment_name, "subject": subject},
        skip_duplicates=True,
    )

    _register_synthetic_probe_insertion(experiment_name, subject, epoch_start, probe_label, device_name)

    # Ingest SyncModel rows first, then chunks
    ephys.EphysSyncModel.ingest(experiment_name)
    sm_count = len(ephys.EphysSyncModel & {"experiment_name": experiment_name})
    assert sm_count == 3, f"Expected 3 SyncModel rows, got {sm_count}"

    ephys.EphysChunk.ingest_chunks(experiment_name)

    chunk_rows = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts()
    assert len(chunk_rows) == 3, f"Expected 3 chunks, got {len(chunk_rows)}"

    # Every chunk must have at least one SyncModel link row
    link_rows = (ephys.EphysChunk.SyncModel & {"experiment_name": experiment_name}).to_dicts()
    assert len(link_rows) >= 3, f"Expected >= 3 link rows, got {len(link_rows)}"

    # All link rows must reference existing EphysSyncModel sync_starts
    sm_sync_starts = set(
        (ephys.EphysSyncModel & {"experiment_name": experiment_name}).to_arrays("sync_start")
    )
    link_sync_starts = {r["sync_start"] for r in link_rows}
    assert link_sync_starts <= sm_sync_starts, (
        f"Link rows reference unknown sync_starts: {link_sync_starts - sm_sync_starts}"
    )

    # Idempotency: re-running ingest_chunks must not insert duplicate rows
    ephys.EphysChunk.ingest_chunks(experiment_name)
    assert len(ephys.EphysChunk & {"experiment_name": experiment_name}) == 3


# ============================================================================
# Task 8: OnixImuChunk table definition
# ============================================================================


def test_onix_imu_chunk_table_shape(dj_config_integration):
    """OnixImuChunk has 13 IMU column attrs + sample_count + timestamps + stream_df."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    table = ephys.OnixImuChunk()
    attrs = set(table.heading.attributes.keys())

    assert {"experiment_name", "epoch_start", "sync_start"} <= set(table.primary_key)
    assert {"sample_count", "timestamps", "stream_df"} <= attrs
    for col in IMU_COLUMNS:
        assert col in attrs, f"Missing per-column stat field: {col}"


# ============================================================================
# Task 9: OnixImuChunk.populate tests
# ============================================================================


def test_onix_imu_chunk_populate_with_data(dj_config_integration, tmp_path):
    """OnixImuChunk.populate() ingests Bno055 streams when files exist."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    experiment_name = "test_onix_imu_with_data"
    epoch_dir_name = "2024-06-07T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).proj().to_dicts()
    assert len(rows) == 2

    full_rows = (
        (ephys.OnixImuChunk & {"experiment_name": experiment_name})
        .proj("sample_count", "timestamps", *IMU_COLUMNS)
        .to_dicts()
    )
    for r in full_rows:
        assert r["sample_count"] == 100
        assert isinstance(r["timestamps"], dict)
        for col in IMU_COLUMNS:
            assert isinstance(r[col], dict)


def test_onix_imu_chunk_populate_no_imu_rig(dj_config_integration, tmp_path):
    """When Bno055 files are absent, OnixImuChunk inserts an empty row."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    experiment_name = "test_onix_imu_no_data"
    epoch_dir_name = "2024-06-08T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # HarpSync CSVs but NO Bno055 binaries
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    full_rows = (
        (ephys.OnixImuChunk & {"experiment_name": experiment_name})
        .proj("sample_count", "timestamps", *IMU_COLUMNS)
        .to_dicts()
    )
    assert len(full_rows) == 2
    for r in full_rows:
        assert r["sample_count"] == 0
        assert r["timestamps"] == {}
        for col in IMU_COLUMNS:
            assert r[col] == {}


# ============================================================================
# Task 10: OnixImuChunk.synced_df classmethod
# ============================================================================


def test_synced_df_returns_harp_indexed_dataframe(dj_config_integration, tmp_path):
    """synced_df fetches stream_df and applies the SyncModel regression."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    experiment_name = "test_onix_imu_synced"
    epoch_dir_name = "2024-06-09T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=1)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
    df = ephys.OnixImuChunk.synced_df(key)

    assert tuple(df.columns) == IMU_COLUMNS
    assert len(df) == 100
    # HARP-indexed → datetime dtype, NOT uint64
    assert df.index.dtype.kind == "M"
    assert df.index.tz is not None  # UTC-aware per spec
    assert str(df.index.tz) == "UTC"


def test_synced_df_raises_on_ambiguous_key(dj_config_integration, tmp_path):
    """A non-PK restriction matching multiple rows raises (via fetch1)."""
    import datajoint as dj

    from aeon.dj_pipeline import ephys

    experiment_name = "test_onix_imu_ambiguous"
    epoch_dir_name = "2024-06-10T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    # Ambiguous key — matches 2 rows
    with pytest.raises(dj.errors.DataJointError):
        ephys.OnixImuChunk.synced_df({"experiment_name": experiment_name})
