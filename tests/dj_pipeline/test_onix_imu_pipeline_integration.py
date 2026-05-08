"""Integration tests for the ONIX IMU pipeline (EphysSyncModel, EphysChunk, OnixImuChunk)."""

import csv
import logging
import uuid
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

# Auto-marked as integration via fixture usage; explicit marker for clarity:
pytestmark = pytest.mark.integration


# ============================================================================
# Fixture helpers (used by Task 5 ingest tests)
# ============================================================================


def _make_synthetic_ephys_epoch(
    raw_dir: Path,
    experiment_name: str,
    epoch_dir_name: str,
    device_name: str,
    n_chunks: int,
):
    """Create an epoch directory with ``n_chunks`` synthetic HarpSync CSVs.

    Each CSV has 60 rows (one per second). ONIX clock advances at 1000 ticks/sec;
    HARP time advances at 1 sec increments. CSV filenames carry hourly HARP
    timestamps, but actual content covers an hour-long window.
    """
    epoch_dir = raw_dir / epoch_dir_name
    device_dir = epoch_dir / device_name
    device_dir.mkdir(parents=True, exist_ok=True)

    # HARP epoch base: arbitrary seconds-since-1904 for plausible wall-clock times.
    harp_base = 3000.0

    for n in range(n_chunks):
        ts_str = f"2024-06-04T1{n}-00-00"
        csv_path = device_dir / f"{device_name}_HarpSync_{ts_str}.csv"
        rows = []
        for s in range(60):
            rows.append(
                {
                    "clock": 1000 * (n * 60 + s) + 1,
                    "hub_clock": s,
                    "harp_time": harp_base + n * 60 + s,
                }
            )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clock", "hub_clock", "harp_time"])
            writer.writeheader()
            writer.writerows(rows)


def _register_synthetic_experiment(
    tmp_path: Path,
    raw_dir: Path,
    experiment_name: str,
    epoch_dir_name: str,
):
    """Register the minimum fixtures needed for EphysSyncModel.ingest() to work.

    Inserts: lab.Arena, acquisition.PipelineRepository, acquisition.DevicesSchema,
    acquisition.Experiment, acquisition.Experiment.Directory pointing at raw_dir,
    acquisition.Epoch, and a minimal ephys.EphysEpoch (has_ephys=True, n_probes=0).

    Returns the epoch_start datetime.
    """
    import aeon.dj_pipeline as _pipeline
    from aeon.dj_pipeline import acquisition, ephys, lab
    from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

    epoch_dt = parse_epoch_timestamp(epoch_dir_name)

    # Register repository: tmp_path is the repo root, raw_dir is tmp_path / "raw".
    # PipelineRepository.repository_name is varchar(16) — use a short fixed key.
    # Each test uses a unique tmp_path so there's no collision.
    repo_key = "test_repo"
    _pipeline.repository_config[repo_key] = str(tmp_path)

    # PipelineRepository is a Lookup table — insert if not present
    acquisition.PipelineRepository.insert1({"repository_name": repo_key}, skip_duplicates=True)

    # Lab fixtures (Lookup tables with pre-populated contents — just ensure present)
    lab.Arena.insert1(
        {
            "arena_name": "synthetic-arena",
            "arena_description": "synthetic test arena",
            "arena_shape": "circular",
            "arena_x_dim": 2.0,
            "arena_y_dim": 2.0,
            "arena_z_dim": 0.2,
        },
        skip_duplicates=True,
    )

    # DevicesSchema
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": "synthetic.schema:Synthetic"},
        skip_duplicates=True,
    )

    # Experiment
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": epoch_dt,
            "experiment_description": "synthetic ephys ingest test",
            "arena_name": "synthetic-arena",
            "lab": "SWC",
            "location": "room-0",
            "experiment_type": "foraging",
        },
        skip_duplicates=True,
    )
    acquisition.Experiment.DevicesSchema.insert1(
        {
            "experiment_name": experiment_name,
            "devices_schema_name": "synthetic.schema:Synthetic",
        },
        skip_duplicates=True,
    )

    # Directory: directory_path is "raw" (relative to tmp_path), so
    # get_data_directory resolves to tmp_path / "raw" = raw_dir.
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": experiment_name,
            "directory_type": "raw",
            "repository_name": repo_key,
            "directory_path": "raw",
        },
        skip_duplicates=True,
    )

    # Epoch — insert directly (skip ingest_epochs which scans camera files)
    acquisition.Epoch.insert1(
        {
            "experiment_name": experiment_name,
            "epoch_start": epoch_dt,
            "directory_type": "raw",
            "repository_name": repo_key,
            "epoch_dir": epoch_dir_name,
        },
        skip_duplicates=True,
        ignore_extra_fields=True,
    )

    # EphysEpoch is dj.Imported — must use allow_direct_insert=True outside of make()
    ephys.EphysEpoch.insert1(
        {
            "experiment_name": experiment_name,
            "epoch_start": epoch_dt,
            "has_ephys": True,
            "n_probes": 0,
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )

    return epoch_dt


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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=3)
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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=2)
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
# Task 7: Helpers for AmplifierData fixtures + probe/insertion registration
# ============================================================================


def _make_synthetic_amplifier_data(
    raw_dir: Path,
    epoch_dir_name: str,
    device_name: str,
    probe_label: str,
    n_chunks: int,
):
    """Write synthetic AmplifierData_N.bin + Clock_N.bin files for each chunk.

    ONIX timestamps in each Clock file fall WITHIN the matching HarpSync CSV's
    clock range so the BETWEEN query in ingest_chunks finds the right SyncModel.

    The synthetic HarpSync CSV for chunk n has:
      clock_start = 1000 * (n * 60) + 1
      clock_end   = 1000 * (n * 60 + 59) + 1  = 60000*n + 59001
    We write Clock timestamps inside those bounds: clock_start+500 to clock_end-500.
    """
    device_dir = raw_dir / epoch_dir_name / device_name
    device_dir.mkdir(parents=True, exist_ok=True)

    # Each AmplifierData file has 10 samples (minimal valid binary).
    n_samples = 10
    n_channels = 4  # minimal channel count (uint16 per sample)

    for n in range(n_chunks):
        # ONIX clock range for this chunk (matches _make_synthetic_ephys_epoch)
        clock_start = 60000 * n + 1
        clock_end = 60000 * n + 59001
        # Keep timestamps strictly inside the SyncModel's ONIX range
        ts_start = clock_start + 500
        ts_end = clock_end - 500
        onix_ts = np.linspace(ts_start, ts_end, n_samples, dtype=np.uint64)

        # Write Clock binary (uint64 ONIX timestamps)
        clock_path = device_dir / f"{device_name}_{probe_label}_Clock_{n}.bin"
        onix_ts.tofile(clock_path)

        # Write AmplifierData binary (uint16, shape n_samples x n_channels, all zeros)
        amp_path = device_dir / f"{device_name}_{probe_label}_AmplifierData_{n}.bin"
        amp_data = np.zeros((n_samples, n_channels), dtype=np.uint16)
        amp_data.tofile(amp_path)


def _register_synthetic_probe_insertion(
    experiment_name: str,
    subject: str,
    epoch_start,
    probe_label: str,
    device_name: str = "NeuropixelsV2Beta",
):
    """Insert ProbeType, ElectrodeConfig, Probe, ProbeInsertion, EphysEpoch.Insertion.

    Uses minimal valid data to satisfy FKs without exercising probe semantics.
    """
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.ephys_utils import DEVICE_PROBE_TYPE_MAP

    probe_type = DEVICE_PROBE_TYPE_MAP[device_name]
    probe_id = f"{device_name}_{probe_label}_test"
    electrode_config_name = "test-config-0"

    # ProbeType (Lookup)
    ephys.ProbeType.insert1({"probe_type": probe_type}, skip_duplicates=True)

    # ProbeType.Electrode — insert one minimal electrode so ElectrodeConfig.Electrode works
    ephys.ProbeType.Electrode.insert1(
        {
            "probe_type": probe_type,
            "electrode": 0,
            "shank": 0,
            "x_coord": 0.0,
            "y_coord": 0.0,
            "electrode_name": "e0",
        },
        skip_duplicates=True,
    )

    # Probe (Lookup)
    ephys.Probe.insert1(
        {"probe": probe_id, "probe_type": probe_type, "probe_comment": "synthetic"},
        skip_duplicates=True,
    )

    # ElectrodeConfig (Lookup)
    config_hash = uuid.uuid5(uuid.NAMESPACE_DNS, f"{probe_type}-{electrode_config_name}")
    ephys.ElectrodeConfig.insert1(
        {
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
            "electrode_config_description": "synthetic test config",
            "electrode_config_hash": config_hash,
        },
        skip_duplicates=True,
    )
    ephys.ElectrodeConfig.Electrode.insert1(
        {
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
            "electrode": 0,
        },
        skip_duplicates=True,
    )

    # ProbeInsertion (Manual) — insertion_number=1
    ephys.ProbeInsertion.insert1(
        {
            "experiment_name": experiment_name,
            "subject": subject,
            "insertion_number": 1,
            "probe": probe_id,
        },
        skip_duplicates=True,
    )

    # EphysEpoch.Insertion (Part, allow_direct_insert)
    ephys.EphysEpoch.Insertion.insert1(
        {
            "experiment_name": experiment_name,
            "epoch_start": epoch_start,
            "subject": subject,
            "insertion_number": 1,
            "probe_label": probe_label,
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )


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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=3)
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
# Task 9: Bno055 fixture helper and OnixImuChunk.populate tests
# ============================================================================


def _make_synthetic_bno055_data(
    raw_dir: Path,
    epoch_dir_name: str,
    device_name: str,
    n_chunks: int,
    samples_per_chunk: int = 100,
):
    """Write synthetic Bno055 Clock + 4 stream binaries.

    Each chunk's Clock_N.bin first sample equals the HarpSync CSV's onix_ts_start
    for that chunk (i.e. ``1000 * n * 60 + 1 = 60000*n + 1``) so that
    ``locate_bno055_chunk_index`` can find each chunk by its first ONIX timestamp.

    Layout matches ``aeon/schema/ephys.py`` Bno055 readers:
    - Bno055_Clock_N.bin: uint64 ONIX timestamps
    - Bno055_Euler_N.bin: float32, 3 columns (x, y, z)
    - Bno055_GravityVector_N.bin: float32, 3 columns
    - Bno055_LinearAcceleration_N.bin: float32, 3 columns
    - Bno055_Quaternion_N.bin: float32, 4 columns (w, x, y, z)
    """
    device_dir = raw_dir / epoch_dir_name / device_name
    device_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for n in range(n_chunks):
        # Must start at exactly onix_ts_start[n] = 60000*n + 1 so
        # locate_bno055_chunk_index finds the match.
        chunk_ts_start = 60000 * n + 1
        chunk_ts_end = 60000 * n + 59001
        clocks = np.linspace(chunk_ts_start, chunk_ts_end, samples_per_chunk, dtype=np.float64).astype(
            np.uint64
        )
        (device_dir / f"{device_name}_Bno055_Clock_{n}.bin").write_bytes(clocks.tobytes())

        for stream, n_cols in [
            ("Euler", 3),
            ("GravityVector", 3),
            ("LinearAcceleration", 3),
            ("Quaternion", 4),
        ]:
            data = rng.standard_normal((samples_per_chunk, n_cols)).astype(np.float32)
            (device_dir / f"{device_name}_Bno055_{stream}_{n}.bin").write_bytes(data.tobytes())


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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=2)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).proj().to_dicts()
    assert len(rows) == 2

    full_rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).proj(
        "sample_count", "timestamps", *IMU_COLUMNS
    ).to_dicts()
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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    full_rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).proj(
        "sample_count", "timestamps", *IMU_COLUMNS
    ).to_dicts()
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
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=1)
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


def test_synced_df_raises_on_ambiguous_key(dj_config_integration, tmp_path):
    """A non-PK restriction matching multiple rows raises (via fetch1)."""
    import datajoint as dj

    from aeon.dj_pipeline import ephys

    experiment_name = "test_onix_imu_ambiguous"
    epoch_dir_name = "2024-06-10T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=2)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=2)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    # Ambiguous key — matches 2 rows
    with pytest.raises(dj.errors.DataJointError):
        ephys.OnixImuChunk.synced_df({"experiment_name": experiment_name})
