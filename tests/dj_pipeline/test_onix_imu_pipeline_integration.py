"""Integration tests for the ONIX IMU pipeline (EphysSyncModel, EphysChunk, OnixImuChunk)."""

import csv
import logging
from pathlib import Path

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
