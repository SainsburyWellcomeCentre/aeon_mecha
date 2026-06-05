"""Synthetic data factory helpers for ONIX ephys pipeline integration tests.

Imported by test_onix_imu_pipeline_integration.py and utils/test_codec_integration.py.
"""

import csv
import uuid
from pathlib import Path

import numpy as np


def make_synthetic_ephys_epoch(
    raw_dir: Path,
    epoch_dir_name: str,
    device_name: str,
    n_chunks: int,
):
    """Create an epoch directory with ``n_chunks`` synthetic HarpSync CSVs.

    Each CSV has 60 rows (one per second). ONIX clock advances at 1000 ticks/sec;
    HARP time advances at 1 sec increments. CSV filenames carry hourly HARP
    timestamps, but actual content covers an hour-long window.

    Also writes a minimal Metadata.yml with per-probe ProbeInterfaceFileName
    entries (required by EphysChunk.ingest_chunks's per-epoch config lookup).
    """
    import json as _json

    epoch_dir = raw_dir / epoch_dir_name
    device_dir = epoch_dir / device_name
    device_dir.mkdir(parents=True, exist_ok=True)

    # Minimal Metadata.yml — Devices.NeuropixelsV2e.ConfigurationA.ProbeInterfaceFileName
    # matches the synthetic ElectrodeConfig.config_file_name from
    # register_synthetic_probe_insertion ("test-config-0.json").
    metadata = {
        "Devices": {
            "NeuropixelsV2e": {
                "DeviceName": device_name,
                "ConfigurationA": {"ProbeInterfaceFileName": "test-config-0.json"},
            },
        },
    }
    (epoch_dir / "Metadata.yml").write_text(_json.dumps(metadata))

    # HARP epoch base: arbitrary seconds-since-1904 for plausible wall-clock times.
    harp_base = 3000.0

    # Aeon CSV convention: the first column is the Aeon timestamp (in seconds),
    # promoted to the DataFrame index via ``index_col=0`` in ``Csv.read``. The
    # remaining columns are named per ``HarpSync.Reader``'s ``columns=`` list.
    # So real CSVs have 4 columns total even though only 3 are named.
    for n in range(n_chunks):
        ts_str = f"2024-06-04T1{n}-00-00"
        csv_path = device_dir / f"{device_name}_HarpSync_{ts_str}.csv"
        rows = []
        for s in range(60):
            harp = harp_base + n * 60 + s
            rows.append(
                {
                    "aeon_time": harp,
                    "clock": 1000 * (n * 60 + s) + 1,
                    "hub_clock": s,
                    "harp_time": harp,
                }
            )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["aeon_time", "clock", "hub_clock", "harp_time"])
            writer.writeheader()
            writer.writerows(rows)


def register_synthetic_experiment(
    tmp_path: Path,
    raw_dir: Path,
    experiment_name: str,
    epoch_dir_name: str,
):
    """Register the minimum fixtures needed for EphysSyncModel.ingest() to work.

    Inserts: lab.Arena, acquisition.PipelineRepository, acquisition.DevicesSchema,
    acquisition.Experiment, acquisition.Experiment.Directory pointing at raw_dir,
    ephys.EphysEpoch (Manual peer, no has_ephys/n_probes on the master row),
    and ephys.EphysEpochConfig (with has_ephys=True, n_probes=0).

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
    # Register the same path under both "raw" and "raw-ephys" — the ephys
    # downstream (resolve_raw_dir_and_epochs) looks up "raw-ephys".
    for dir_type in ("raw", "raw-ephys"):
        acquisition.Experiment.Directory.insert1(
            {
                "experiment_name": experiment_name,
                "directory_type": dir_type,
                "repository_name": repo_key,
                "directory_path": "raw",
            },
            skip_duplicates=True,
        )

    # EphysEpoch is a Manual peer of acquisition.Epoch — insert directly.
    # has_ephys/n_probes live on EphysEpochConfig, not on the master row.
    ephys.EphysEpoch.insert1(
        {
            "experiment_name": experiment_name,
            "epoch_start": epoch_dt,
            "directory_type": "raw-ephys",
            "repository_name": repo_key,
            "epoch_dir": epoch_dir_name,
        },
        skip_duplicates=True,
        ignore_extra_fields=True,
    )

    # EphysEpochConfig is dj.Imported — direct insert with allow_direct_insert=True.
    # n_probes=0 is fine for sync-model-only tests that don't exercise probe discovery.
    ephys.EphysEpochConfig.insert1(
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


def make_synthetic_amplifier_data(
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
        # ONIX clock range for this chunk (matches make_synthetic_ephys_epoch)
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


def register_synthetic_probe_insertion(
    experiment_name: str,
    subject: str,
    epoch_start,
    probe_label: str,
    device_name: str = "NeuropixelsV2Beta",
):
    """Insert ProbeType, ElectrodeConfig, Probe, ProbeInsertion, EphysEpochConfig.Insertion.

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

    # ElectrodeConfig (Lookup) — config_file_name is required for the new
    # (probe_type, config_file_name) unique-index lookup used by ingest_chunks.
    config_hash = uuid.uuid5(uuid.NAMESPACE_DNS, f"{probe_type}-{electrode_config_name}")
    ephys.ElectrodeConfig.insert1(
        {
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
            "electrode_config_description": "synthetic test config",
            "electrode_config_hash": config_hash,
            "config_file_name": f"{electrode_config_name}.json",
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

    # EphysEpochConfig.Insertion (Part, allow_direct_insert).
    ephys.EphysEpochConfig.Insertion.insert1(
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


def make_synthetic_bno055_data(
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
