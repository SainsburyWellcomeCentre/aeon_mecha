"""Synthetic-data integration tests for ephys schema invariants.

These tests do not require the golden dataset on disk. Each test builds its
own minimal DB state to exercise a specific behavior:

- TestEphysEpochEndLookback: ``EphysEpoch.ingest_epochs`` backfills
  ``EphysEpochEnd`` for the previous epoch when a newer one is discovered.
- TestEphysBlockInfoMultiConfigValidation: ``EphysBlockInfo.populate`` refuses
  to populate a block whose chunks reference different ElectrodeConfigs.
"""

import csv
import json
import logging
import uuid
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

pytest.importorskip("probeinterface", reason="Ephys tests require probeinterface")
pytest.importorskip("aeon.schema.ephys", reason="Ephys tests require aeon.schema.ephys (from swc-aeon)")


# ---------------------------------------------------------------------------
# Helpers (local — different scope from tests/fixtures/ephys/ephys_factories.py
# which inserts EphysEpoch directly. These tests need ingest_epochs to run.)
# ---------------------------------------------------------------------------


def _write_harpsync_csv(epoch_dir: Path, device_name: str, ts_label: str,
                        harp_base: float, onix_base: int, n_rows: int = 60):
    """Write one HarpSync_*.csv with monotonically-increasing HARP + ONIX clocks."""
    device_dir = epoch_dir / device_name
    device_dir.mkdir(parents=True, exist_ok=True)
    csv_path = device_dir / f"{device_name}_HarpSync_{ts_label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["aeon_time", "clock", "hub_clock", "harp_time"]
        )
        writer.writeheader()
        for s in range(n_rows):
            writer.writerow({
                "aeon_time": harp_base + s,
                "clock": onix_base + 1000 * s,
                "hub_clock": s,
                "harp_time": harp_base + s,
            })


def _write_metadata_yml(epoch_dir: Path, device_name: str,
                        probe_b_filename: str | None = None):
    """Write a minimal Metadata.yml with one or two probe configurations."""
    metadata = {
        "Devices": {
            device_name + "e": {
                "DeviceName": device_name,
                "ConfigurationA": {"ProbeInterfaceFileName": None},  # disabled
                "ConfigurationB": {"ProbeInterfaceFileName": probe_b_filename}
                if probe_b_filename else {"ProbeInterfaceFileName": None},
            },
        },
    }
    (epoch_dir / "Metadata.yml").write_text(json.dumps(metadata))


def _register_experiment_only(tmp_path: Path, raw_dir: Path, experiment_name: str):
    """Register the bare-minimum experiment + raw-ephys directory mapping.

    Unlike ``register_synthetic_experiment``, this does NOT insert EphysEpoch
    or EphysEpochConfig — we want ingest_epochs / populate to create those.
    """
    import aeon.dj_pipeline as _pipeline
    from aeon.dj_pipeline import acquisition, lab

    repo_key = "test_repo"
    _pipeline.repository_config[repo_key] = str(tmp_path)

    acquisition.PipelineRepository.insert1({"repository_name": repo_key}, skip_duplicates=True)
    lab.Arena.insert1(
        {
            "arena_name": "synthetic-arena",
            "arena_description": "",
            "arena_shape": "circular",
            "arena_x_dim": 2.0,
            "arena_y_dim": 2.0,
            "arena_z_dim": 0.2,
        },
        skip_duplicates=True,
    )
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": "synthetic.schema:Synthetic"},
        skip_duplicates=True,
    )
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2024-01-01 00:00:00",
            "experiment_description": "synthetic ephys test",
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


# ===========================================================================
# TestEphysEpochEndLookback
# ===========================================================================


class TestEphysEpochEndLookback:
    """EphysEpoch.ingest_epochs backfills EphysEpochEnd via look-back.

    When the Nth epoch is discovered, the (N-1)th epoch gets an EphysEpochEnd
    row inserted with epoch_end = Nth epoch's harp_start. The most recent
    epoch has no EphysEpochEnd row until a newer one is discovered.
    """

    def test_look_back_inserts_end_for_previous_epoch(self, dj_config_integration, tmp_path):
        from aeon.dj_pipeline import ephys

        experiment_name = "test_ephys_epoch_end_lookback"
        device_name = "NeuropixelsV2"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Two epoch directories with distinct, monotonic HARP starts (1 hour apart).
        epoch_a_name = "2024-06-04T10-00-00"
        epoch_b_name = "2024-06-04T11-00-00"
        harp_a_start = 3000.0
        harp_b_start = 3600.0  # exactly 600 seconds later (10 min)
        (raw_dir / epoch_a_name).mkdir()
        (raw_dir / epoch_b_name).mkdir()
        _write_harpsync_csv(
            raw_dir / epoch_a_name, device_name, "2024-06-04T10-00-00",
            harp_base=harp_a_start, onix_base=1,
        )
        _write_harpsync_csv(
            raw_dir / epoch_b_name, device_name, "2024-06-04T11-00-00",
            harp_base=harp_b_start, onix_base=600_001,
        )

        _register_experiment_only(tmp_path, raw_dir, experiment_name)

        ephys.EphysEpoch.ingest_epochs(experiment_name)

        # Two EphysEpoch rows
        epochs = (
            ephys.EphysEpoch & {"experiment_name": experiment_name}
        ).to_dicts(order_by="epoch_start")
        assert len(epochs) == 2, f"Expected 2 EphysEpoch rows, got {len(epochs)}"

        # Exactly one EphysEpochEnd row — for the FIRST epoch
        ends = (
            ephys.EphysEpochEnd & {"experiment_name": experiment_name}
        ).to_dicts()
        assert len(ends) == 1, (
            f"Expected 1 EphysEpochEnd row (look-back backfills only the previous "
            f"epoch when a newer one is discovered); got {len(ends)}"
        )

        # The EphysEpochEnd row references the FIRST epoch and its epoch_end equals
        # the SECOND epoch's epoch_start
        assert ends[0]["epoch_start"] == epochs[0]["epoch_start"], (
            "EphysEpochEnd should backfill the OLDER epoch, not the newer one"
        )
        assert ends[0]["epoch_end"] == epochs[1]["epoch_start"], (
            f"EphysEpochEnd.epoch_end ({ends[0]['epoch_end']}) should equal the "
            f"next epoch's epoch_start ({epochs[1]['epoch_start']})"
        )

        # Duration is in hours
        expected_duration_hours = (
            epochs[1]["epoch_start"] - epochs[0]["epoch_start"]
        ).total_seconds() / 3600.0
        assert abs(ends[0]["epoch_duration"] - expected_duration_hours) < 1e-6


# ===========================================================================
# TestEphysBlockInfoMultiConfigValidation
# ===========================================================================


class TestEphysBlockInfoMultiConfigValidation:
    """EphysBlockInfo.make refuses to populate a block spanning multiple ElectrodeConfigs.

    Builds the minimum DB state for a single EphysBlock that references chunks
    from two different epochs, each with its own ElectrodeConfig. populate()
    must raise — concatenating recordings from different electrode setups is
    meaningless downstream.
    """

    def test_multi_config_block_raises(self, dj_config_integration):
        from aeon.dj_pipeline import ephys, subject

        experiment_name = "test_bi_multi_config"
        subject_name = "test_subj_mc"
        probe_type_name = "neuropixels2.0-multishank-test"
        probe_serial = "test-probe-mc-001"
        config_a = "config-shank0"
        config_b = "config-shank1"

        # --- Minimum reference data ---
        ephys.ProbeType.insert1({"probe_type": probe_type_name}, skip_duplicates=True)
        ephys.ProbeType.Electrode.insert1(
            {
                "probe_type": probe_type_name,
                "electrode": 0,
                "shank": 0,
                "x_coord": 0.0,
                "y_coord": 0.0,
                "electrode_name": "e0",
            },
            skip_duplicates=True,
        )
        ephys.Probe.insert1(
            {"probe": probe_serial, "probe_type": probe_type_name, "probe_comment": "mc"},
            skip_duplicates=True,
        )

        # Two ElectrodeConfigs for the same probe_type (different recording configs)
        for name in (config_a, config_b):
            ephys.ElectrodeConfig.insert1(
                {
                    "probe_type": probe_type_name,
                    "electrode_config_name": name,
                    "electrode_config_description": f"synthetic {name}",
                    "electrode_config_hash": uuid.uuid5(uuid.NAMESPACE_DNS, name),
                },
                skip_duplicates=True,
            )
            ephys.ElectrodeConfig.Electrode.insert1(
                {
                    "probe_type": probe_type_name,
                    "electrode_config_name": name,
                    "electrode": 0,
                },
                skip_duplicates=True,
            )

        # Experiment + subject scaffolding (skip the directory machinery — not needed)
        from aeon.dj_pipeline import acquisition, lab
        lab.Arena.insert1(
            {
                "arena_name": "synthetic-arena",
                "arena_description": "",
                "arena_shape": "circular",
                "arena_x_dim": 2.0, "arena_y_dim": 2.0, "arena_z_dim": 0.2,
            },
            skip_duplicates=True,
        )
        acquisition.Experiment.insert1(
            {
                "experiment_name": experiment_name,
                "experiment_start_time": "2024-01-01 00:00:00",
                "experiment_description": "mc test",
                "arena_name": "synthetic-arena",
                "lab": "SWC",
                "location": "room-0",
                "experiment_type": "foraging",
            },
            skip_duplicates=True,
        )
        subject.Subject.insert1(
            {"subject": subject_name, "sex": "U", "subject_birth_date": "2024-01-01"},
            skip_duplicates=True,
        )
        acquisition.Experiment.Subject.insert1(
            {"experiment_name": experiment_name, "subject": subject_name},
            skip_duplicates=True,
        )

        # Two epochs, same ProbeInsertion, different ElectrodeConfigs
        from datetime import datetime
        epoch_a_start = datetime(2024, 6, 4, 10, 0, 0)
        epoch_b_start = datetime(2024, 6, 4, 11, 0, 0)
        for es in (epoch_a_start, epoch_b_start):
            ephys.EphysEpoch.insert1(
                {
                    "experiment_name": experiment_name,
                    "epoch_start": es,
                    "epoch_dir": "",
                },
                skip_duplicates=True,
                ignore_extra_fields=True,
            )

        ephys.ProbeInsertion.insert1(
            {
                "experiment_name": experiment_name,
                "subject": subject_name,
                "insertion_number": 1,
                "probe": probe_serial,
            },
            skip_duplicates=True,
        )

        for es, config_name in ((epoch_a_start, config_a), (epoch_b_start, config_b)):
            ephys.EphysEpochConfig.insert1(
                {
                    "experiment_name": experiment_name,
                    "epoch_start": es,
                    "has_ephys": True,
                    "n_probes": 1,
                },
                skip_duplicates=True,
                allow_direct_insert=True,
            )
            ephys.EphysEpochConfig.Insertion.insert1(
                {
                    "experiment_name": experiment_name,
                    "epoch_start": es,
                    "subject": subject_name,
                    "insertion_number": 1,
                    "probe_label": "ProbeB",
                    "probe_type": probe_type_name,
                    "electrode_config_name": config_name,
                    "config_file_name": f"{config_name}.json",
                },
                skip_duplicates=True,
                allow_direct_insert=True,
            )

        # One chunk per epoch (same ProbeInsertion). Chunks straddle a Block.
        chunk_a_start = datetime(2024, 6, 4, 10, 0, 0)
        chunk_a_end = datetime(2024, 6, 4, 10, 30, 0)
        chunk_b_start = datetime(2024, 6, 4, 11, 0, 0)
        chunk_b_end = datetime(2024, 6, 4, 11, 30, 0)
        for cs, ce, es in (
            (chunk_a_start, chunk_a_end, epoch_a_start),
            (chunk_b_start, chunk_b_end, epoch_b_start),
        ):
            ephys.EphysChunk.insert1(
                {
                    "experiment_name": experiment_name,
                    "subject": subject_name,
                    "insertion_number": 1,
                    "chunk_start": cs,
                    "chunk_end": ce,
                    "epoch_start": es,
                },
                skip_duplicates=True,
            )

        # Block spanning both chunks
        ephys.EphysBlock.insert1(
            {
                "experiment_name": experiment_name,
                "subject": subject_name,
                "insertion_number": 1,
                "block_start": chunk_a_start,
                "block_end": chunk_b_end,
            },
            skip_duplicates=True,
        )

        # populate() must refuse: chunks reference different ElectrodeConfigs
        with pytest.raises(Exception, match="multiple ElectrodeConfigs"):
            ephys.EphysBlockInfo.populate(
                {"experiment_name": experiment_name},
                display_progress=False,
                suppress_errors=False,
            )
