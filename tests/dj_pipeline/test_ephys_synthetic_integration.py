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


# ---------------------------------------------------------------------------
# Helpers (local — different scope from tests/fixtures/ephys/ephys_factories.py
# which inserts EphysEpoch directly. These tests need ingest_epochs to run.)
# ---------------------------------------------------------------------------


def _write_harpsync_csv(
    epoch_dir: Path, device_name: str, ts_label: str, harp_base: float, onix_base: int, n_rows: int = 60
):
    """Write one HarpSync_*.csv with monotonically-increasing HARP + ONIX clocks."""
    device_dir = epoch_dir / device_name
    device_dir.mkdir(parents=True, exist_ok=True)
    csv_path = device_dir / f"{device_name}_HarpSync_{ts_label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["aeon_time", "clock", "hub_clock", "harp_time"])
        writer.writeheader()
        for s in range(n_rows):
            writer.writerow(
                {
                    "aeon_time": harp_base + s,
                    "clock": onix_base + 1000 * s,
                    "hub_clock": s,
                    "harp_time": harp_base + s,
                }
            )


def _write_metadata_yml(epoch_dir: Path, device_name: str, probe_b_filename: str | None = None):
    """Write a minimal Metadata.yml with one or two probe configurations."""
    metadata = {
        "Devices": {
            device_name + "e": {
                "DeviceName": device_name,
                "ConfigurationA": {"ProbeInterfaceFileName": None},  # disabled
                "ConfigurationB": {"ProbeInterfaceFileName": probe_b_filename}
                if probe_b_filename
                else {"ProbeInterfaceFileName": None},
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
            raw_dir / epoch_a_name,
            device_name,
            "2024-06-04T10-00-00",
            harp_base=harp_a_start,
            onix_base=1,
        )
        _write_harpsync_csv(
            raw_dir / epoch_b_name,
            device_name,
            "2024-06-04T11-00-00",
            harp_base=harp_b_start,
            onix_base=600_001,
        )

        _register_experiment_only(tmp_path, raw_dir, experiment_name)

        ephys.EphysEpoch.ingest_epochs(experiment_name)

        # Two EphysEpoch rows
        epochs = (ephys.EphysEpoch & {"experiment_name": experiment_name}).to_dicts(order_by="epoch_start")
        assert len(epochs) == 2, f"Expected 2 EphysEpoch rows, got {len(epochs)}"

        # Exactly one EphysEpochEnd row — for the FIRST epoch
        ends = (ephys.EphysEpochEnd & {"experiment_name": experiment_name}).to_dicts()
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

        # And — positively assert the LAST epoch has NO EphysEpochEnd row
        # (it should only get one when a newer epoch arrives).
        exp_key = {"experiment_name": experiment_name}
        assert not (ephys.EphysEpochEnd & {**exp_key, "epoch_start": epochs[1]["epoch_start"]}), (
            "Most recent epoch must have no EphysEpochEnd until a successor is discovered"
        )

        # Duration is in hours
        expected_duration_hours = (
            epochs[1]["epoch_start"] - epochs[0]["epoch_start"]
        ).total_seconds() / 3600.0
        assert abs(ends[0]["epoch_duration"] - expected_duration_hours) < 1e-6


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
                "arena_x_dim": 2.0,
                "arena_y_dim": 2.0,
                "arena_z_dim": 0.2,
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
        with pytest.raises(ValueError, match="multiple ElectrodeConfigs"):
            ephys.EphysBlockInfo.populate(
                {"experiment_name": experiment_name},
                display_progress=False,
                suppress_errors=False,
            )


class TestEphysChunkOutsideAllWindows:
    """A chunk entirely before or after an epoch's HarpSync rows must still be ingested.

    It must link to the nearest window: the first for a chunk that ends too early, the
    last for one that starts too late - extrapolated.
    """

    @pytest.mark.parametrize(
        ("case", "ts_range"),
        [
            ("before", (0, 0)),
            ("after", (59500, 59600)),
        ],
    )
    def test_ephys_chunk_outside_window_is_still_ingested(
        self, dj_config_integration, tmp_path, case, ts_range
    ):
        from ephys_factories import (
            make_synthetic_amplifier_data,
            make_synthetic_ephys_epoch,
            register_synthetic_experiment,
            register_synthetic_probe_insertion,
        )

        from aeon.dj_pipeline import acquisition, ephys
        from aeon.dj_pipeline import subject as subj_mod

        # experiment_name is varchar(32) — keep the case suffix short.
        experiment_name = f"test_chunk_outside_win_{case}"
        epoch_dir_name = "2024-06-12T10-24-07"
        device_name = "NeuropixelsV2Beta"
        probe_label = "ProbeA"
        subject_name = "test-mouse-outside"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # Single HarpSync window: ONIX clock range [1, 59001].
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        # Entire amplifier chunk's ONIX range falls outside that window (before or after it).
        make_synthetic_amplifier_data(
            raw_dir, epoch_dir_name, device_name, probe_label, n_chunks=1, ts_ranges=[ts_range]
        )

        epoch_start = register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        subj_mod.Subject.insert1(
            {"subject": subject_name, "sex": "U", "subject_birth_date": "2024-01-01"},
            skip_duplicates=True,
        )
        acquisition.Experiment.Subject.insert1(
            {"experiment_name": experiment_name, "subject": subject_name},
            skip_duplicates=True,
        )
        register_synthetic_probe_insertion(
            experiment_name, subject_name, epoch_start, probe_label, device_name
        )

        ephys.EphysSyncModel.ingest(experiment_name)
        sync_rows = (ephys.EphysSyncModel & {"experiment_name": experiment_name}).to_dicts()
        assert len(sync_rows) == 1
        the_only_window_sync_start = sync_rows[0]["sync_start"]

        ephys.EphysChunk.ingest_chunks(experiment_name)

        chunk_rows = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts()
        assert len(chunk_rows) == 1, (
            f"Chunk entirely outside every HarpSync row ({case}) was not ingested; got {len(chunk_rows)}."
        )

        link_rows = (ephys.EphysChunk.SyncModel & {"experiment_name": experiment_name}).to_dicts()
        assert len(link_rows) == 1, (
            f"Expected the chunk linked to the epoch's one window, got {len(link_rows)}"
        )
        assert link_rows[0]["sync_start"] == the_only_window_sync_start, (
            f"Chunk linked to sync_start={link_rows[0]['sync_start']}, "
            f"but the epoch's only window has sync_start={the_only_window_sync_start}"
        )


class TestSyncedSpikesIndexingAndExtrapolation:
    """Build the minimal DB state for SyncedSpikes.make() without the full spike-sorting pipeline.

    Ephys chunks (each backed by a real 10-sample Clock.bin), SortedSpikes, and everything
    upstream of it are inserted directly.
    """

    def _build_and_populate(
        self, tmp_path, experiment_name, n_chunks, spike_indices, chunk_indices, ts_ranges=None
    ):
        """Set up minimal DB state for SyncedSpikes.make(), run it, and return spike_counts.

        - ``n_chunks``: number of ephys chunks (and matching HarpSync windows) to create.
        - ``ts_ranges``: optional list of (chunk_start, chunk_end) ONIX-clock overrides,
          one per chunk. Omit to use the factory's default of placing each chunk n's
          samples inside its own matching window n (use this to push a chunk's samples
          outside its window instead, e.g. the extrapolation test below).
        - ``spike_indices``: list of absolute indices into the concatenated recording.
        - ``chunk_indices``: which ephys chunks' spike_counts to return (0-based, ordered
          by chunk_start). Returns a list in the same order; 0 for a chunk that got no
          SyncedSpikes.Unit row at all.
        """
        from datetime import UTC, datetime

        import numpy as np
        from ephys_factories import (
            make_synthetic_amplifier_data,
            make_synthetic_ephys_epoch,
            register_synthetic_experiment,
            register_synthetic_probe_insertion,
        )

        from aeon.dj_pipeline import acquisition, ephys, spike_sorting
        from aeon.dj_pipeline import subject as subj_mod

        epoch_dir_name = "2024-06-13T10-24-07"
        device_name = "NeuropixelsV2Beta"
        probe_label = "ProbeA"
        subject_name = "test-mouse-synced"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        # n_chunks HarpSync windows; amplifier chunk n matches window n by default.
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=n_chunks)
        make_synthetic_amplifier_data(
            raw_dir,
            epoch_dir_name,
            device_name,
            probe_label,
            n_chunks=n_chunks,
            ts_ranges=ts_ranges,
        )

        epoch_start = register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        subj_mod.Subject.insert1(
            {"subject": subject_name, "sex": "U", "subject_birth_date": "2024-01-01"},
            skip_duplicates=True,
        )
        acquisition.Experiment.Subject.insert1(
            {"experiment_name": experiment_name, "subject": subject_name},
            skip_duplicates=True,
        )
        register_synthetic_probe_insertion(
            experiment_name, subject_name, epoch_start, probe_label, device_name
        )

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.EphysChunk.ingest_chunks(experiment_name)

        chunk_rows = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts(
            order_by="chunk_start"
        )
        assert len(chunk_rows) == n_chunks, f"Expected {n_chunks} ephys chunk(s), got {len(chunk_rows)}"
        block_start = chunk_rows[0]["chunk_start"]
        block_end = chunk_rows[-1]["chunk_end"]

        block_key = {
            "experiment_name": experiment_name,
            "subject": subject_name,
            "insertion_number": 1,
            "block_start": block_start,
            "block_end": block_end,
        }
        ephys.EphysBlock.insert1(block_key)

        probe_type, electrode_config_name = (
            ephys.EphysEpochConfig.Insertion & {"experiment_name": experiment_name, "subject": subject_name}
        ).fetch1("probe_type", "electrode_config_name")

        ephys.EphysBlockInfo.insert1(
            {
                **block_key,
                "block_duration": (block_end - block_start).total_seconds() / 3600.0,
                "probe_type": probe_type,
                "electrode_config_name": electrode_config_name,
            },
            allow_direct_insert=True,
        )
        ephys.EphysBlockInfo.Chunk.insert(
            [{**block_key, "chunk_start": c["chunk_start"]} for c in chunk_rows],
            allow_direct_insert=True,
        )

        # ElectrodeGroup/SortingParamSet are global lookups, not experiment-scoped —
        # skip_duplicates since both tests in this class share the same probe_type/
        # electrode_config_name (from register_synthetic_probe_insertion's fixed values).
        econfig_key = {"probe_type": probe_type, "electrode_config_name": electrode_config_name}
        spike_sorting.ElectrodeGroup.insert1(
            {
                **econfig_key,
                "electrode_group": "all",
                "electrode_group_description": "synthetic",
                "electrode_count": 1,
            },
            skip_duplicates=True,
        )
        spike_sorting.ElectrodeGroup.Electrode.insert1(
            {**econfig_key, "electrode_group": "all", "electrode": 0}, skip_duplicates=True
        )

        spike_sorting.SortingParamSet.insert1(
            {
                "paramset_id": "test-paramset",
                "sorting_method": "kilosort4",
                "paramset_description": "synthetic",
                "params": {},
            },
            skip_duplicates=True,
        )

        sorting_task_key = {
            **block_key,
            **econfig_key,
            "electrode_group": "all",
            "paramset_id": "test-paramset",
        }
        spike_sorting.SortingTask.insert1(sorting_task_key)

        now = datetime.now(UTC)
        spike_sorting.PreProcessing.insert1(
            {
                **sorting_task_key,
                "execution_time": now,
                "execution_duration": 0.0,
                "sorting_output_dir": "x",
            },
            allow_direct_insert=True,
        )
        spike_sorting.SpikeSorting.insert1(
            {**sorting_task_key, "execution_time": now, "execution_duration": 0.0},
            allow_direct_insert=True,
        )
        spike_sorting.PostProcessing.insert1(
            {**sorting_task_key, "execution_time": now, "execution_duration": 0.0},
            allow_direct_insert=True,
        )
        spike_sorting.SortedSpikes.insert1(
            {**sorting_task_key, "execution_time": now, "execution_duration": 0.0},
            allow_direct_insert=True,
        )

        spike_indices = np.asarray(spike_indices)
        spike_sorting.SortedSpikes.Unit.insert1(
            {
                **sorting_task_key,
                "unit": 1,
                "electrode": 0,
                "unit_quality": "good",
                "spike_count": len(spike_indices),
                "spike_indices": spike_indices,
                "spike_sites": np.zeros(len(spike_indices), dtype=int),
            },
            allow_direct_insert=True,
        )

        spike_sorting.SyncedSpikes.populate(
            {"experiment_name": experiment_name}, display_progress=False, suppress_errors=False
        )

        unit_rows = (spike_sorting.SyncedSpikes.Unit & {"experiment_name": experiment_name}).to_dicts(
            order_by="chunk_start"
        )

        counts_by_chunk = {r["chunk_start"]: r["spike_count"] for r in unit_rows}
        return [counts_by_chunk.get(chunk_rows[i]["chunk_start"], 0) for i in chunk_indices]

    def test_offset_counted_from_chunk_start(self, ephys_full_pipeline, tmp_path):
        """Neither chunk's spike count may cross the other chunk's boundary.

        Two chunks, each with its own HarpSync window and its 10 ONIX samples inside
        it; spike_indices = [5, 9, 10, 14, 19] into the concatenated [chunk A | chunk B]
        recording, so chunk A must get only [5, 9] (2 spikes), chunk B only
        [10, 14, 19] (3 spikes).
        """
        chunk_a_count, chunk_b_count = self._build_and_populate(
            tmp_path,
            "test_synced_spikes_offset",
            n_chunks=2,
            spike_indices=[5, 9, 10, 14, 19],
            chunk_indices=[0, 1],
        )

        assert (chunk_a_count, chunk_b_count) == (2, 3), (
            f"Expected chunk A=2, chunk B=3 spikes; got chunk A={chunk_a_count}, chunk B={chunk_b_count}."
        )

    def test_spikes_outside_window_bounds(self, ephys_full_pipeline, tmp_path):
        """Spikes outside a matched sync window's own bounds must be extrapolated, not dropped."""
        (chunk_count,) = self._build_and_populate(
            tmp_path,
            "test_synced_spikes_extrap",
            n_chunks=1,
            spike_indices=[0, 3, 7],  # start at 0 so the chunk-start offset is 0
            chunk_indices=[0],
            ts_ranges=[(59500, 59600)],  # push it outside the window's [1, 59001] bounds
        )

        assert chunk_count == 3, (
            "The chunk's 3 spikes fall outside its matched sync window's own "
            "[onix_ts_start, onix_ts_end] bounds; they must be extrapolated via the window's "
            f"model, not dropped. Got spike_count={chunk_count}."
        )
