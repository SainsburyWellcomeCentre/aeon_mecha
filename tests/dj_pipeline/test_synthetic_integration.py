"""Synthetic-data integration tests for acquisition and ephys schema invariants.

These tests do not require the golden dataset on disk. Each test builds its
own minimal DB state to exercise a specific behavior.
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


class TestEpochEndLookback:
    """Epoch.ingest_epochs backfills EpochEnd for each epoch once its successor is found.

    epoch_end is the earlier of the previous epoch's last chunk end and the next
    epoch's start, so the three epochs below cover both sides of that min().
    """

    def test_look_back_inserts_end_for_previous_epochs(self, dj_config_integration, tmp_path):
        import datetime

        from aeon.dj_pipeline import acquisition

        experiment_name = "test_epoch_end_lookback"
        raw_dir = tmp_path / "raw"
        # Only the chunk file names matter: Epoch.ingest_epochs reads the epoch from the
        # parent dir and the chunk start from the file name. CameraTop is the default
        # chunk-discovery device for experiments not in _ref_device_mapping.
        # As on a real rig, epochs start at arbitrary times but chunk files are named for
        # the whole hour they cover, so an epoch's first chunk starts before the epoch does.
        chunks = {
            "2024-06-04T10-00-00": ["2024-06-04T10-00-00", "2024-06-04T11-00-00"],  # A
            # B starts mid-way through A's last chunk; both have an 11:00 chunk.
            "2024-06-04T11-30-00": ["2024-06-04T11-00-00"],
            # C starts after B's last chunk ends.
            "2024-06-04T14-20-00": ["2024-06-04T14-00-00"],
        }
        for epoch_dir, chunk_starts in chunks.items():
            device_dir = raw_dir / epoch_dir / "CameraTop"
            device_dir.mkdir(parents=True)
            for chunk_start in chunk_starts:
                (device_dir / f"CameraTop_{chunk_start}.csv").touch()

        _register_experiment_only(tmp_path, raw_dir, experiment_name)

        acquisition.Epoch.ingest_epochs(experiment_name)

        exp_key = {"experiment_name": experiment_name}
        epoch_starts = (acquisition.Epoch & exp_key).to_arrays("epoch_start", order_by="epoch_start")
        assert [ts.isoformat() for ts in epoch_starts] == [
            "2024-06-04T10:00:00",
            "2024-06-04T11:30:00",
            "2024-06-04T14:20:00",
        ]

        ends = {
            row["epoch_start"]: row
            for row in (acquisition.EpochEnd & exp_key).to_dicts(order_by="epoch_start")
        }
        a, b = (datetime.datetime.fromisoformat(s) for s in ("2024-06-04T10:00", "2024-06-04T11:30"))
        assert set(ends) == {a, b}, "Only epochs with a successor get an EpochEnd (not C)"

        # A: next epoch starts (11:30) before A's last chunk ends (12:00).
        assert ends[a]["epoch_end"] == b
        assert ends[a]["epoch_duration"] == pytest.approx(1.5)
        # B: B's last chunk ends (12:00 - chunk start + 1 h) before the next epoch starts (14:20).
        assert ends[b]["epoch_end"] == datetime.datetime.fromisoformat("2024-06-04T12:00")
        assert ends[b]["epoch_duration"] == pytest.approx(0.5)


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
        raw_dir.mkdir(parents=True)

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
        raw_dir.mkdir(parents=True)

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
    """SyncedSpikes.make() indexes spikes from the chunk start and extrapolates out-of-window spikes.

    Builds the minimal DB state without the full spike-sorting pipeline: ephys chunks
    (each backed by a real 10-sample Clock.bin), SortedSpikes, and everything upstream
    of it are inserted directly.
    """

    def _build_and_populate(
        self,
        tmp_path,
        experiment_name,
        n_chunks,
        spike_indices,
        chunk_indices,
        ts_ranges=None,
        amp_samples=None,
        clock_samples=None,
        sorted_samples=None,
        amp_as_zarr=False,
        fmt="binary",
        before_ingest=None,
        after_ingest=None,
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
        - ``amp_samples`` / ``clock_samples``: optional per-chunk sample counts to truncate
          the AmplifierData / Clock files to (the factory writes 10 samples per chunk).
        - ``sorted_samples``: sample count of the sorted recording; defaults to the sum of
          the AmplifierData sample counts, as PreProcessing would have concatenated them.
        - ``before_ingest(raw_dir, epoch_dir_name, device_name)``: seed extra raw files
          before discovery runs.
        - ``after_ingest(experiment_name)``: register EphysChunk.File rows the discovery
          globs would never produce, to exercise the downstream file restriction directly.
        - ``amp_as_zarr``: replace each AmplifierData .bin with its compressed .zarr twin
          under the processed root, as raw ephys compression leaves them.
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
        raw_dir.mkdir(parents=True)

        if fmt != "binary" and (amp_samples is not None or clock_samples is not None or amp_as_zarr):
            raise ValueError(
                f"amp_samples/clock_samples/amp_as_zarr all operate on the raw .bin "
                f"files, which fmt={fmt!r} does not write. The request would be ignored "
                f"and the test would assert against the plain path while claiming otherwise."
            )

        # n_chunks HarpSync windows; amplifier chunk n matches window n by default.
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=n_chunks)
        make_synthetic_amplifier_data(
            raw_dir,
            epoch_dir_name,
            device_name,
            probe_label,
            n_chunks=n_chunks,
            ts_ranges=ts_ranges,
            fmt=fmt,
        )
        amp_samples = amp_samples or [10] * n_chunks
        clock_samples = clock_samples or [10] * n_chunks
        device_dir = raw_dir / epoch_dir_name / device_name
        for n in range(n_chunks) if fmt == "binary" else ():  # truncate, as a half-copied file would
            for stream, dtype, n_keep in (
                ("AmplifierData", np.uint16, amp_samples[n]),
                ("Clock", np.uint64, clock_samples[n]),
            ):
                path = device_dir / f"{device_name}_{probe_label}_{stream}_{n}.bin"
                np.fromfile(path, dtype=dtype)[:n_keep].tofile(path)

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

        if before_ingest is not None:  # seed raw files that must not be picked up
            before_ingest(raw_dir, epoch_dir_name, device_name)

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.EphysChunk.ingest_chunks(experiment_name)
        if after_ingest is not None:  # register rows the discovery globs never produce
            after_ingest(experiment_name)
        if amp_as_zarr:  # after ingestion, which reads the raw files
            import gc

            import spikeinterface.extractors as se

            zarr_dir = tmp_path / "processed" / epoch_dir_name / device_name
            for amp_bin in sorted(device_dir.glob("*AmplifierData_*.bin")):
                se.read_binary(amp_bin, sampling_frequency=30e3, dtype=np.uint16, num_channels=1).save(
                    format="zarr", folder=zarr_dir / amp_bin.with_suffix(".zarr").name
                )
                gc.collect()  # read_binary's memmap blocks unlink on Windows
                amp_bin.unlink()

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

        # SyncedSpikes checks the AmplifierData sample counts against the sorted recording,
        # so write a minimal analyzer with the sorted recording's sample count.
        import probeinterface as pi
        import spikeinterface as si

        probe = pi.generate_linear_probe(num_elec=1)
        probe.set_device_channel_indices([0])
        recording = si.NumpyRecording(
            np.zeros((sorted_samples or sum(amp_samples), 1), dtype=np.float32), sampling_frequency=30e3
        )
        sorting_output_dir = experiment_name
        si.create_sorting_analyzer(
            si.NumpySorting.from_unit_dict({1: np.array([0])}, sampling_frequency=30e3),
            recording.set_probe(probe),
            format="binary_folder",
            folder=spike_sorting.get_sorting_root_dir() / sorting_output_dir / "sorting_analyzer",
            sparse=False,
        )

        now = datetime.now(UTC)
        spike_sorting.PreProcessing.insert1(
            {
                **sorting_task_key,
                "execution_time": now,
                "execution_duration": 0.0,
                "sorting_output_dir": sorting_output_dir,
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

    def test_truncated_amplifier_file_sets_chunk_boundary(self, ephys_full_pipeline, tmp_path):
        """Chunk boundaries follow the AmplifierData sample counts, not the Clock file lengths.

        Chunk A's AmplifierData lost its last 4 samples (6 left, Clock still 10), so the
        sorted recording is [A: 0-5 | B: 6-15]. Spikes [5, 6, 15] belong 1 to A and 2 to B;
        boundaries from the Clock files would put index 6 in A instead.
        """
        chunk_a_count, chunk_b_count = self._build_and_populate(
            tmp_path,
            "test_synced_spikes_trunc_amp",
            n_chunks=2,
            spike_indices=[5, 6, 15],
            chunk_indices=[0, 1],
            amp_samples=[6, 10],
        )

        assert (chunk_a_count, chunk_b_count) == (1, 2)

    def test_arrow_chunks_assign_spikes_like_binary(self, ephys_full_pipeline, tmp_path):
        """Same ONIX timestamps and spike indices must split identically either way."""
        arrow = self._build_and_populate(
            tmp_path / "arrow",
            "test_synced_arrow",
            n_chunks=2,
            spike_indices=[5, 6, 15],
            chunk_indices=[0, 1],
            fmt="arrow",
        )
        binary = self._build_and_populate(
            tmp_path / "binary",
            "test_synced_binary",
            n_chunks=2,
            spike_indices=[5, 6, 15],
            chunk_indices=[0, 1],
            fmt="binary",
        )
        # The absolute split is pinned too: a run where SyncedSpikes inserts no Unit
        # rows at all returns [0, 0] for both formats, and a pure differential
        # assertion would call that agreement.
        expected = [2, 1]  # 10 samples/chunk, spikes at 5 and 6 in chunk 0, 15 in chunk 1
        assert arrow == expected, f"arrow split {arrow}, expected {expected}"
        assert binary == expected, f"binary split {binary}, expected {expected}"

        # The split is by sample index, so it would survive a corrupted clock. Compare
        # the synced HARP times too: the arrow clock is spilled to a memmap on the way
        # into extend_clock, and a dtype or truncation slip there moves every spike
        # without changing any count.
        import numpy as np

        from aeon.dj_pipeline import spike_sorting

        def spike_times(experiment_name):
            return [
                r["spike_times"]
                for r in (spike_sorting.SyncedSpikes.Unit & {"experiment_name": experiment_name}).to_dicts(
                    order_by="chunk_start"
                )
            ]

        arrow_times, binary_times = spike_times("test_synced_arrow"), spike_times("test_synced_binary")
        assert len(arrow_times) == len(binary_times) == 2
        for a, b in zip(arrow_times, binary_times, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_commutator_clock_files_are_not_mistaken_for_ephys(self, ephys_full_pipeline, tmp_path):
        """A commutator clock must never be selected as amplifier data.

        It is a 10 Hz turns counter; splicing it into the ONIX timeline shifts every
        spike. The row is inserted directly because discovery never registers one, so
        writing the file to disk alone would leave the restriction nothing to reject.
        """
        import numpy as np

        from aeon.dj_pipeline import ephys

        def seed_commutator(raw_dir, epoch_dir_name, device_name):
            d = raw_dir / epoch_dir_name / device_name
            for n in range(2):
                # plausible but wrong timestamps: same dtype, different rate
                np.arange(1, 7, dtype=np.uint64).tofile(d / f"{device_name}_Commutator_Clock_{n}.bin")
                np.zeros(6, dtype=np.uint32).tofile(d / f"{device_name}_Commutator_Turns_{n}.bin")

        def register_commutator(experiment_name):
            rows = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts(
                order_by="chunk_start"
            )
            device_name = "NeuropixelsV2Beta"
            ephys.EphysChunk.File.insert(
                [
                    {
                        **{k: r[k] for k in ephys.EphysChunk.primary_key},
                        "directory_type": "raw-ephys",
                        "file_name": f"{device_name}_Commutator_Clock_{n}.bin",
                        "file_path": (
                            f"2024-06-13T10-24-07/{device_name}/{device_name}_Commutator_Clock_{n}.bin"
                        ),
                    }
                    for n, r in enumerate(rows)
                ],
                allow_direct_insert=True,
            )
            selected = (ephys.EphysChunk.File & {"experiment_name": experiment_name}).to_arrays("file_name")
            assert any("Commutator" in n for n in selected), (
                "the commutator row must be in EphysChunk.File for the restriction to reject it"
            )

        counts = self._build_and_populate(
            tmp_path,
            "test_synced_commutator",
            n_chunks=2,
            spike_indices=[5, 15],
            chunk_indices=[0, 1],
            before_ingest=seed_commutator,
            after_ingest=register_commutator,
        )
        assert list(counts) == [1, 1], f"commutator files perturbed the split: {counts}"

    def test_short_clock_file_is_extrapolated(self, ephys_full_pipeline, tmp_path):
        """Spikes past the end of a short Clock file get extrapolated times, not dropped."""
        (chunk_count,) = self._build_and_populate(
            tmp_path,
            "test_synced_spikes_short_clock",
            n_chunks=1,
            spike_indices=[0, 8, 9],  # 8 and 9 lie past the truncated Clock file's 8 samples
            chunk_indices=[0],
            clock_samples=[8],
        )

        assert chunk_count == 3

    def test_raw_files_changed_since_sorting_raises(self, ephys_full_pipeline, tmp_path):
        """AmplifierData files that no longer add up to the sorted recording raise."""
        with pytest.raises(ValueError, match="changed since sorting"):
            self._build_and_populate(
                tmp_path,
                "test_synced_spikes_changed",
                n_chunks=1,
                spike_indices=[0],
                chunk_indices=[0],
                sorted_samples=8,  # e.g. sorted while the file was truncated, then re-copied
            )

    def test_compressed_amplifier_file_sets_chunk_boundary(self, ephys_full_pipeline, tmp_path):
        """With only the compressed .zarr twin on disk, its sample count sets the boundary."""
        chunk_a_count, chunk_b_count = self._build_and_populate(
            tmp_path,
            "test_synced_spikes_zarr",
            n_chunks=2,
            spike_indices=[5, 6, 15],
            chunk_indices=[0, 1],
            amp_samples=[6, 10],
            amp_as_zarr=True,
        )

        assert (chunk_a_count, chunk_b_count) == (1, 2)


class TestSortingParamSetPreprocessingMethod:
    """SortingParamSet.insert validates the preprocessing selection."""

    def test_unknown_method_rejected_on_insert(self, ephys_full_pipeline):
        """A typo in SI_PREPROCESSING_METHOD fails at insert, not hours into PreProcessing."""
        from aeon.dj_pipeline import spike_sorting

        row = {
            "paramset_id": "test-bad-preproc",
            "sorting_method": "kilosort4",
            "params": {"SI_PREPROCESSING_METHOD": "aeon_defualt"},
        }
        with pytest.raises(ValueError, match="aeon_defualt"):
            spike_sorting.SortingParamSet.insert1(row)
        assert not (spike_sorting.SortingParamSet & {"paramset_id": "test-bad-preproc"})


class TestSpikeSortingLupin:
    """A non-Kilosort sorter runs through SpikeSorting."""

    def test_lupin_sorts_unpreprocessed_recording(self, ephys_full_pipeline, tmp_path):
        """Lupin runs through SpikeSorting.make_compute with SI_PREPROCESSING_METHOD="none".

        The recording is laid out as PreProcessing writes it (si_recording.pkl + recording.zarr).
        Guards against Kilosort-only sorter params (skip_kilosort_preprocessing) being passed
        to non-Kilosort sorters, which reject them.
        """
        import spikeinterface.full as si

        from aeon.dj_pipeline import spike_sorting
        from aeon.dj_pipeline.utils.ephys_preprocessing import apply_preprocessing

        assert spike_sorting.SortingMethod & {"sorting_method": "lupin"}

        recording, _ = si.generate_ground_truth_recording(
            durations=[10.0], num_channels=32, num_units=5, seed=0
        )
        recording = apply_preprocessing(recording, {"SI_PREPROCESSING_METHOD": "none"})

        output_dir = tmp_path / "lupin_test"
        recording_dir = output_dir.parent / "recording"
        recording_dir.mkdir(parents=True)
        recording_file = recording_dir / "si_recording.pkl"
        recording.dump_to_pickle(file_path=recording_file)
        recording.save(format="zarr", folder=recording_dir / "recording.zarr")

        params = {
            "SI_PREPROCESSING_METHOD": "none",
            # n_jobs=1: SI 0.104's threaded template matching can deadlock in lupin.
            "SI_SORTING_PARAMS": {"job_kwargs": {"n_jobs": 1, "chunk_duration": "1s"}},
        }
        sorting_output_dir, _, _ = spike_sorting.SpikeSorting().make_compute(
            {}, recording_file, output_dir, params, "lupin"
        )

        sorting = si.load(sorting_output_dir / "si_sorting.pkl", base_folder=output_dir)
        assert len(sorting.unit_ids) > 0


class TestProbeInsertionFromImplants:
    """Metadata-asserted insertion numbers are authoritative, and conflicts raise."""

    def _tables(self, ephys_full_pipeline):
        ephys = ephys_full_pipeline["ephys"]
        acquisition = ephys_full_pipeline["acquisition"]
        return ephys, acquisition

    def _experiment(self, tmp_path, name):
        """Register a minimal experiment and return its name."""
        from ephys_factories import make_synthetic_ephys_epoch, register_synthetic_experiment

        from aeon.dj_pipeline import acquisition, ephys, subject

        raw_dir, epoch = tmp_path / "raw", "2026-10-13T094255Z"
        make_synthetic_ephys_epoch(raw_dir, epoch, "NeuropixelsV2", n_chunks=1)
        register_synthetic_experiment(tmp_path, raw_dir, name, epoch)
        ephys.ProbeType.insert1({"probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True)
        subject.Subject.insert1(
            {"subject": "SUBJ-1", "sex": "U", "subject_birth_date": "2026-01-01"},
            skip_duplicates=True,
        )
        acquisition.Experiment.Subject.insert1(
            {"experiment_name": name, "subject": "SUBJ-1"}, skip_duplicates=True
        )
        return name

    def test_explicit_insertion_number_is_used(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_explicit")
        ephys.Probe.insert1(
            {"probe": "SER-1", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        key = find_or_create_probe_insertion(
            exp, "SUBJ-1", "SER-1", ephys.ProbeInsertion, ephys.Probe, insertion_number=7
        )
        assert key["insertion_number"] == 7

    def test_conflicting_insertion_number_raises(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_conflict")
        ephys.Probe.insert1(
            {"probe": "SER-2", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        find_or_create_probe_insertion(
            exp, "SUBJ-1", "SER-2", ephys.ProbeInsertion, ephys.Probe, insertion_number=1
        )
        with pytest.raises(ValueError, match="insertion_number"):
            find_or_create_probe_insertion(
                exp, "SUBJ-1", "SER-2", ephys.ProbeInsertion, ephys.Probe, insertion_number=2
            )

    def test_target_area_auto_created_and_linked(self, ephys_full_pipeline, tmp_path, caplog):
        """A new area is created (warned about) and linked to the insertion."""
        from aeon.dj_pipeline.ephys import _insert_target_area
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_area")
        ephys.Probe.insert1(
            {"probe": "SER-4", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        pi_key = find_or_create_probe_insertion(
            exp, "SUBJ-1", "SER-4", ephys.ProbeInsertion, ephys.Probe, insertion_number=1
        )
        # a name no other test uses: the warning fires only the first time an area appears
        area = "TestArea-NeverSeenBefore"
        with caplog.at_level("WARNING"):
            _insert_target_area(area, pi_key)

        assert ephys.TargetArea & {"target_area": area}
        assert ephys.InsertionTargetArea & {**pi_key, "target_area": area}
        assert any(area in r.message for r in caplog.records), "a new area should warn"

        # ...and not again for an area that already exists
        caplog.clear()
        with caplog.at_level("WARNING"):
            _insert_target_area(area, pi_key)
        assert not any(area in r.message for r in caplog.records), "existing area should not warn"

    def test_changing_target_area_raises(self, ephys_full_pipeline, tmp_path):
        """A second area on one insertion would fan out every join through the link table."""
        from aeon.dj_pipeline.ephys import _insert_target_area
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_area_conflict")
        ephys.Probe.insert1(
            {"probe": "SER-6", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        pi_key = find_or_create_probe_insertion(
            exp, "SUBJ-1", "SER-6", ephys.ProbeInsertion, ephys.Probe, insertion_number=1
        )
        _insert_target_area("AreaOne", pi_key)
        with pytest.raises(ValueError, match="already linked"):
            _insert_target_area("AreaTwo", pi_key)

    def test_target_area_none_is_a_noop(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline.ephys import _insert_target_area
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_noarea")
        ephys.Probe.insert1(
            {"probe": "SER-5", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        pi_key = find_or_create_probe_insertion(
            exp, "SUBJ-1", "SER-5", ephys.ProbeInsertion, ephys.Probe, insertion_number=1
        )
        _insert_target_area(None, pi_key)
        assert not (ephys.InsertionTargetArea & pi_key)

    def test_implantation_date_recorded(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import find_or_create_probe_insertion

        ephys, _ = self._tables(ephys_full_pipeline)
        exp = self._experiment(tmp_path, "test_implants_date")
        ephys.Probe.insert1(
            {"probe": "SER-3", "probe_type": "neuropixels2.0-multishank"}, skip_duplicates=True
        )
        key = find_or_create_probe_insertion(
            exp,
            "SUBJ-1",
            "SER-3",
            ephys.ProbeInsertion,
            ephys.Probe,
            insertion_number=1,
            implantation_date="2026-05-20",
        )
        assert str((ephys.ProbeInsertion & key).fetch1("implantation_date")) == "2026-05-20 00:00:00"


class TestFactoryFormatValidation:
    """An unvalidated fmt would make the arrow-vs-binary equivalence test vacuous."""

    def test_unknown_format_raises(self, tmp_path):
        from ephys_factories import make_synthetic_amplifier_data

        with pytest.raises(ValueError, match="fmt"):
            make_synthetic_amplifier_data(
                tmp_path, "ep", "NeuropixelsV2", "ProbeA", n_chunks=1, fmt="Arrow"
            )

    def test_arrow_and_binary_write_different_trees(self, tmp_path):
        """Guards the equivalence test: the formats must not produce the same files."""
        from ephys_factories import make_synthetic_amplifier_data

        for fmt in ("binary", "arrow"):
            make_synthetic_amplifier_data(
                tmp_path / fmt, "ep", "NeuropixelsV2", "ProbeA", n_chunks=1, fmt=fmt
            )
        suffixes = {
            fmt: sorted(q.suffix for q in (tmp_path / fmt / "ep" / "NeuropixelsV2").iterdir())
            for fmt in ("binary", "arrow")
        }
        assert suffixes["binary"] == [".bin", ".bin"], suffixes
        assert suffixes["arrow"] == [".arrow"], suffixes


class TestPreProcessingOnArrow:
    """PreProcessing must build a SpikeInterface recording from arrow chunks.

    This is the spec's definition of "works" for arrow, and the fixture split in
    the golden tests does not reach it: PreProcessing is keyed on SortingTask,
    which sits inside the sorting cascade.
    """

    def _populate(self, tmp_path, experiment_name, fmt):
        from aeon.dj_pipeline import ephys, spike_sorting

        exp = TestArrowChunkIngestion()._setup(tmp_path, experiment_name, [fmt, fmt])

        chunks = (ephys.EphysChunk & {"experiment_name": exp}).to_dicts(order_by="chunk_start")
        assert len(chunks) == 2, chunks
        insertion = {k: chunks[0][k] for k in ("experiment_name", "subject", "insertion_number")}
        block_key = {
            **insertion,
            "block_start": chunks[0]["chunk_start"],
            "block_end": chunks[-1]["chunk_end"],
        }
        ephys.EphysBlock.insert1(block_key)
        probe_type, econfig = (ephys.EphysEpochConfig.Insertion & {"experiment_name": exp}).fetch1(
            "probe_type", "electrode_config_name"
        )
        econfig_key = {"probe_type": probe_type, "electrode_config_name": econfig}
        ephys.EphysBlockInfo.insert1(
            {
                **block_key,
                "block_duration": 1.0,
                **econfig_key,
            },
            allow_direct_insert=True,
        )
        ephys.EphysBlockInfo.Chunk.insert(
            [{**block_key, "chunk_start": c["chunk_start"]} for c in chunks],
            allow_direct_insert=True,
        )
        ephys.EphysBlockInfo.Channel.insert(
            [{**block_key, **econfig_key, "electrode": 0, "channel_idx": 0}],
            allow_direct_insert=True,
        )
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
                "paramset_id": "pp-arrow",
                "sorting_method": "kilosort4",
                "paramset_description": "synthetic",
                "params": {"SI_PREPROCESSING_METHOD": "none"},
            },
            skip_duplicates=True,
        )
        task = {**block_key, **econfig_key, "electrode_group": "all", "paramset_id": "pp-arrow"}
        spike_sorting.SortingTask.insert1(task)

        spike_sorting.PreProcessing.populate(task, suppress_errors=False)
        return spike_sorting, task

    def test_preprocessing_runs_on_arrow_chunks(self, ephys_full_pipeline, tmp_path):
        spike_sorting, task = self._populate(tmp_path, "test_pp_arrow", "arrow")
        assert spike_sorting.PreProcessing & task, "PreProcessing produced no row for arrow"

    def test_preprocessed_recording_is_loadable_and_has_expected_samples(
        self, ephys_full_pipeline, tmp_path
    ):
        """The written si_recording.pkl must reload with the arrow chunks' samples."""
        import spikeinterface as si

        spike_sorting, task = self._populate(tmp_path, "test_pp_arrow_load", "arrow")
        out = spike_sorting.get_sorting_root_dir() / (spike_sorting.PreProcessing & task).fetch1(
            "sorting_output_dir"
        )
        pkl = out.parent / "recording" / "si_recording.pkl"
        assert pkl.exists(), f"si_recording.pkl not written to {pkl}"
        rec = si.load(pkl, base_folder=out)
        # the factory writes 10 samples per chunk, two chunks concatenated
        assert rec.get_num_samples() == 20, rec.get_num_samples()


class TestAmplifierFileSelection:
    """The restriction that finds amplifier files must match both formats.

    PreProcessing and SyncedSpikes both select the amplifier file for a block.
    A restriction that misses arrow leaves PreProcessing with an empty file list
    and si.concatenate_recordings([]) raises IndexError naming nothing.
    """

    def test_restriction_selects_arrow_files(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.spike_sorting import AMPLIFIER_FILE_RESTRICTION

        exp = TestArrowChunkIngestion()._setup(tmp_path, "test_amp_sel_arrow", ["arrow", "arrow"])
        rows = (ephys.EphysChunk.File & {"experiment_name": exp} & AMPLIFIER_FILE_RESTRICTION).to_dicts()
        names = sorted(r["file_name"] for r in rows)
        assert len(names) == 2, f"arrow amplifier files not selected: {names}"
        assert all(n.endswith(".arrow") for n in names), names

    def test_restriction_selects_binary_files(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.spike_sorting import AMPLIFIER_FILE_RESTRICTION

        exp = TestArrowChunkIngestion()._setup(tmp_path, "test_amp_sel_bin", ["binary", "binary"])
        rows = (ephys.EphysChunk.File & {"experiment_name": exp} & AMPLIFIER_FILE_RESTRICTION).to_dicts()
        names = sorted(r["file_name"] for r in rows)
        assert len(names) == 2, f"expected 2 AmplifierData files, got {names}"
        assert all("AmplifierData" in n for n in names), names

    def test_restriction_excludes_clock_commutator_and_other_arrow_streams(
        self, ephys_full_pipeline, tmp_path
    ):
        """Only amplifier files may match, in either format.

        The non-amplifier rows are inserted into EphysChunk.File directly: discovery
        registers amplifier files and their derived Clock twins only, so writing them
        to disk would leave the restriction with nothing to reject. The Bno055 arrow
        row is the one that matters for the arrow half -- it sits in the same device
        directory, and a bare '%.arrow' would hand PreProcessing a 17-column IMU table
        as amplifier data and give SyncedSpikes its length as a chunk boundary.
        """
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.spike_sorting import AMPLIFIER_FILE_RESTRICTION

        device_name = "NeuropixelsV2"
        extra = [
            f"{device_name}_Commutator_Clock_0.bin",
            f"{device_name}_Bno055_0.arrow",
        ]

        def register_non_amplifier_rows(experiment_name):
            chunk = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts()[0]
            ephys.EphysChunk.File.insert(
                [
                    {
                        **{k: chunk[k] for k in ephys.EphysChunk.primary_key},
                        "directory_type": "raw-ephys",
                        "file_name": name,
                        "file_path": f"2026-10-13T094255Z/{device_name}/{name}",
                    }
                    for name in extra
                ],
                allow_direct_insert=True,
            )

        exp = TestArrowChunkIngestion()._setup(
            tmp_path, "test_amp_sel_excl", ["binary"], after_ingest=register_non_amplifier_rows
        )
        all_names = set((ephys.EphysChunk.File & {"experiment_name": exp}).to_arrays("file_name"))
        assert set(extra) <= all_names, (
            f"the rows the restriction must reject are not in EphysChunk.File: {all_names}"
        )

        rows = (ephys.EphysChunk.File & {"experiment_name": exp} & AMPLIFIER_FILE_RESTRICTION).to_dicts()
        names = [r["file_name"] for r in rows]
        assert not any("Commutator" in n for n in names), names
        assert not any("Bno055" in n for n in names), names
        assert not any(n.endswith("_Clock_0.bin") for n in names), names
        assert names, "the binary amplifier file must still be selected"


class TestArrowChunkIngestion:
    """ingest_chunks discovers arrow amplifier files and reads their Clock column."""

    def _setup(self, tmp_path, experiment_name, fmts, after_ingest=None, before_ingest=None):
        """Build an epoch whose chunk n uses fmts[n]; return the experiment name.

        ``before_ingest(raw_dir, epoch_dir_name, device_name)`` runs before discovery,
        for tests that seed extra raw files. ``after_ingest(experiment_name)`` runs
        once chunks are registered, for tests that need EphysChunk.File rows the
        discovery globs would never produce.
        """
        from ephys_factories import (
            make_synthetic_amplifier_data,
            make_synthetic_ephys_epoch,
            register_synthetic_experiment,
            register_synthetic_probe_insertion,
        )

        import aeon.dj_pipeline.subject as subj_mod
        from aeon.dj_pipeline import acquisition, ephys

        epoch_dir_name, device_name, probe_label = "2026-10-13T094255Z", "NeuropixelsV2", "ProbeA"
        subject_name = "test-mouse-arrow"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True)

        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=len(fmts))
        # one call per chunk so each can pick its own on-disk format
        dest = raw_dir / epoch_dir_name / device_name
        for n, fmt in enumerate(fmts):
            # The factory always writes chunk index 0, so build each chunk in its own
            # scratch tree and move it into place under the right index -- writing in
            # place would overwrite the previous chunk before it could be renamed.
            scratch = tmp_path / f"scratch{n}"
            make_synthetic_amplifier_data(
                scratch,
                epoch_dir_name,
                device_name,
                probe_label,
                n_chunks=1,
                ts_ranges=[(60000 * n + 501, 60000 * n + 58501)],
                fmt=fmt,
            )
            for src in (scratch / epoch_dir_name / device_name).iterdir():
                src.rename(dest / src.name.replace("_0.", f"_{n}."))

        epoch_start = register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)
        subj_mod.Subject.insert1(
            {"subject": subject_name, "sex": "U", "subject_birth_date": "2026-01-01"},
            skip_duplicates=True,
        )
        acquisition.Experiment.Subject.insert1(
            {"experiment_name": experiment_name, "subject": subject_name}, skip_duplicates=True
        )
        register_synthetic_probe_insertion(
            experiment_name, subject_name, epoch_start, probe_label, device_name
        )
        if before_ingest is not None:  # seed extra raw files before discovery
            before_ingest(raw_dir, epoch_dir_name, device_name)
        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.EphysChunk.ingest_chunks(experiment_name)
        if after_ingest is not None:  # register rows the discovery globs never produce
            after_ingest(experiment_name)
        return experiment_name

    def test_arrow_chunks_are_ingested(self, ephys_full_pipeline, tmp_path):
        from aeon.dj_pipeline import ephys

        exp = self._setup(tmp_path, "test_arrow_ingest", ["arrow", "arrow"])
        assert len((ephys.EphysChunk & {"experiment_name": exp}).to_dicts()) == 2

    def test_mixed_binary_and_arrow_in_one_epoch(self, ephys_full_pipeline, tmp_path):
        """Review Focus #2: discovery must handle both formats in one pass."""
        from aeon.dj_pipeline import ephys

        exp = self._setup(tmp_path, "test_arrow_mixed", ["binary", "arrow"])
        chunks = (ephys.EphysChunk & {"experiment_name": exp}).to_dicts(order_by="chunk_start")
        assert len(chunks) == 2, f"mixed-format epoch ingested {len(chunks)} chunks, expected 2"

    def test_same_chunk_index_in_both_formats_raises(self, ephys_full_pipeline, tmp_path):
        """One chunk index in both formats is ambiguous, not two chunks.

        Discovery globs both patterns in one pass and the paths differ, so the pair
        gets registered twice and PreProcessing concatenates the chunk twice -- every
        later spike shifts by a chunk.
        """
        from ephys_factories import make_synthetic_amplifier_data

        device_name, probe_label = "NeuropixelsV2", "ProbeA"
        epoch_dir_name = "2026-10-13T094255Z"

        def write_arrow_twin(raw_dir, _epoch_dir_name, _device_name):
            # the same chunk 0, also written as arrow, one sample later
            scratch = tmp_path / "twin"
            make_synthetic_amplifier_data(
                scratch,
                epoch_dir_name,
                device_name,
                probe_label,
                n_chunks=1,
                ts_ranges=[(502, 58501)],
                fmt="arrow",
            )
            src = scratch / epoch_dir_name / device_name / f"{device_name}_{probe_label}_0.arrow"
            src.rename(raw_dir / epoch_dir_name / device_name / src.name)

        with pytest.raises(ValueError, match="both binary and arrow"):
            self._setup(tmp_path, "test_arrow_dup_index", ["binary"], before_ingest=write_arrow_twin)

    def test_arrow_chunk_bounds_match_binary(self, ephys_full_pipeline, tmp_path):
        """Same ONIX timestamps must yield the same chunk_start/end either way."""
        from aeon.dj_pipeline import ephys

        a = self._setup(tmp_path / "a", "test_arrow_bounds_a", ["arrow"])
        b = self._setup(tmp_path / "b", "test_arrow_bounds_b", ["binary"])
        ca = (ephys.EphysChunk & {"experiment_name": a}).fetch1("chunk_start", "chunk_end")
        cb = (ephys.EphysChunk & {"experiment_name": b}).fetch1("chunk_start", "chunk_end")
        assert ca == cb
