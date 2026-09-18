"""Golden baseline and integration tests for the ephys pipeline.

Tests the ephys ingestion pipeline using a known dataset (8-channel
subset of abcGolden01 NeuropixelsV2 recording). Tests gracefully skip
if data unavailable.

Requirements:
1. Ephys golden dataset at ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/...

Pipeline cascade tested (all live except SpikeSorting):
    EphysChunk.ingest_chunks → EphysBlockInfo.populate → PreProcessing.populate
    → [SpikeSorting: force-injected from golden KS4 output]
    → PostProcessing.populate → SortedSpikes.populate → SyncedSpikes.populate
"""

import pytest

pytestmark = pytest.mark.integration


class TestEphysEpochDiscovery:
    """Verify ephys epoch and probe discovery setup."""

    def test_ephys_epoch_exists(self, ephys_test_epochs, ctx):
        count = len(ctx.ephys.EphysEpoch & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_probe_count(self, ephys_test_epochs, ctx):
        n_probes = len(
            ctx.ephys.EphysEpochConfig.Insertion
            & {
                "experiment_name": ctx.cfg["experiment_name"],
                "epoch_start": ephys_test_epochs[0]["epoch_start"],
            }
        )
        assert n_probes == ctx.cfg["expected_probe_count"]

    def test_probe_insertions_created(self, ephys_test_epochs, ctx):
        insertions = (
            ctx.ephys.EphysEpochConfig.Insertion & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(insertions) == ctx.cfg["expected_probe_count"]

    def test_probe_insertion_links_correct_subject(self, ephys_test_epochs, ctx):
        pis = (ctx.ephys.ProbeInsertion & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        assert len(pis) >= 1
        assert all(pi["subject"] == ctx.cfg["subject"] for pi in pis)

    def test_discover_epoch_probes_on_golden_data(self, require_ephys_golden_data, ctx):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        epoch_path = require_ephys_golden_data
        device_name, _, labels = discover_epoch_probes(epoch_path)
        assert device_name is not None
        # discover_epoch_probes returns raw-discovery (ProbeA + ProbeB).
        # expected_probe_count is for REGISTERED insertions (ProbeB only).
        assert len(labels) == ctx.cfg["expected_discovered_probes"]


class TestEphysChunkIngestion:
    """Verify EphysChunk.ingest_chunks() output."""

    def test_chunks_ingested(self, ephys_chunks_ingested, ctx):
        count = len(ctx.ephys.EphysChunk & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_chunk_timestamps_valid(self, ephys_chunks_ingested, ctx):
        chunks = (ctx.ephys.EphysChunk & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        for chunk in chunks:
            assert chunk["chunk_start"] < chunk["chunk_end"]

    def test_chunk_files_registered(self, ephys_chunks_ingested, ctx):
        files = list(
            (ctx.ephys.EphysChunk.File & {"experiment_name": ctx.cfg["experiment_name"]}).fetch("file_name")
        )
        assert any("AmplifierData_0.bin" in fn for fn in files), (
            f"AmplifierData_0.bin not registered. Got first 5: {files[:5]}"
        )
        assert any("Clock_0.bin" in fn for fn in files), (
            f"Clock_0.bin not registered. Got first 5: {files[:5]}"
        )
        assert len(files) % 2 == 0, f"Expected even file count (Amp+Clock pairs), got {len(files)}"


class TestEphysBlockInfo:
    """Verify EphysBlockInfo.populate() output."""

    def test_block_info_populated(self, ephys_block_info_populated, ctx):
        blocks = len(ctx.ephys.EphysBlock & {"experiment_name": ctx.cfg["experiment_name"]})
        infos = len(ctx.ephys.EphysBlockInfo & {"experiment_name": ctx.cfg["experiment_name"]})
        assert infos == blocks

    def test_block_duration_correct(self, ephys_block_info_populated, ctx):
        # Block is set up as exactly 35 minutes (block_end - block_start in the
        # ephys_test_blocks fixture); block_duration in hours is exactly 35/60.
        # Tight tolerance to catch any conversion drift.
        infos = (ctx.ephys.EphysBlockInfo & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        for info in infos:
            assert info["block_duration"] == pytest.approx(35 / 60, abs=1e-6)

    def test_block_chunks_associated(self, ephys_block_info_populated, ctx):
        chunk_links = len(ctx.ephys.EphysBlockInfo.Chunk & {"experiment_name": ctx.cfg["experiment_name"]})
        assert chunk_links >= 1

    def test_channel_mappings_created(self, ephys_block_info_populated, ctx):
        # EphysBlockInfo.Channel records the recording's channels (full active set,
        # not the sorting subset), so we check n_recording_channels (384), not
        # n_channels (8 — the sorting subset in ElectrodeGroup.Electrode).
        channel_rows = (
            ctx.ephys.EphysBlockInfo.Channel & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(channel_rows) == ctx.cfg["n_recording_channels"]
        channel_indices = sorted(r["channel_idx"] for r in channel_rows)
        assert channel_indices == list(range(ctx.cfg["n_recording_channels"]))


class TestPreProcessing:
    """Verify PreProcessing.populate() output.

    PreProcessing reads raw ephys binary, selects electrode group channels,
    applies bandpass filter + common average reference, and writes
    recording.zarr + si_recording.pkl.
    """

    def _ensure_prerequisites(self, ctx):
        ctx.ephys.EphysChunk.ingest_chunks(ctx.cfg["experiment_name"])
        ctx.ephys.EphysBlockInfo.populate()
        ctx.spike_sorting.PreProcessing.populate(display_progress=True, suppress_errors=False)

    def test_preprocessing_populated(self, ephys_sorting_setup, require_ephys_golden_data, ctx):
        self._ensure_prerequisites(ctx)
        count = len(ctx.spike_sorting.PreProcessing & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_recording_files_registered(self, ephys_sorting_setup, require_ephys_golden_data, ctx):
        self._ensure_prerequisites(ctx)
        files = (
            ctx.spike_sorting.PreProcessing.File & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        file_names = [f["file_name"] for f in files]
        assert any("si_recording.pkl" in fn for fn in file_names)
        assert not any("recording.dat" in fn for fn in file_names), "recording.dat should not be registered"
        assert not any("recording.zarr" in fn for fn in file_names), (
            "recording.zarr contents should not be registered"
        )

    def test_recording_zarr_exists(self, ephys_sorting_setup, require_ephys_golden_data, ctx):
        self._ensure_prerequisites(ctx)
        key = (ctx.spike_sorting.SortingTask & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()[
            0
        ]
        from aeon.dj_pipeline.utils.paths import scratch_recording_dir

        output_dir = ctx.spike_sorting.PreProcessing.infer_output_dir(key)
        # recording.zarr lives on the scratch mirror when configured, else in-place on ceph.
        recording_zarr = scratch_recording_dir(output_dir.parent / "recording") / "recording.zarr"
        assert recording_zarr.exists(), f"Expected zarr recording at {recording_zarr}"
        assert any(recording_zarr.iterdir()), "recording.zarr directory is empty"

        import numpy as np
        import spikeinterface as si

        rec = si.load(recording_zarr)
        assert rec.get_num_channels() == ctx.cfg["n_channels"]

        # Sample count should reflect a real multi-minute block, not a truncated
        # write (the golden block is ~30 min at 30 kHz). A duration range catches
        # truncation that a bare "> 0" check would miss.
        duration_s = rec.get_num_samples() / rec.get_sampling_frequency()
        assert 300 < duration_s < 7200, (
            f"recording.zarr duration {duration_s:.1f}s outside expected range "
            "(expected a multi-minute block)"
        )

        # Read a slice back to confirm the zarr actually decompresses to real
        # data, not just that the directory and metadata exist.
        traces = rec.get_traces(start_frame=0, end_frame=1000)
        assert traces.shape == (1000, ctx.cfg["n_channels"])
        assert np.any(traces != 0), "recording.zarr decompressed to all-zero traces"


class TestCompressedReadEquivalence:
    """A compressed .zarr twin, read back and given the pipeline's gains/offsets,
    must reproduce the raw .bin read on real golden data.

    This does NOT execute ``PreProcessing.make_compute``; it isolates the
    SpikeInterface round-trip that the read-compressed wiring relies on. The
    companion ``aeon_raw_compression`` library compresses from a plain
    ``read_binary`` (no gains), so the zarr on disk carries no gain/offset
    metadata and ``make_compute`` re-attaches it after ``si.load``. Here we
    confirm, on a real golden amplifier file, that the round-trip preserves the
    raw traces byte-for-byte and that the re-applied gains/offsets match the
    ``.bin`` read. The resolver's own path logic is covered by the unit tests in
    ``tests/dj_pipeline/utils/test_ephys_utils_unit.py::TestResolveEphysFile``.
    """

    def test_zarr_roundtrip_matches_binary(
        self, ephys_chunks_ingested, require_ephys_golden_data, ctx, tmp_path
    ):
        import numpy as np
        import spikeinterface as si
        import spikeinterface.extractors as se

        from aeon.dj_pipeline import acquisition

        exp_key = {"experiment_name": ctx.cfg["experiment_name"]}
        amp_files = (
            ctx.ephys.EphysChunk.File & exp_key & "file_name LIKE '%AmplifierData%.bin'"
        ).to_dicts()
        assert amp_files, "no golden AmplifierData .bin registered"

        f0 = amp_files[0]
        ephys_dir = acquisition.Experiment.get_data_directory(exp_key, directory_type=f0["directory_type"])
        bin_path = ephys_dir / f0["file_path"]
        assert bin_path.exists(), f"golden .bin missing: {bin_path}"

        # Must match PreProcessing.make_compute.
        fs_hz = 30e3
        gain_to_uV = 3.05176
        offset_to_uV = -2048 * gain_to_uV
        num_channels = ctx.cfg["n_recording_channels"]

        # .bin branch: read with gains (as make_compute does), take a short slice.
        # 1 s keeps the zarr write fast (reads are lazy/memmapped); min() guards a
        # chunk shorter than that.
        rec_bin = se.read_binary(
            bin_path,
            sampling_frequency=fs_hz,
            dtype=np.uint16,
            num_channels=num_channels,
            gain_to_uV=gain_to_uV,
            offset_to_uV=offset_to_uV,
        )
        n_frames = min(30_000, rec_bin.get_num_samples())
        rec_bin = rec_bin.frame_slice(0, n_frames)

        # .zarr branch: mimic the library (read WITHOUT gains, save to zarr), then
        # load + re-apply gains/offsets exactly as make_compute's zarr branch does.
        zarr_path = tmp_path / "amp_slice.zarr"
        se.read_binary(
            bin_path,
            sampling_frequency=fs_hz,
            dtype=np.uint16,
            num_channels=num_channels,
        ).frame_slice(0, n_frames).save(format="zarr", folder=zarr_path, n_jobs=1)
        rec_zarr = si.load(zarr_path)
        rec_zarr.set_channel_gains(gain_to_uV)
        rec_zarr.set_channel_offsets(offset_to_uV)

        # Raw traces byte-identical, and re-applied metadata matches the .bin read.
        assert np.array_equal(rec_bin.get_traces(), rec_zarr.get_traces())
        assert np.array_equal(rec_bin.get_channel_gains(), rec_zarr.get_channel_gains())
        assert np.array_equal(rec_bin.get_channel_offsets(), rec_zarr.get_channel_offsets())


class TestPostProcessing:
    """Verify PostProcessing.populate() output.

    PostProcessing creates a SpikeInterface sorting_analyzer with computed
    extensions (waveforms, templates, spike_locations, quality_metrics).
    Depends on SpikeSorting being force-injected from golden data.
    """

    def _ensure_prerequisites(self, ctx):
        ctx.spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)

    def test_postprocessing_populated(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        count = len(ctx.spike_sorting.PostProcessing & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_sorting_analyzer_created(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        output_dir = ephys_sorting_injected["output_dir"]
        analyzer_dir = output_dir / "sorting_analyzer.zarr"
        assert analyzer_dir.exists(), f"Expected zarr analyzer at {analyzer_dir}"
        assert any(analyzer_dir.iterdir())


class TestSortedSpikes:
    """Verify SortedSpikes.populate() output.

    SortedSpikes extracts unit info from the sorting_analyzer: spike counts,
    spike indices, electrode assignments, quality labels.
    """

    def _ensure_prerequisites(self, ctx):
        ctx.spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)
        ctx.spike_sorting.SortedSpikes.populate(display_progress=True, suppress_errors=False)

    def test_sorted_spikes_populated(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        count = len(ctx.spike_sorting.SortedSpikes & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_unit_count(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = len(ctx.spike_sorting.SortedSpikes.Unit & {"experiment_name": ctx.cfg["experiment_name"]})
        assert units == ctx.cfg["expected_unit_count"]

    def test_spike_counts_reasonable(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = (
            ctx.spike_sorting.SortedSpikes.Unit & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for u in units:
            assert u["spike_count"] > 0
        total = sum(u["spike_count"] for u in units)
        assert total == ctx.cfg["expected_total_spikes"]

    def test_quality_labels_assigned(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = (
            ctx.spike_sorting.SortedSpikes.Unit & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        qualities = [u["unit_quality"] for u in units]
        assert set(qualities) <= {"good", "mua", "noise"}
        expected = ctx.cfg["expected_quality_counts"]
        for label, count in expected.items():
            assert qualities.count(label) == count, (
                f"Quality label '{label}' count mismatch: expected {count}, got {qualities.count(label)}"
            )


class TestSyncedSpikes:
    """Verify clock-synchronized spike times.

    SyncedSpikes reads binary Clock files and HarpSync models to convert
    spike sample indices to absolute datetime timestamps.
    """

    def _ensure_prerequisites(self, ctx):
        ctx.spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)
        ctx.spike_sorting.SortedSpikes.populate(display_progress=True, suppress_errors=False)
        ctx.spike_sorting.SyncedSpikes.populate(display_progress=True, suppress_errors=False)

    def test_synced_spikes_populated(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        count = len(ctx.spike_sorting.SyncedSpikes & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_spike_times_are_datetimes(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        import numpy as np

        units = (
            ctx.spike_sorting.SyncedSpikes.Unit & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(units) >= 1
        for unit in units:
            assert np.issubdtype(unit["spike_times"].dtype, np.datetime64)

    def test_spike_times_within_chunk_range(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        import numpy as np

        # Chunks, not sync rows: spikes before the first / after the last HarpSync
        # row of an epoch are kept, with times extrapolated from the sync model.
        chunk_rows = (ctx.ephys.EphysChunk & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        chunk_start = np.datetime64(min(r["chunk_start"] for r in chunk_rows))
        chunk_end = np.datetime64(max(r["chunk_end"] for r in chunk_rows))

        units = (
            ctx.spike_sorting.SyncedSpikes.Unit & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for unit in units:
            assert unit["spike_times"].min() >= chunk_start
            assert unit["spike_times"].max() <= chunk_end


class TestEphysSyncModel:
    """Verify EphysSyncModel.ingest produces sensible rows on golden data.

    The golden epoch has 3 HarpSync CSVs (hourly cadence: 07-00, 08-00, 09-00),
    so we expect 3 sync model rows. Each should have a high-quality regression
    (r² very close to 1.0 for clean NTP-synced clocks) and monotonically
    increasing ONIX/HARP bounds.
    """

    def test_one_sync_row_per_harpsync_csv(self, ephys_test_epochs, ctx):
        rows = (ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        assert len(rows) == 3, (
            f"Expected 3 EphysSyncModel rows (one per hourly HarpSync CSV in the "
            f"golden epoch); got {len(rows)}."
        )

    def test_sync_model_regression_quality(self, ephys_test_epochs, ctx):
        rows = (ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        for row in rows:
            assert row["r2"] > 0.99, (
                f"HARP↔ONIX regression r² is suspiciously low: r²={row['r2']:.6f} "
                f"at sync_start={row['sync_start']}. NTP-synced clocks should give "
                f"r² > 0.99; a low value suggests stale or corrupt sync data."
            )

    def test_sync_model_bounds_monotonic(self, ephys_test_epochs, ctx):
        rows = (ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts(
            order_by="sync_start"
        )
        # Per-row: start < end on both clocks.
        for row in rows:
            assert row["sync_start"] < row["sync_end"], (
                f"Inverted HARP bounds at sync_start={row['sync_start']}"
            )
            assert row["onix_ts_start"] < row["onix_ts_end"], (
                f"Inverted ONIX bounds at sync_start={row['sync_start']}"
            )
        # Across rows: monotonically increasing on both clocks.
        harp_starts = [r["sync_start"] for r in rows]
        onix_starts = [r["onix_ts_start"] for r in rows]
        assert harp_starts == sorted(harp_starts), "sync_start values not monotonic"
        assert onix_starts == sorted(onix_starts), "onix_ts_start values not monotonic"

    def test_sync_start_aligns_with_hour_boundary(self, ephys_test_epochs, ctx):
        """HarpSync CSVs at hour boundaries start one second before the boundary.

        The golden epoch starts at 07:50:11 with HarpSync CSVs at 07-00, 08-00,
        09-00. The acquisition workflow splits these files by its Seconds column,
        which adds the HARP second again on top of ``Value.HarpTime`` and is
        therefore 1 s late. The first row of the 08-00 and 09-00 CSVs thus has a
        ``Value.HarpTime`` (and sync_start) in the last second of the previous
        hour — i.e. ``sync_start.minute == 59`` and ``sync_start.second == 59``.

        Reading the Seconds column instead (PR 592) would put these values at
        ``minute == 0, second == 0``. This is an independent HARP anchor that
        catches a systematic offset the existing r²/monotonicity checks miss.
        """
        rows = (ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts(
            order_by="sync_start"
        )
        # Skip the first row — its source CSV's hour bucket begins before
        # epoch_start (recording started mid-hour), so sync_start is the
        # epoch_start, not the bucket hour.
        for row in rows[1:]:
            sync_start = row["sync_start"]
            assert sync_start.minute == 59 and sync_start.second == 59, (
                f"sync_start={sync_start} is not within the last second of an "
                f"hour (got minute={sync_start.minute}, second={sync_start.second}). "
                f"HARP CSVs at hour boundaries should give sync_start = XX:59:59.xxx. "
                f"A value like XX:00:00 indicates the reader uses the Seconds column, "
                f"which is 1 s late, instead of Value.HarpTime."
            )

    def test_sync_start_matches_value_harptime_column(
        self, ephys_test_epochs, require_ephys_golden_data, ctx
    ):
        """sync_start must come from the CSV's Value.HarpTime, not the Seconds index.

        Each HarpSync CSV has two integer-second timestamp columns:
        ``Value.HarpTime`` (correct — OpenEphys.Onix1 >= 0.4.0 already adds the
        HARP second) and ``Seconds`` (the index, which adds it again — 1 s late).
        We re-read one CSV from the golden epoch, derive what ``sync_start``
        would be from each column, and assert the stored value matches the
        Value.HarpTime-derived one. This is independent of the hour-boundary
        check above (catches non-hour-aligned files too).
        """
        import pandas as pd

        from aeon.dj_pipeline.utils.ephys_utils import harp_to_naive

        csvs = sorted(require_ephys_golden_data.rglob("*_HarpSync_*.csv"))
        if not csvs:
            pytest.skip("No HarpSync CSVs in golden epoch")
        csv_path = csvs[0]

        df = pd.read_csv(csv_path, index_col=0).dropna()
        expected_harp_start = harp_to_naive(int(df["Value.HarpTime"].iloc[0]))
        seconds_harp_start = harp_to_naive(int(df.index[0]))

        # Sanity: the source data really does have the 1s difference. If this
        # ever fails, the premise has changed and the test below is moot.
        assert (seconds_harp_start - expected_harp_start).total_seconds() == 1.0, (
            f"Expected Seconds ({df.index[0]}) and Value.HarpTime "
            f"({df['Value.HarpTime'].iloc[0]}) to differ by 1 second in {csv_path.name}; "
            f"got delta={(seconds_harp_start - expected_harp_start).total_seconds()}s."
        )

        rows = (ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}).to_dicts()
        stored_sync_starts = [r["sync_start"] for r in rows]
        assert expected_harp_start in stored_sync_starts, (
            f"sync_start={expected_harp_start} (Value.HarpTime from {csv_path.name}) "
            f"not found in stored values {stored_sync_starts}. If the reader used the "
            f"Seconds index, sync_start would instead be {seconds_harp_start}."
        )
        assert seconds_harp_start not in stored_sync_starts, (
            f"sync_start={seconds_harp_start} (Seconds index from {csv_path.name}) "
            f"found in stored values — the reader uses the 1 s late Seconds column."
        )


class TestOnixImuChunkOnGoldenData:
    """Exercise OnixImuChunk.populate against the actual golden dataset.

    Each EphysSyncModel row corresponds to a HarpSync sync window. The populate
    finds Bno055 binary chunks whose ONIX range overlaps the window, loads
    them, and filters to the window's ONIX bounds. Every sync window in the
    golden recording overlaps at least one Bno055 chunk, so every OnixImuChunk
    row should carry sample_count > 0.
    """

    def test_one_imu_chunk_per_sync_model(self, ephys_test_epochs, ctx):
        ctx.ephys.OnixImuChunk.populate(
            {"experiment_name": ctx.cfg["experiment_name"]},
            display_progress=False,
            suppress_errors=False,
        )
        n_sync = len(ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]})
        n_imu = len(ctx.ephys.OnixImuChunk & {"experiment_name": ctx.cfg["experiment_name"]})
        assert n_imu == n_sync, f"Expected one OnixImuChunk per EphysSyncModel ({n_sync}); got {n_imu}."

    def test_all_chunks_have_imu_samples(self, ephys_test_epochs, ctx):
        ctx.ephys.OnixImuChunk.populate(
            {"experiment_name": ctx.cfg["experiment_name"]},
            display_progress=False,
            suppress_errors=False,
        )
        sample_counts = list(
            (ctx.ephys.OnixImuChunk & {"experiment_name": ctx.cfg["experiment_name"]}).to_arrays(
                "sample_count"
            )
        )
        # The golden recording is fully covered by Bno055 chunks, so every
        # sync window overlaps real IMU data.
        assert all(c > 0 for c in sample_counts), (
            f"At least one OnixImuChunk row has sample_count=0. "
            f"Per-row counts: {sample_counts}. Expected all > 0 for the golden "
            f"recording whose Bno055 chunks cover the entire ONIX range."
        )


class TestPynappleCodecOnGoldenSpikes:
    """The pynapple codec against real Kilosort4 output.

    Every other codec test uses synthetic spikes: uniform or lognormal draws with
    no bursting, no refractory structure, no drift. These load real sortings —
    64-101 units and 0.9-2.7 M spikes each, 30 kHz — straight off disk.

    Deliberately independent of the DataJoint pipeline. The codec stores pynapple
    objects; it does not care where the spike times came from, and routing through
    SyncedSpikes would couple this to a fixture rework and to PR #611 for no gain.

    Spike *times* and their inter-spike structure are real. The absolute offset is
    derived from the block start, which is what SpikeTrains will do, so the round
    trip is exercised at true Harp magnitude where the float64 ULP is 477 ns.
    """

    @staticmethod
    def _tsgroup_from_sorting(sorting_path):
        """Load a Kilosort4 sorting and wrap it as a TsGroup on Harp seconds."""
        import numpy as np
        import pynapple as nap
        import spikeinterface as si
        from swc.aeon.io import api as io_api

        from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

        sorting = si.load(sorting_path)
        block_dir = sorting_path.parents[3].name  # <block>/<shank>/<paramset>/spike_sorting/…
        t0 = io_api.to_seconds(parse_epoch_timestamp(block_dir.split("_")[0]))

        data = {
            int(u): nap.Ts(t=t0 + sorting.get_unit_spike_train(u) / sorting.sampling_frequency)
            for u in sorting.unit_ids
        }
        all_t = np.concatenate([ts.t for ts in data.values() if len(ts)])
        support = nap.IntervalSet(start=float(all_t.min()), end=float(all_t.max()))
        return nap.TsGroup(data, time_support=support)

    @pytest.fixture(scope="class")
    def golden_tsgroup(self, require_ephys_golden_data, ephys_golden_dataset_config):
        """The largest golden sorting, as a TsGroup. Skips if the artifacts are absent.

        ``require_ephys_golden_data`` resolves ``repository_config["ceph_aeon"]`` —
        honouring ``DJ_REPOSITORY_CONFIG`` — and brings the DB config along, which
        the codec import needs because it pulls in ``aeon.dj_pipeline``. Nothing here
        reads or writes a table.

        The sorter/paramset directory is globbed rather than named, and so are the
        block and shank: per the artifacts' own PROVENANCE.md the block names encode
        a pre-PR-#611 clock and will change when ephys is re-ingested.
        """
        from aeon.dj_pipeline.utils.paths import get_repository_path

        cfg = ephys_golden_dataset_config
        root = get_repository_path("ceph_aeon") / "raw" / cfg["experiment_path"] / cfg["golden_sorting_dir"]
        sortings = sorted(root.glob("*/*/*/spike_sorting/in_container_sorting"))
        if not sortings:
            pytest.skip(f"no golden spike-sorting artifacts under {root}")
        largest = max(sortings, key=lambda p: sum(f.stat().st_size for f in p.rglob("*")))
        return self._tsgroup_from_sorting(largest)

    def test_round_trip_is_bit_exact_on_real_spikes(self, golden_tsgroup, tmp_path):
        """Test that real spike times survive a round trip exactly, not approximately."""
        import numpy as np
        from datajoint.settings import Config

        from aeon.dj_pipeline.utils.codec import PynappleCodec

        config = Config()
        config.stores = {"pynapple_store": {"protocol": "file", "location": str(tmp_path)}}
        codec = PynappleCodec()
        key = {"_schema": "golden", "_table": "spikes", "rec_id": 1, "_config": config}
        stored = codec.encode(golden_tsgroup, key=key, store_name="pynapple_store")
        decoded = codec.decode(stored, key={"_config": config})

        assert list(decoded.index) == list(golden_tsgroup.index)
        for unit in golden_tsgroup.index:
            # exact, not allclose: the float64 ULP at Harp magnitude is 477 ns, and a
            # quantising round trip would shift spikes within a sample undetected
            assert (decoded[unit].t == golden_tsgroup[unit].t).all()
        np.testing.assert_array_equal(decoded.time_support.values, golden_tsgroup.time_support.values)
        assert stored["t_start"] > 3.0e9  # still on the 1904 epoch
        assert stored["n_rows"] == len(golden_tsgroup.index)

    def test_fast_path_matches_stock_on_real_spikes(self, golden_tsgroup, tmp_path):
        """Test fast-path equivalence and report the speed-up on real data.

        The figure in SPEC_PYNAPPLE_CODEC.md comes from synthetic rates. This prints
        the measured value on real Kilosort4 output; update the spec from it.
        """
        import time

        import numpy as np
        import pynapple as nap

        from aeon.dj_pipeline.utils.codec import _tsgroup_from_npz

        path = tmp_path / "golden.npz"
        golden_tsgroup.save(str(path))

        start = time.perf_counter()
        stock = nap.load_file(str(path))
        stock_s = time.perf_counter() - start

        start = time.perf_counter()
        fast = _tsgroup_from_npz(str(path))
        fast_s = time.perf_counter() - start

        assert list(fast.index) == list(stock.index)
        for unit in stock.index:
            np.testing.assert_array_equal(fast[unit].t, stock[unit].t)
        np.testing.assert_allclose(np.asarray(fast.rate), np.asarray(stock.rate))

        n_spikes = sum(len(stock[u]) for u in stock.index)
        print(
            f"\ngolden: {len(stock.index)} units, {n_spikes:,} spikes, "
            f"{path.stat().st_size / 1e6:.1f} MB — "
            f"stock {stock_s:.3f}s, fast {fast_s:.3f}s ({stock_s / fast_s:.1f}x)"
        )
