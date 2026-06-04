"""Golden baseline and integration tests for the ephys pipeline.

Tests the ephys ingestion pipeline using a known dataset (8-channel
subset of abcGolden01 NeuropixelsV2 recording). Tests gracefully skip
if data unavailable.

Requirements:
1. Ephys golden dataset at ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/...
2. probeinterface package installed
3. aeon.schema.ephys available (from swc-aeon package)

Pipeline cascade tested (all live except SpikeSorting):
    EphysChunk.ingest_chunks → EphysBlockInfo.populate → PreProcessing.populate
    → [SpikeSorting: force-injected from golden KS4 output]
    → PostProcessing.populate → SortedSpikes.populate → SyncedSpikes.populate
"""

import pytest

INT16_BYTES = 2  # sizeof(int16); recording.dat stores int16 per channel-sample

pytestmark = pytest.mark.integration
pytest.importorskip("probeinterface", reason="Ephys tests require probeinterface")
pytest.importorskip("aeon.schema.ephys", reason="Ephys tests require aeon.schema.ephys (from swc-aeon)")


class TestEphysEpochDiscovery:
    """Verify ephys epoch and probe discovery setup."""

    def test_ephys_epoch_exists(self, ephys_test_epochs, ctx):
        count = len(ctx.ephys.EphysEpoch & {"experiment_name": ctx.cfg["experiment_name"]})
        assert count >= 1

    def test_has_ephys_flag(self, ephys_test_epochs, ctx):
        configs = (
            ctx.ephys.EphysEpochConfig & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert all(c["has_ephys"] for c in configs)

    def test_probe_count(self, ephys_test_epochs, ctx):
        n_probes = (
            ctx.ephys.EphysEpochConfig
            & {
                "experiment_name": ctx.cfg["experiment_name"],
                "epoch_start": ephys_test_epochs[0]["epoch_start"],
                "has_ephys": True,
            }
        ).fetch1("n_probes")
        assert n_probes == ctx.cfg["expected_probe_count"]

    def test_probe_insertions_created(self, ephys_test_epochs, ctx):
        insertions = (
            ctx.ephys.EphysEpochConfig.Insertion
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(insertions) == ctx.cfg["expected_probe_count"]

    def test_probe_insertion_links_correct_subject(self, ephys_test_epochs, ctx):
        pis = (
            ctx.ephys.ProbeInsertion & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
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
        chunks = (
            ctx.ephys.EphysChunk & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for chunk in chunks:
            assert chunk["chunk_start"] < chunk["chunk_end"]

    def test_chunk_files_registered(self, ephys_chunks_ingested, ctx):
        files = list(
            (
                ctx.ephys.EphysChunk.File & {"experiment_name": ctx.cfg["experiment_name"]}
            ).fetch("file_name")
        )
        assert any("AmplifierData_0.bin" in fn for fn in files), (
            f"AmplifierData_0.bin not registered. Got first 5: {files[:5]}"
        )
        assert any("Clock_0.bin" in fn for fn in files), (
            f"Clock_0.bin not registered. Got first 5: {files[:5]}"
        )
        assert len(files) % 2 == 0, (
            f"Expected even file count (Amp+Clock pairs), got {len(files)}"
        )


class TestEphysBlockInfo:
    """Verify EphysBlockInfo.populate() output."""

    def test_block_info_populated(self, ephys_block_info_populated, ctx):
        blocks = len(ctx.ephys.EphysBlock & {"experiment_name": ctx.cfg["experiment_name"]})
        infos = len(ctx.ephys.EphysBlockInfo & {"experiment_name": ctx.cfg["experiment_name"]})
        assert infos == blocks

    def test_block_duration_correct(self, ephys_block_info_populated, ctx):
        infos = (
            ctx.ephys.EphysBlockInfo & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for info in infos:
            assert abs(info["block_duration"] - 35 / 60) < 0.01

    def test_block_chunks_associated(self, ephys_block_info_populated, ctx):
        chunk_links = len(
            ctx.ephys.EphysBlockInfo.Chunk & {"experiment_name": ctx.cfg["experiment_name"]}
        )
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
    recording.dat + si_recording.pkl.
    """

    def _ensure_prerequisites(self, ctx):
        ctx.ephys.EphysChunk.ingest_chunks(ctx.cfg["experiment_name"])
        ctx.ephys.EphysBlockInfo.populate()
        ctx.spike_sorting.PreProcessing.populate(display_progress=True, suppress_errors=False)

    def test_preprocessing_populated(
        self, ephys_sorting_setup, require_ephys_golden_data, ctx
    ):
        self._ensure_prerequisites(ctx)
        count = len(
            ctx.spike_sorting.PreProcessing & {"experiment_name": ctx.cfg["experiment_name"]}
        )
        assert count >= 1

    def test_recording_files_registered(
        self, ephys_sorting_setup, require_ephys_golden_data, ctx
    ):
        self._ensure_prerequisites(ctx)
        files = (
            ctx.spike_sorting.PreProcessing.File
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        file_names = [f["file_name"] for f in files]
        assert any("si_recording.pkl" in fn for fn in file_names)

    def test_recording_binary_exists(
        self, ephys_sorting_setup, require_ephys_golden_data, ctx
    ):
        self._ensure_prerequisites(ctx)
        key = (
            ctx.spike_sorting.SortingTask & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()[0]
        output_dir = ctx.spike_sorting.PreProcessing.infer_output_dir(key)
        recording_dat = output_dir.parent / "recording" / "recording.dat"
        assert recording_dat.exists()
        file_size = recording_dat.stat().st_size
        assert file_size > 0
        expected_size = ctx.cfg["n_channels"] * INT16_BYTES
        assert file_size % expected_size == 0, (
            f"File size {file_size} not divisible by frame size {expected_size}"
        )
        assert file_size >= 500_000_000, (
            f"recording.dat too small ({file_size} bytes), expected ~1 GB"
        )


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
        count = len(
            ctx.spike_sorting.PostProcessing & {"experiment_name": ctx.cfg["experiment_name"]}
        )
        assert count >= 1

    def test_sorting_analyzer_created(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        output_dir = ephys_sorting_injected["output_dir"]
        analyzer_dir = output_dir / "sorting_analyzer"
        assert analyzer_dir.exists()
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
        count = len(
            ctx.spike_sorting.SortedSpikes & {"experiment_name": ctx.cfg["experiment_name"]}
        )
        assert count >= 1

    def test_unit_count(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = len(
            ctx.spike_sorting.SortedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"]}
        )
        assert units == ctx.cfg["expected_unit_count"]

    def test_spike_counts_reasonable(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = (
            ctx.spike_sorting.SortedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for u in units:
            assert u["spike_count"] > 0
        total = sum(u["spike_count"] for u in units)
        assert total == ctx.cfg["expected_total_spikes"]

    def test_quality_labels_assigned(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        units = (
            ctx.spike_sorting.SortedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        qualities = [u["unit_quality"] for u in units]
        assert set(qualities) <= {"good", "mua", "noise"}
        assert qualities.count("good") == 7
        assert qualities.count("mua") == 7


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
        count = len(
            ctx.spike_sorting.SyncedSpikes & {"experiment_name": ctx.cfg["experiment_name"]}
        )
        assert count >= 1

    def test_spike_times_are_datetimes(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        import numpy as np

        units = (
            ctx.spike_sorting.SyncedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(units) >= 1
        for unit in units:
            assert np.issubdtype(unit["spike_times"].dtype, np.datetime64)

    def test_spike_times_within_sync_range(self, ephys_sorting_injected, ctx):
        self._ensure_prerequisites(ctx)
        import numpy as np

        sync_rows = (
            ctx.ephys.EphysSyncModel & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        sync_start = np.datetime64(min(r["sync_start"] for r in sync_rows))
        sync_end = np.datetime64(max(r["sync_end"] for r in sync_rows))

        units = (
            ctx.spike_sorting.SyncedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        for unit in units:
            assert unit["spike_times"].min() >= sync_start
            assert unit["spike_times"].max() <= sync_end
