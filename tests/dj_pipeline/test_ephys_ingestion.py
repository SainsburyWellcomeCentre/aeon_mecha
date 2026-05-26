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

pytestmark = pytest.mark.integration
pytest.importorskip("probeinterface", reason="Ephys tests require probeinterface")
pytest.importorskip("aeon.schema.ephys", reason="Ephys tests require aeon.schema.ephys (from swc-aeon)")


class TestEphysEpochDiscovery:
    """Verify ephys epoch and probe discovery setup."""

    def test_ephys_epoch_exists(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        count = len(ephys.EphysEpoch & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_has_ephys_flag(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        epochs = (ephys.EphysEpoch & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        assert any(e["has_ephys"] for e in epochs)

    def test_probe_count(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        ephys_epochs = (
            ephys.EphysEpoch & {"experiment_name": cfg["experiment_name"], "has_ephys": True}
        ).to_dicts()
        assert len(ephys_epochs) >= 1
        assert ephys_epochs[0]["n_probes"] == cfg["expected_probe_count"]

    def test_probe_insertions_created(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        insertions = (
            ephys.EphysEpoch.Insertion & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()
        assert len(insertions) == cfg["expected_probe_count"]

    def test_probe_insertion_links_correct_subject(
        self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        pis = (
            ephys.ProbeInsertion & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()
        assert len(pis) >= 1
        assert all(pi["subject"] == cfg["subject"] for pi in pis)

    def test_discover_epoch_probes_on_golden_data(
        self, require_ephys_golden_data, ephys_golden_dataset_config
    ):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        epoch_path = require_ephys_golden_data
        device_name, _, labels = discover_epoch_probes(epoch_path)
        assert device_name is not None
        assert len(labels) == ephys_golden_dataset_config["expected_probe_count"]


class TestEphysChunkIngestion:
    """Verify EphysChunk.ingest_chunks() output."""

    def _ensure_chunks_ingested(self, ephys_full_pipeline, cfg):
        ephys = ephys_full_pipeline["ephys"]
        ephys.EphysChunk.ingest_chunks(cfg["experiment_name"])

    def test_chunks_ingested(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_chunks_ingested(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        count = len(ephys.EphysChunk & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_chunk_timestamps_valid(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_chunks_ingested(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        chunks = (ephys.EphysChunk & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        for chunk in chunks:
            assert chunk["chunk_start"] < chunk["chunk_end"]

    def test_chunk_files_registered(self, ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_chunks_ingested(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        files = (ephys.EphysChunk.File & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        assert len(files) >= 2


class TestEphysBlockInfo:
    """Verify EphysBlockInfo.populate() output."""

    def _ensure_prerequisites(self, ephys_full_pipeline, cfg):
        ephys = ephys_full_pipeline["ephys"]
        ephys.EphysChunk.ingest_chunks(cfg["experiment_name"])
        ephys.EphysBlockInfo.populate()

    def test_block_info_populated(self, ephys_test_blocks, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        blocks = len(ephys.EphysBlock & {"experiment_name": cfg["experiment_name"]})
        infos = len(ephys.EphysBlockInfo & {"experiment_name": cfg["experiment_name"]})
        assert infos == blocks

    def test_block_duration_correct(self, ephys_test_blocks, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        infos = (ephys.EphysBlockInfo & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        for info in infos:
            expected_hours = (info["block_end"] - info["block_start"]).total_seconds() / 3600
            assert abs(info["block_duration"] - expected_hours) < 0.01

    def test_block_chunks_associated(self, ephys_test_blocks, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        chunk_links = len(ephys.EphysBlockInfo.Chunk & {"experiment_name": cfg["experiment_name"]})
        assert chunk_links >= 1

    def test_channel_mappings_created(self, ephys_test_blocks, ephys_full_pipeline, ephys_golden_dataset_config):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        ephys = ephys_full_pipeline["ephys"]
        cfg = ephys_golden_dataset_config
        channels = len(ephys.EphysBlockInfo.Channel & {"experiment_name": cfg["experiment_name"]})
        assert channels == cfg["n_channels"]


class TestPreProcessing:
    """Verify PreProcessing.populate() output.

    PreProcessing reads raw ephys binary, selects electrode group channels,
    applies bandpass filter + common average reference, and writes
    recording.dat + si_recording.pkl.
    """

    def _ensure_prerequisites(self, ephys_full_pipeline, cfg):
        ephys = ephys_full_pipeline["ephys"]
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        ephys.EphysChunk.ingest_chunks(cfg["experiment_name"])
        ephys.EphysBlockInfo.populate()
        spike_sorting.PreProcessing.populate(display_progress=True, suppress_errors=False)

    def test_preprocessing_populated(
        self, ephys_sorting_setup, ephys_full_pipeline, ephys_golden_dataset_config, require_ephys_golden_data
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        count = len(spike_sorting.PreProcessing & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_recording_files_registered(
        self, ephys_sorting_setup, ephys_full_pipeline, ephys_golden_dataset_config, require_ephys_golden_data
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        files = (
            spike_sorting.PreProcessing.File
            & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()
        file_names = [f["file_name"] for f in files]
        assert any("si_recording.pkl" in fn for fn in file_names)

    def test_recording_binary_exists(
        self, ephys_sorting_setup, ephys_full_pipeline, ephys_golden_dataset_config, require_ephys_golden_data
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        key = (
            spike_sorting.SortingTask & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()[0]
        output_dir = spike_sorting.PreProcessing.infer_output_dir(key)
        recording_dat = output_dir.parent / "recording" / "recording.dat"
        assert recording_dat.exists()
        assert recording_dat.stat().st_size > 0


class TestPostProcessing:
    """Verify PostProcessing.populate() output.

    PostProcessing creates a SpikeInterface sorting_analyzer with computed
    extensions (waveforms, templates, spike_locations, quality_metrics).
    Depends on SpikeSorting being force-injected from golden data.
    """

    def _ensure_prerequisites(self, ephys_full_pipeline, cfg):
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)

    def test_postprocessing_populated(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        count = len(spike_sorting.PostProcessing & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_sorting_analyzer_created(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        output_dir = ephys_sorting_injected["output_dir"]
        analyzer_dir = output_dir / "sorting_analyzer"
        assert analyzer_dir.exists()
        assert any(analyzer_dir.iterdir())


class TestSortedSpikes:
    """Verify SortedSpikes.populate() output.

    SortedSpikes extracts unit info from the sorting_analyzer: spike counts,
    spike indices, electrode assignments, quality labels.
    """

    def _ensure_prerequisites(self, ephys_full_pipeline, cfg):
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)
        spike_sorting.SortedSpikes.populate(display_progress=True, suppress_errors=False)

    def test_sorted_spikes_populated(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        count = len(spike_sorting.SortedSpikes & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_unit_count(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        units = len(spike_sorting.SortedSpikes.Unit & {"experiment_name": cfg["experiment_name"]})
        assert units == cfg["expected_unit_count"]

    def test_spike_counts_reasonable(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        units = (
            spike_sorting.SortedSpikes.Unit & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()
        for u in units:
            assert u["spike_count"] >= 100
        total = sum(u["spike_count"] for u in units)
        assert total > 300_000

    def test_quality_labels_assigned(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        units = (
            spike_sorting.SortedSpikes.Unit & {"experiment_name": cfg["experiment_name"]}
        ).to_dicts()
        qualities = {u["unit_quality"] for u in units}
        assert qualities <= {"good", "mua", "noise", "ok", "n.a."}
        assert len(qualities) >= 1


class TestSyncedSpikes:
    """Verify clock-synchronized spike times.

    SyncedSpikes reads binary Clock files and HarpSync models to convert
    spike sample indices to absolute datetime timestamps.
    """

    def _ensure_prerequisites(self, ephys_full_pipeline, cfg):
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)
        spike_sorting.SortedSpikes.populate(display_progress=True, suppress_errors=False)
        spike_sorting.SyncedSpikes.populate(display_progress=True, suppress_errors=False)

    def test_synced_spikes_populated(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config
        count = len(spike_sorting.SyncedSpikes & {"experiment_name": cfg["experiment_name"]})
        assert count >= 1

    def test_spike_times_are_datetimes(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config

        import numpy as np

        units = (spike_sorting.SyncedSpikes.Unit & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        assert len(units) >= 1
        assert np.issubdtype(units[0]["spike_times"].dtype, np.datetime64)

    def test_spike_times_within_epoch_range(
        self, ephys_sorting_injected, ephys_full_pipeline, ephys_golden_dataset_config
    ):
        self._ensure_prerequisites(ephys_full_pipeline, ephys_golden_dataset_config)
        spike_sorting = ephys_full_pipeline["spike_sorting"]
        cfg = ephys_golden_dataset_config

        import numpy as np
        from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

        epoch_start = np.datetime64(parse_epoch_timestamp(cfg["epoch_dir"]))

        units = (spike_sorting.SyncedSpikes.Unit & {"experiment_name": cfg["experiment_name"]}).to_dicts()
        for unit in units:
            assert unit["spike_times"].min() >= epoch_start
