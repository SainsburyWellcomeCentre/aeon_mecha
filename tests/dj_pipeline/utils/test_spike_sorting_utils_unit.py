"""Unit tests for spike_sorting_utils pure helpers.

Covers analyzer dir resolution and non-numeric property filtering. Tests
target pure functions with no database dependency. Imports from
aeon.dj_pipeline are done inside test methods (not at module level) so pytest
collection does not trigger aeon/dj_pipeline/__init__.py schema activation.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestResolveAnalyzerDir:
    """resolve_analyzer_dir picks the binary dir if present, else the zarr dir."""

    def test_binary_dir_when_present(self, tmp_path):
        (tmp_path / "sorting_analyzer").mkdir()
        from aeon.dj_pipeline.utils.spike_sorting_utils import resolve_analyzer_dir

        assert resolve_analyzer_dir(tmp_path) == tmp_path / "sorting_analyzer"

    def test_zarr_dir_when_only_zarr_present(self, tmp_path):
        (tmp_path / "sorting_analyzer.zarr").mkdir()
        from aeon.dj_pipeline.utils.spike_sorting_utils import resolve_analyzer_dir

        assert resolve_analyzer_dir(tmp_path) == tmp_path / "sorting_analyzer.zarr"

    def test_raises_when_neither_present(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import resolve_analyzer_dir

        with pytest.raises(FileNotFoundError):
            resolve_analyzer_dir(tmp_path)

    def test_binary_dir_preferred_when_both_present(self, tmp_path):
        (tmp_path / "sorting_analyzer").mkdir()
        (tmp_path / "sorting_analyzer.zarr").mkdir()
        from aeon.dj_pipeline.utils.spike_sorting_utils import resolve_analyzer_dir

        assert resolve_analyzer_dir(tmp_path) == tmp_path / "sorting_analyzer"


class TestStripNonNumericProperties:
    """strip_non_numeric_properties keeps numeric properties and removes the rest."""

    def _make_recording(self):
        import spikeinterface as si

        traces = np.zeros((1000, 4), dtype="int16")
        return si.NumpyRecording(traces, sampling_frequency=30000.0)

    def test_keeps_numeric_strips_string(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import strip_non_numeric_properties

        rec = self._make_recording()
        rec.set_property("myfloat", np.array([1.0, 2.0, 3.0, 4.0]))  # kind 'f' -> keep
        rec.set_property("myint", np.array([0, 1, 0, 1], dtype="int64"))  # kind 'i' -> keep
        rec.set_property("mybool", np.array([True, False, True, False]))  # kind 'b' -> keep
        rec.set_property("mystr", np.array(["a", "b", "c", "d"]))  # kind 'U' -> strip

        strip_non_numeric_properties(rec)

        keys = set(rec.get_property_keys())
        assert "myfloat" in keys
        assert "myint" in keys
        assert "mybool" in keys
        assert "mystr" not in keys

    def test_keeps_uint_property(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import strip_non_numeric_properties

        rec = self._make_recording()
        rec.set_property("myuint", np.array([1, 2, 3, 4], dtype="uint32"))  # kind 'u' -> keep

        strip_non_numeric_properties(rec)

        assert "myuint" in set(rec.get_property_keys())


class TestForkSafeJobKwargs:
    """fork_safe_job_kwargs caps the worker count to the cgroup allocation, no fork."""

    def test_respects_cpu_affinity(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import fork_safe_job_kwargs

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
        assert fork_safe_job_kwargs("30s")["n_jobs"] == 4

    def test_caps_at_max_jobs(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import fork_safe_job_kwargs

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
        assert fork_safe_job_kwargs("30s", max_jobs=8)["n_jobs"] == 8

    def test_fallback_to_cpu_count_without_affinity(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import fork_safe_job_kwargs

        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        assert fork_safe_job_kwargs("30s")["n_jobs"] == 6

    def test_thread_pool_and_chunk_passthrough(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import fork_safe_job_kwargs

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
        kw = fork_safe_job_kwargs("2s")
        assert kw["pool_engine"] == "thread"
        assert kw["max_threads_per_worker"] == 1
        assert kw["chunk_duration"] == "2s"
