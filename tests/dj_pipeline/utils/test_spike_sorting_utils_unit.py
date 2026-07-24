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


class TestDeletePreprocessedRecording:
    """delete_preprocessed_recording removes recording.zarr / recording.dat, keeps everything else."""

    def test_removes_recording_zarr(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import delete_preprocessed_recording

        zarr = tmp_path / "recording.zarr"
        zarr.mkdir()
        (zarr / "chunk").write_bytes(b"data")

        deleted = delete_preprocessed_recording(tmp_path)

        assert not zarr.exists()
        assert deleted == ["recording.zarr"]

    def test_removes_recording_dat(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import delete_preprocessed_recording

        dat = tmp_path / "recording.dat"
        dat.write_bytes(b"binary")

        deleted = delete_preprocessed_recording(tmp_path)

        assert not dat.exists()
        assert deleted == ["recording.dat"]

    def test_keeps_pkl_and_other_files(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import delete_preprocessed_recording

        (tmp_path / "recording.zarr").mkdir()
        pkl = tmp_path / "si_recording.pkl"
        pkl.write_bytes(b"pickle")

        delete_preprocessed_recording(tmp_path)

        assert pkl.exists(), "si_recording.pkl must be preserved (needed to regenerate)"

    def test_noop_when_nothing_to_delete(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import delete_preprocessed_recording

        assert delete_preprocessed_recording(tmp_path) == []


class TestRequirePreprocessedRecording:
    """require_preprocessed_recording raises an actionable error when the recording is gone."""

    def test_raises_when_missing(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import require_preprocessed_recording

        missing = tmp_path / "recording.zarr"
        with pytest.raises(FileNotFoundError) as exc:
            require_preprocessed_recording(missing)
        # message must guide the user to the recovery path
        assert "PreProcessing" in str(exc.value)

    def test_noop_when_present(self, tmp_path):
        from aeon.dj_pipeline.utils.spike_sorting_utils import require_preprocessed_recording

        present = tmp_path / "recording.zarr"
        present.mkdir()
        require_preprocessed_recording(present)  # should not raise


class TestSafeNJobs:
    """safe_n_jobs respects the cgroup CPU allocation and never oversubscribes."""

    def test_respects_cpu_affinity(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import safe_n_jobs

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
        assert safe_n_jobs(max_jobs=8) == 4

    def test_caps_at_max_jobs(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import safe_n_jobs

        # a big node (64 cores) must not spawn 64 workers -- the os.cpu_count() trap
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
        assert safe_n_jobs(max_jobs=8) == 8

    def test_floor_of_one(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import safe_n_jobs

        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(), raising=False)
        assert safe_n_jobs(max_jobs=8) == 1

    def test_fallback_to_cpu_count_without_affinity(self, monkeypatch):
        import os

        from aeon.dj_pipeline.utils.spike_sorting_utils import safe_n_jobs

        # non-Linux (no os.sched_getaffinity) falls back to os.cpu_count()
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(os, "cpu_count", lambda: 6)
        assert safe_n_jobs(max_jobs=8) == 6


class TestIsSafeToDeleteSharedRecording:
    """A shared recording.zarr is safe to delete only if no sibling is preprocessed-but-not-sorted."""

    def test_no_siblings_is_safe(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import is_safe_to_delete_shared_recording

        assert is_safe_to_delete_shared_recording([]) is True

    def test_sibling_preprocessed_and_sorted_is_safe(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import is_safe_to_delete_shared_recording

        assert is_safe_to_delete_shared_recording([{"preprocessed": True, "sorted": True}]) is True

    def test_sibling_preprocessed_not_sorted_is_unsafe(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import is_safe_to_delete_shared_recording

        assert is_safe_to_delete_shared_recording([{"preprocessed": True, "sorted": False}]) is False

    def test_sibling_not_preprocessed_is_safe(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import is_safe_to_delete_shared_recording

        # not yet preprocessed -> its future PreProcessing will regenerate the file
        assert is_safe_to_delete_shared_recording([{"preprocessed": False, "sorted": False}]) is True

    def test_any_pending_sibling_makes_it_unsafe(self):
        from aeon.dj_pipeline.utils.spike_sorting_utils import is_safe_to_delete_shared_recording

        siblings = [
            {"preprocessed": True, "sorted": True},
            {"preprocessed": True, "sorted": False},
        ]
        assert is_safe_to_delete_shared_recording(siblings) is False
