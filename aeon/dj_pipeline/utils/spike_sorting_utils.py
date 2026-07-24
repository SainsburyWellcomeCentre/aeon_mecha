"""Pure helpers for the spike sorting pipeline's zarr/binary intermediate handling.

These functions have no database or table dependencies, so they live here (rather
than in spike_sorting.py) to keep them importable and unit-testable without
activating the spike_sorting schema.
"""

import os
import shutil
from pathlib import Path

import numpy as np

# job_kwargs that avoid the fork/oversubscription hang: a cgroup-aware, capped worker
# count on a thread pool (no fork), with one BLAS thread per worker. Callers add
# chunk_duration. See safe_n_jobs for why n_jobs=-1 / the fork engine is unsafe on SLURM.
FORK_SAFE_JOB_KWARGS = {"pool_engine": "thread", "max_threads_per_worker": 1}


def safe_n_jobs(max_jobs: int = 8) -> int:
    """Number of parallel workers that respects the SLURM/cgroup CPU allocation.

    SpikeInterface resolves ``n_jobs`` (and ``n_jobs=-1``) against ``os.cpu_count()``,
    which on SLURM returns every physical core on the node rather than the cgroup
    allocation. That makes ``n_jobs=-1`` spawn dozens of workers on a few allocated
    CPUs, which either thrashes the node to a standstill or fork-deadlocks on BLAS.

    This uses the schedulable-CPU count (cgroup-aware on Linux via
    ``os.sched_getaffinity``) instead, capped at ``max_jobs``. The cap is the real
    safety net: even if the scheduler does not pin CPUs, we never exceed ``max_jobs``.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # os.sched_getaffinity is Linux-only
        available = os.cpu_count() or 1
    return max(1, min(max_jobs, available))


def resolve_analyzer_dir(output_dir: Path) -> Path:
    """Find sorting analyzer directory, checking both binary and zarr paths.

    Raises:
        FileNotFoundError: If neither sorting_analyzer nor sorting_analyzer.zarr exists.
    """
    analyzer_dir = output_dir / "sorting_analyzer"
    if analyzer_dir.exists():
        return analyzer_dir
    analyzer_dir = output_dir / "sorting_analyzer.zarr"
    if analyzer_dir.exists():
        return analyzer_dir
    raise FileNotFoundError(
        f"Sorting analyzer directory not found in {output_dir} "
        f"(checked sorting_analyzer and sorting_analyzer.zarr). "
        f"Please verify the key is correct and that PreProcessing has been run for this block."
    )


def strip_non_numeric_properties(si_recording) -> None:
    """Remove non-numeric recording properties that zarr v2 cannot serialize."""
    for prop in list(si_recording.get_property_keys()):
        values = si_recording.get_property(prop)
        if np.asarray(values).dtype.kind not in ("f", "i", "u", "b"):
            si_recording.delete_property(prop)


def delete_preprocessed_recording(recording_dir: Path) -> list[str]:
    """Delete the preprocessed-recording intermediate(s) from ``recording_dir``.

    Removes ``recording.zarr`` (a directory) and/or ``recording.dat`` (a file) if
    present, and leaves ``si_recording.pkl`` and everything else intact (the pkl is
    small, database-registered, and needed to regenerate the recording). Returns the
    names that were deleted, for logging. A no-op (returns ``[]``) if neither exists.
    """
    recording_dir = Path(recording_dir)
    deleted: list[str] = []
    for name in ("recording.zarr", "recording.dat"):
        target = recording_dir / name
        if target.is_dir():
            shutil.rmtree(target)
            deleted.append(name)
        elif target.exists():
            target.unlink()
            deleted.append(name)
    return deleted


def require_preprocessed_recording(recording_path: Path) -> None:
    """Raise an actionable error if the preprocessed recording is missing.

    The preprocessed recording is deleted after a successful spike sort to save
    space, so a missing file here usually means a re-sort is being attempted on a
    block whose recording was already reclaimed. Point the user at the recovery path
    instead of letting a bare loader error surface.
    """
    recording_path = Path(recording_path)
    if not recording_path.exists():
        raise FileNotFoundError(
            f"Preprocessed recording not found at {recording_path}. It is deleted after a "
            "successful spike sort to reclaim space. To re-sort this block, delete its "
            "PreProcessing entry (which cascades to SpikeSorting and the downstream steps) and "
            "re-run PreProcessing.populate() to regenerate the recording."
        )


def is_safe_to_delete_shared_recording(sibling_states) -> bool:
    """Decide whether the shared preprocessed recording can be deleted.

    ``recording.zarr`` is shared by every sorting paramset for the same block and
    electrode group (it lives at the electrode-group level, not per paramset). It is
    safe to delete only once no sibling task still needs it: a sibling that is
    preprocessed but not yet sorted still needs the file, whereas a not-yet-preprocessed
    sibling will regenerate it through its own PreProcessing run.

    Args:
        sibling_states: iterable of ``{"preprocessed": bool, "sorted": bool}`` dicts for
            the sibling sorting tasks (excluding the current one).
    """
    return not any(s["preprocessed"] and not s["sorted"] for s in sibling_states)
