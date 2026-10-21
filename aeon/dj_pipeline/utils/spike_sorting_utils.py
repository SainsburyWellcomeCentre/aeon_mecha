"""Pure helpers for the spike sorting pipeline's zarr/binary intermediate handling.

These functions have no database or table dependencies, so they live here (rather
than in spike_sorting.py) to keep them importable and unit-testable without
activating the spike_sorting schema.
"""

import os
from pathlib import Path

import datajoint as dj
import numpy as np

logger = dj.logger


MAX_FORK_SAFE_JOBS = 32  # default cap on SpikeInterface workers


def fork_safe_job_kwargs(chunk_duration: str, max_jobs: int | None = MAX_FORK_SAFE_JOBS) -> dict:
    """SpikeInterface job_kwargs that avoid the SLURM fork/oversubscription hang.

    Uses a thread pool (no fork) and a worker count taken from the cgroup CPU
    allocation (``os.sched_getaffinity``), capped at ``max_jobs`` -- rather than
    ``n_jobs=-1``, which SpikeInterface resolves against ``os.cpu_count()`` (every
    core on the node, not the allocation) and then thrashes or fork-deadlocks.

    ``max_jobs`` may come from a paramset's ``n_jobs``; anything other than a
    positive int, e.g. ``-1`` or ``None``, falls back to ``MAX_FORK_SAFE_JOBS``.
    """
    # Check if OS can access `sched_getaffinity`` method
    sched_affinity = getattr(os, "sched_getaffinity", None)
    n_jobs = len(sched_affinity(0)) if sched_affinity else (os.cpu_count() or 1)
    if not isinstance(max_jobs, int) or max_jobs < 1:
        max_jobs = MAX_FORK_SAFE_JOBS
    return {
        "n_jobs": max(1, min(max_jobs, n_jobs)),
        "pool_engine": "thread",
        "max_threads_per_worker": 1,
        "chunk_duration": chunk_duration,
    }


def extend_clock(onix_ts: np.ndarray, n_samples: int, max_extrapolation: int, name: str) -> np.ndarray:
    """Return ONIX clock times covering ``n_samples`` AmplifierData samples.

    A Clock file longer than its AmplifierData file is returned as is: sample k still
    sits at Clock[k]. A short one is extrapolated linearly at the tick rate of its last
    ``max_extrapolation`` samples, over at most that many samples. Either mismatch
    (a file lost its tail, e.g. in a copying error) is logged under ``name``.

    Raises:
        ValueError: If more than ``max_extrapolation`` samples are missing, or the
            clock has fewer than 2 samples to take a tick rate from.
    """
    n_missing = n_samples - len(onix_ts)
    if n_missing < 0:
        logger.warning(
            f"{name}: AmplifierData file is {-n_missing} samples shorter than its Clock file; "
            f"its chunk ends early."
        )
    if n_missing <= 0:
        return onix_ts
    if n_missing > max_extrapolation or len(onix_ts) < 2:
        raise ValueError(
            f"{name}: cannot extrapolate a {len(onix_ts)}-sample clock over {n_missing} missing "
            f"samples (at most {max_extrapolation})."
        )
    logger.warning(
        f"{name}: Clock file is {n_missing} samples shorter than its AmplifierData file; "
        f"extrapolating the clock over them."
    )
    tail = onix_ts[-max_extrapolation:]
    ticks_per_sample = (float(tail[-1]) - float(tail[0])) / (len(tail) - 1)
    extra = np.round(ticks_per_sample * np.arange(1, n_missing + 1)).astype(np.uint64)
    return np.concatenate([onix_ts, tail[-1] + extra])


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
