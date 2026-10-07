"""Pure helpers for the spike sorting pipeline's zarr/binary intermediate handling.

These functions have no database or table dependencies, so they live here (rather
than in spike_sorting.py) to keep them importable and unit-testable without
activating the spike_sorting schema.
"""

import os
from pathlib import Path

import numpy as np


def fork_safe_job_kwargs(chunk_duration: str, max_jobs: int = 8) -> dict:
    """SpikeInterface job_kwargs that avoid the SLURM fork/oversubscription hang.

    Uses a thread pool (no fork) and a worker count taken from the cgroup CPU
    allocation (``os.sched_getaffinity``), capped at ``max_jobs`` -- rather than
    ``n_jobs=-1``, which SpikeInterface resolves against ``os.cpu_count()`` (every
    core on the node, not the allocation) and then thrashes or fork-deadlocks.
    """
    try:
        n_jobs = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        n_jobs = os.cpu_count() or 1
    return {
        "n_jobs": max(1, min(max_jobs, n_jobs)),
        "pool_engine": "thread",
        "max_threads_per_worker": 1,
        "chunk_duration": chunk_duration,
    }


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


# TODO: move into aeon_api
def load_recording(
    root_directory,
    ephys_paths,
    probe_path,
    file_type="arrow",
):
    """"""

    from probeinterface import read_probeinterface
    from spikeinterface.core import concatenate_recordings
    from spikeinterface.extractors import read_binary
    from spikeinterface.preprocessing import unsigned_to_signed

    fs_hz = 30_000
    gain_to_uV = 3.05176
    offset_to_uV = 0
    rec_dtype = np.uint16
    num_channels = 384

    recordings = []
    for ephys_path in ephys_paths:
        if file_type == "arrow":
            recording = read_bonsai_onix_arrow(
                root_directory / ephys_path,
                sampling_frequency=fs_hz,
                gain_to_uV=gain_to_uV,
                offset_to_uV=offset_to_uV,
            )
        elif file_type == "binary":
            recording = read_binary(
                root_directory / ephys_path,
                sampling_frequency=fs_hz,
                dtype=rec_dtype,
                num_channels=num_channels,
                gain_to_uV=gain_to_uV,
                offset_to_uV=offset_to_uV,
            )
        else:
            raise ValueError("Only support file types 'arrow' and 'binary'")

        recordings.append(recording)

    concatenated_recording = concatenate_recordings(recordings)
    if concatenated_recording.dtype == "u":
        concatenated_recording = unsigned_to_signed(concatenated_recording)

    probe = read_probeinterface(probe_path)

    concatenated_recording.set_probe(probe=probe, in_place=True)

    return concatenated_recording


def read_bonsai_onix_arrow(path, sampling_frequency, gain_to_uV, offset_to_uV):
    return
