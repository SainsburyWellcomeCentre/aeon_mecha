"""Pure helpers for the spike sorting pipeline's zarr/binary intermediate handling.

These functions have no database or table dependencies, so they live here (rather
than in spike_sorting.py) to keep them importable and unit-testable without
activating the spike_sorting schema.
"""

import importlib.util
import os
import warnings
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


### Arrow reader zone ###
# This code will eventually be upstreamed into SpikeInterface.
# Track progress here: https://github.com/SpikeInterface/spikeinterface/pull/4778

# Put everything in a horrible try/except so that the codebase can stil
# run, even if SpikeInterface is not installed. Cannot import within class
# definitions since classes inherit from base SpikeInterface classes

try:
    from spikeinterface.core import BaseRecording, BaseRecordingSegment
    from spikeinterface.core.core_tools import define_function_from_class

    class BonsaiOnixArrowRecordingSegment(BaseRecordingSegment):
        def __init__(self, filepath, channel_ids, batch_len, **time_kwargs):
            BaseRecordingSegment.__init__(self, **time_kwargs)

            from pyarrow import memory_map
            from pyarrow.ipc import RecordBatchFileReader

            self._num_samples = None

            self._source = memory_map(filepath, "r")
            self._reader = RecordBatchFileReader(self._source)
            self.batch_len = batch_len

            self._all_channel_ids = channel_ids

        def get_num_samples(self) -> int:
            """Returns the number of samples in this signal block.

            Returns:
                SampleIndex : Number of samples in the signal block
            """
            # cache num_samples, since it requires a computation to find them.
            if self._num_samples is None:
                # arrow has fast access to number of samples per batch
                num_samples = sum(
                    self._reader.get_batch(i).num_rows for i in range(self._reader.num_record_batches)
                )
                self._num_samples = num_samples

            return self._num_samples

        def get_traces(
            self,
            start_frame: int | None = None,
            end_frame: int | None = None,
            channel_indices: list[int | str] | None = None,
        ) -> np.ndarray:
            """Materialises actual data traces."""
            if channel_indices is None:
                channel_ids = list(self._all_channel_ids)
            else:
                channel_ids = list(self._all_channel_ids[channel_indices])

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = self.get_num_samples()

            import pyarrow as pa

            # Arrow saves data in "batch"es, in the time dimension, which we can
            # load individually. We need to figure out which batches our requested
            # samples are in.

            batch_size = self.batch_len
            first_batch_idx = start_frame // batch_size
            last_batch_idx = (end_frame - 1) // batch_size

            # This is super easy if our samples are in a single batch
            if first_batch_idx == last_batch_idx:
                batch = self._reader.get_batch(first_batch_idx)
                local_start = start_frame % batch_size
                sliced = batch.slice(local_start, end_frame - start_frame)
                return np.column_stack(
                    [sliced.column(c).to_numpy(zero_copy_only=False) for c in channel_ids]
                )

            # Otherwise, we find all batches, then grab the data
            slices = []
            for b_idx in range(first_batch_idx, last_batch_idx + 1):
                batch = self._reader.get_batch(b_idx)
                b_start = b_idx * batch_size

                local_start = max(0, start_frame - b_start)
                local_end = min(batch.num_rows, end_frame - b_start)

                slices.append(batch.slice(local_start, local_end - local_start).select(channel_ids))

            table = pa.Table.from_batches(slices)
            return np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in channel_ids])

    class BonsaiOnixArrowRecording(BaseRecording):
        """Recording class for the openephys arrow format.

        Parameters
        ----------
        file_path : str
            Path to the directory where the zarr array is stored
        sampling_frequency : float
            The sampling frequency
        stream_name : str, default: AmplifierData
            The stream name of the data you want to load. By default, the ephys AP stream is
            called "AmplifierData".
        gain_to_uV : float or array-like, default: None
            The gain to apply to the traces
        offset_to_uV : float or array-like, default: None
            The offset to apply to the traces
        is_filtered : bool or None, default: None
            If True, the recording is assumed to be filtered. If None, is_filtered is not set.
        storage_options : dict or None: None
            Storage options passed to the `zarr.open` function

        Returns:
        --------
        recording : ZarrArrayRecording
            The recording Extractor
        """

        def __init__(
            self,
            file_path: str | Path,
            sampling_frequency: float,
            stream_name="AmplifierData",
            gain_to_uV: float | np.ndarray | None = None,
            offset_to_uV: float | np.ndarray | None = None,
            is_filtered: bool | None = None,
        ):
            if importlib.util.find_spec("pyarrow") is None:
                raise ImportError("You need to add `pyarrow` to your environment to open .arrow files")
            else:
                from pyarrow import memory_map
                from pyarrow.ipc import RecordBatchFileReader

            source = memory_map(file_path, "r")
            reader = RecordBatchFileReader(source)

            stream_names = reader.schema.names
            if sum([stream_name == name for name in stream_names]) == 1:
                # there is a unique stream matching the name
                channel_ids = [stream_name]
            else:
                channel_ids = [name for name in stream_names if stream_name in name]
            if len(channel_ids) == 0:
                raise ValueError(f"Cannot find any data with `stream_name` = {stream_name}")

            first_batch = reader.get_batch(0)
            batch_len = first_batch.num_rows

            one_channel_index = stream_names.index(channel_ids[0])

            # Arrow uses it's own DataType. For ints and floats, it converts to numpy dtype without issue
            ephys_type = reader.schema[one_channel_index].type
            numpy_type = np.dtype(str(ephys_type))

            source.close()

            BaseRecording.__init__(
                self, sampling_frequency=sampling_frequency, channel_ids=channel_ids, dtype=numpy_type
            )

            rec_segment = BonsaiOnixArrowRecordingSegment(
                file_path,
                batch_len=batch_len,
                sampling_frequency=sampling_frequency,
                channel_ids=np.array(channel_ids),
            )

            self.add_recording_segment(rec_segment)

            if is_filtered is not None:
                self.annotate(is_filtered=is_filtered)

            if gain_to_uV is not None:
                self.set_channel_gains(gain_to_uV)

            if offset_to_uV is not None:
                self.set_channel_offsets(offset_to_uV)

            self._kwargs = {
                "file_path": str(Path(file_path).absolute()),
                "sampling_frequency": sampling_frequency,
                # Without this, to_dict()/load() and parallel workers rebuild the
                # recording with the default stream and silently read a different one.
                "stream_name": stream_name,
                "gain_to_uV": gain_to_uV,
                "offset_to_uV": offset_to_uV,
                "is_filtered": is_filtered,
            }

    read_bonsai_onix_arrow = define_function_from_class(
        source_class=BonsaiOnixArrowRecording, name="read_bonsai_onix_arrow"
    )

except ModuleNotFoundError:
    warnings.warn("SpikeInterface is not installed", stacklevel=2)
