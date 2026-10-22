"""Helpers for ONIX-clocked IMU (Bno055) data loading.

Reusable by:
- ``OnixImuChunk.make()`` — for column-level summary stats during ingest.
- ``OnixStreamCodec.decode()`` — for lazy on-fetch DataFrame reconstruction.

Returns ONIX-clock-indexed DataFrames. HARP conversion is the caller's job
(via ``OnixImuChunk.synced_df`` or equivalent).
"""

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

IMU_COLUMNS: tuple[str, ...] = (
    "euler_x",
    "euler_y",
    "euler_z",
    "gravity_vector_x",
    "gravity_vector_y",
    "gravity_vector_z",
    "linear_acceleration_x",
    "linear_acceleration_y",
    "linear_acceleration_z",
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
)

# Arrow column -> canonical name. Verified against real data: the arrow
# EulerAngle_{Yaw,Pitch,Roll} ranges match the binary euler_{x,y,z} exactly
# ([0,360], +/-90, +/-180), and |Acceleration| ~ 4.9 against |Gravity| ~ 9.8
# confirms Acceleration is gravity-compensated, i.e. the binary linear_acceleration.
_ARROW_TO_CANONICAL: dict[str, str] = {
    "EulerAngle_Yaw": "euler_x",
    "EulerAngle_Pitch": "euler_y",
    "EulerAngle_Roll": "euler_z",
    "Gravity_X": "gravity_vector_x",
    "Gravity_Y": "gravity_vector_y",
    "Gravity_Z": "gravity_vector_z",
    "Acceleration_X": "linear_acceleration_x",
    "Acceleration_Y": "linear_acceleration_y",
    "Acceleration_Z": "linear_acceleration_z",
    "Quaternion_W": "quaternion_w",
    "Quaternion_X": "quaternion_x",
    "Quaternion_Y": "quaternion_y",
    "Quaternion_Z": "quaternion_z",
    "Calibration": "calibration",
    "Temperature": "temperature",
}

# Columns an arrow chunk carries beyond the binary format.
ARROW_EXTRA_COLUMNS: tuple[str, ...] = ("calibration", "temperature")

# Map dotmap stream class name -> column prefix used in IMU_COLUMNS
_PREFIX_BY_CLASS: dict[str, str] = {
    "Bno055Euler": "euler",
    "Bno055GravityVector": "gravity_vector",
    "Bno055LinearAcceleration": "linear_acceleration",
    "Bno055Quaternion": "quaternion",
}


def _load_bno055_arrow(path: Path) -> pd.DataFrame:
    """Load one Bno055 arrow chunk, renamed to the canonical column names.

    Returns the 13 shared ``IMU_COLUMNS`` plus ``ARROW_EXTRA_COLUMNS``
    (``calibration``, ``temperature``) -- 15 in all -- indexed by ONIX ``Clock``.
    """
    import pyarrow as pa
    from pyarrow.ipc import RecordBatchFileReader

    source = pa.memory_map(str(path), "r")
    try:
        reader = RecordBatchFileReader(source)
        table = pa.Table.from_batches(
            [reader.get_batch(i) for i in range(reader.num_record_batches)], schema=reader.schema
        )
        present = set(table.column_names)
        unknown = present - set(_ARROW_TO_CANONICAL) - {"Clock", "HubClock"}
        # Clock is required, not merely tolerated: everything downstream is aligned on
        # it, and leaving it out of this check trades the diagnostic below for a bare
        # KeyError four lines later.
        missing = (set(_ARROW_TO_CANONICAL) | {"Clock"}) - present
        if unknown or missing:
            raise ValueError(
                f"Unexpected Bno055 column set in {path.name}: "
                f"unknown={sorted(unknown)}, missing={sorted(missing)}. "
                f"The acquisition schema changed -- update _ARROW_TO_CANONICAL and IMU_COLUMNS."
            )
        data = {
            canonical: table[arrow_name].to_numpy(zero_copy_only=False)
            for arrow_name, canonical in _ARROW_TO_CANONICAL.items()
        }
        df = pd.DataFrame(data, index=table["Clock"].to_numpy(zero_copy_only=False))
    finally:
        source.close()
    return df[[*IMU_COLUMNS, *ARROW_EXTRA_COLUMNS]]


def load_and_merge_bno055(
    device_dir: Path,
    device_name: str,
    chunk_index: int,
) -> pd.DataFrame:
    """Load Bno055 Clock + 4 stream binaries, prefix-rename, merge on ONIX index.

    Args:
        device_dir: Path to the ONIX device directory containing the binaries.
        device_name: ``"NeuropixelsV2Beta"`` or ``"NeuropixelsV2"``.
        chunk_index: Integer suffix of the binary file group.

    Returns:
        ONIX-clock-indexed DataFrame with the 13 columns in :data:`IMU_COLUMNS`.

    Raises:
        FileNotFoundError: If the Clock or any expected stream binary is missing.
        ValueError: If the merged DataFrame's columns don't match :data:`IMU_COLUMNS`
            exactly — a sign of schema drift in ``aeon/schema/ephys.py``. An arrow
            chunk returns those columns plus :data:`ARROW_EXTRA_COLUMNS` instead.
    """
    # Lazy import — avoids triggering aeon.schema.ephys at module load time,
    # consistent with the pattern in aeon/dj_pipeline/ephys.py.
    from aeon.schema.ephys import social_ephys

    device_dir = Path(device_dir)

    arrow_file = device_dir / f"{device_name}_Bno055_{chunk_index}.arrow"
    if arrow_file.exists():
        return _load_bno055_arrow(arrow_file)

    clock_file = device_dir / f"{device_name}_Bno055_Clock_{chunk_index}.bin"
    if not clock_file.exists():
        raise FileNotFoundError(clock_file)
    onix_clock = np.fromfile(clock_file, dtype=np.uint64)

    streams = social_ephys[device_name]
    pieces: list[pd.DataFrame] = []
    for cls_name, prefix in _PREFIX_BY_CLASS.items():
        stream_suffix = cls_name.removeprefix("Bno055")
        bin_path = device_dir / f"{device_name}_Bno055_{stream_suffix}_{chunk_index}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(bin_path)
        reader = streams[cls_name]
        df = reader.read(bin_path)
        df.columns = [f"{prefix}_{c}" for c in df.columns]
        pieces.append(df)

    merged = pd.concat(pieces, axis=1)
    merged.index = onix_clock

    if set(merged.columns) != set(IMU_COLUMNS):
        raise ValueError(
            f"Bno055 stream column mismatch: expected {IMU_COLUMNS}, "
            f"got {tuple(merged.columns)}. Schema in aeon/schema/ephys.py "
            f"may have drifted — update OnixImuChunk + IMU_COLUMNS to match, "
            f"or revert the schema change."
        )

    return merged[list(IMU_COLUMNS)]


def concat_bno055_chunks(device_dir: Path, device_name: str, chunk_indices: Iterable[int]) -> pd.DataFrame:
    """Load and concatenate Bno055 chunks, refusing to mix column sets.

    An epoch written across a format change yields 13-column binary frames and
    15-column arrow frames. ``pd.concat`` would pad the difference with NaN and
    return a frame that looks whole, so mismatched sets raise instead.
    """
    # Materialise once: the argument may be any iterable, and it is used twice.
    chunk_indices = list(chunk_indices)
    if not chunk_indices:
        raise ValueError(
            f"No Bno055 chunk indices given for {device_dir}; callers must handle the "
            f"empty window themselves rather than concatenating nothing."
        )
    frames = [load_and_merge_bno055(device_dir, device_name, n) for n in chunk_indices]
    column_sets = {tuple(f.columns) for f in frames}
    if len(column_sets) > 1:
        raise ValueError(
            f"Bno055 chunks {chunk_indices} in {device_dir} have different column "
            f"sets {[sorted(c) for c in column_sets]} -- concatenating would pad with NaN. "
            f"This epoch mixes acquisition formats; ingest the halves separately."
        )
    return pd.concat(frames)


def _indices_in_both_formats(device_dir: Path, device_name: str) -> set[int]:
    """Chunk indices that have BOTH a Bno055 arrow file and a Bno055 Clock binary."""
    arrow = {
        int(m.group(1))
        for f in device_dir.glob(f"{device_name}_Bno055_*.arrow")
        if (m := re.search(r"_Bno055_(\d+)\.arrow$", f.name))
    }
    binary = {
        int(m.group(1))
        for f in device_dir.glob(f"{device_name}_Bno055_Clock_*.bin")
        if (m := re.search(r"_Clock_(\d+)\.bin$", f.name))
    }
    return arrow & binary


def find_overlapping_bno055_chunks(
    device_dir: Path,
    device_name: str,
    onix_ts_start: int,
    onix_ts_end: int,
) -> list[int]:
    """Return sorted Bno055 chunk indices whose ONIX range overlaps the window.

    Arrow chunks carry their ONIX clock as a column, read from the first and last
    record batch. Binary chunks keep a companion ``Bno055_Clock_N.bin``.

    For each ``Bno055_Clock_N.bin`` file, reads just the first and last uint64
    sample (the binaries are monotonic, so those two values fully bound the
    chunk's coverage — no need to scan the rest). A standard interval-overlap
    test selects chunks where
    ``first <= onix_ts_end and last >= onix_ts_start`` — inclusive on both ends.

    Why this exists: HarpSync sync windows (one per ``EphysSyncModel`` row,
    hourly cadence) and Bno055 binary chunks (~10 min, firmware-flushed)
    partition the ONIX clock on independent boundaries, so each sync window
    typically overlaps multiple Bno055 files. ``OnixImuChunk.make`` uses this
    helper to gather all overlapping chunks before concatenating + filtering.
    """
    device_dir = Path(device_dir)
    arrow_idx: set[int] = set()
    binary_idx: set[int] = set()

    # Arrow chunks keep their ONIX clock as a column; binary chunks have a companion
    # Clock_N.bin. Both are scanned: an epoch written across a format change has some
    # of each, and returning after the arrow pass would silently drop the binary half.
    from aeon.dj_pipeline.utils.arrow_io import EmptyArrowChunk, read_arrow_clock_bounds

    # Ambiguity is decided on file existence alone, BEFORE any overlap test.
    # load_and_merge_bno055 picks arrow because the file is there, not because its
    # clock matched: testing the collision after the overlap filter would let a
    # window selected on the binary clock be loaded from the arrow file, whose index
    # is elsewhere, and the caller would record an empty window as a real one.
    both = _indices_in_both_formats(device_dir, device_name)
    if both:
        raise ValueError(
            f"Bno055 chunk(s) {sorted(both)} exist in both arrow and binary form in "
            f"{device_dir}; which is authoritative is undefined. Remove one set."
        )

    for arrow_file in sorted(device_dir.glob(f"{device_name}_Bno055_*.arrow")):
        match = re.search(r"_Bno055_(\d+)\.arrow$", arrow_file.name)
        if not match:
            continue
        # A chunk whose clock cannot be read is malformed. Skipping it would make the
        # caller compute summary stats over a subset of the window and insert a row
        # that looks healthy, so let it raise. A chunk with no rows is a different
        # thing: it contributes nothing to any window, and the binary pass below skips
        # its equivalent, so failing here would block every sync window in the epoch.
        try:
            first, last = read_arrow_clock_bounds(arrow_file)
        except EmptyArrowChunk:
            continue
        if first <= onix_ts_end and last >= onix_ts_start:
            arrow_idx.add(int(match.group(1)))

    pattern = f"{device_name}_Bno055_Clock_*.bin"
    for clock_file in device_dir.glob(pattern):
        size = clock_file.stat().st_size
        # Floor to a uint64 boundary in case a trailing partial sample was
        # left behind by an interrupted acquisition. Skip files with no
        # whole samples.
        whole = (size // 8) * 8
        if whole < 8:
            continue
        match = re.search(r"_Clock_(\d+)\.bin$", clock_file.name)
        if not match:
            continue
        n = int(match.group(1))
        first = int(np.fromfile(clock_file, dtype=np.uint64, count=1)[0])
        if whole >= 16:
            with open(clock_file, "rb") as f:
                f.seek(whole - 8)
                last = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        else:
            last = first
        # Inclusive interval overlap test on both ends.
        if first <= onix_ts_end and last >= onix_ts_start:
            binary_idx.add(n)
    # Disjoint by construction: the collision check above already raised.
    return sorted(arrow_idx | binary_idx)
