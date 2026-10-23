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
    """Load one Bno055 arrow chunk under the canonical column names.

    ``IMU_COLUMNS`` plus ``ARROW_EXTRA_COLUMNS`` -- 15 in all -- indexed by ONIX ``Clock``.
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
                f"Unexpected Bno055 column set in {path.name}: unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}. Update _ARROW_TO_CANONICAL and IMU_COLUMNS."
            )
        # Same hazard read_arrow_column guards: a null upcasts the column to float64.
        null_columns = sorted(name for name in (*_ARROW_TO_CANONICAL, "Clock") if table[name].null_count)
        if null_columns:
            raise ValueError(f"Null values in {path.name} column(s) {null_columns}.")
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

    Binary frames have 13 columns and arrow 15; ``pd.concat`` would NaN-pad the
    difference into a frame that looks whole.
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
            f"Bno055 chunks {chunk_indices} in {device_dir} have different column sets "
            f"{[sorted(c) for c in column_sets]}; ingest the formats separately."
        )
    return pd.concat(frames)


def _bno055_chunk_files(device_dir: Path, device_name: str) -> tuple[dict[int, Path], dict[int, Path]]:
    """One pass over the device directory: ({index: arrow file}, {index: clock binary}).

    A clock binary with no whole uint64 sample is ignored: an interrupted acquisition
    leaves those behind and neither caller would read one.
    """
    arrow = {
        int(m.group(1)): f
        for f in sorted(device_dir.glob(f"{device_name}_Bno055_*.arrow"))
        if (m := re.search(r"_Bno055_(\d+)\.arrow$", f.name))
    }
    binary = {
        int(m.group(1)): f
        for f in sorted(device_dir.glob(f"{device_name}_Bno055_Clock_*.bin"))
        if (m := re.search(r"_Clock_(\d+)\.bin$", f.name)) and f.stat().st_size >= 8
    }
    return arrow, binary


def _binary_clock_bounds(clock_file: Path) -> tuple[int, int]:
    """First and last ONIX tick of a Clock binary; the file is monotonic."""
    # Floor to a uint64 boundary: an interrupted acquisition can leave a partial sample.
    whole_bytes = (clock_file.stat().st_size // 8) * 8
    first = int(np.fromfile(clock_file, dtype=np.uint64, count=1)[0])
    if whole_bytes < 16:
        return first, first
    with open(clock_file, "rb") as f:
        f.seek(whole_bytes - 8)
        return first, int(np.frombuffer(f.read(8), dtype=np.uint64)[0])


def find_overlapping_bno055_chunks(
    device_dir: Path,
    device_name: str,
    onix_ts_start: int,
    onix_ts_end: int,
) -> list[int]:
    """Return sorted Bno055 chunk indices whose ONIX range overlaps the window.

    Bounds come from the arrow ``Clock`` column or the companion ``Clock_N.bin``;
    overlap is inclusive both ends. Sync windows and Bno055 chunks sit on independent
    boundaries, so one window usually spans several chunks.
    """
    from aeon.dj_pipeline.utils.arrow_io import EmptyArrowChunk, read_arrow_clock_bounds

    device_dir = Path(device_dir)
    arrow_files, binary_files = _bno055_chunk_files(device_dir, device_name)

    # Decided on which files exist, before any overlap test: loading picks arrow on
    # existence, so a window matched on the binary clock would be loaded from the arrow
    # chunk and filtered away -- an empty window recorded as a real one.
    if both := sorted(arrow_files.keys() & binary_files.keys()):
        raise ValueError(
            f"Bno055 chunk(s) {both} exist in both arrow and binary form in {device_dir}; "
            f"which is authoritative is undefined. Remove one set."
        )

    overlapping = set()
    for index, arrow_file in arrow_files.items():
        # A chunk with no rows contributes to no window, as an empty clock binary does.
        # A chunk whose clock cannot be read at all is malformed and must raise, or the
        # caller computes its stats over part of the window and inserts a healthy row.
        try:
            first, last = read_arrow_clock_bounds(arrow_file)
        except EmptyArrowChunk:
            continue
        if first <= onix_ts_end and last >= onix_ts_start:
            overlapping.add(index)
    for index, clock_file in binary_files.items():
        first, last = _binary_clock_bounds(clock_file)
        if first <= onix_ts_end and last >= onix_ts_start:
            overlapping.add(index)

    return sorted(overlapping)
