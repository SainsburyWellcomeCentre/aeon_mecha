"""Helpers for ONIX-clocked IMU (Bno055) data loading.

Reusable by:
- ``OnixImuChunk.make()`` — for column-level summary stats during ingest.
- ``OnixStreamCodec.decode()`` — for lazy on-fetch DataFrame reconstruction.

Returns ONIX-clock-indexed DataFrames. HARP conversion is the caller's job
(via ``OnixImuChunk.synced_df`` or equivalent).
"""

import re
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

    Returns the 13 shared columns plus ``calibration`` and ``temperature``,
    indexed by the ONIX ``Clock`` column.
    """
    import pyarrow as pa
    from pyarrow.ipc import RecordBatchFileReader

    source = pa.memory_map(str(path), "r")
    try:
        reader = RecordBatchFileReader(source)
        table = pa.Table.from_batches([reader.get_batch(i) for i in range(reader.num_record_batches)])
        present = set(table.column_names)
        unknown = present - set(_ARROW_TO_CANONICAL) - {"Clock", "HubClock"}
        missing = set(_ARROW_TO_CANONICAL) - present
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
            exactly — a sign of schema drift in ``aeon/schema/ephys.py``.
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
    overlapping: list[int] = []

    arrow_chunks = sorted(device_dir.glob(f"{device_name}_Bno055_*.arrow"))
    if arrow_chunks:
        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        for arrow_file in arrow_chunks:
            match = re.search(r"_Bno055_(\d+)\.arrow$", arrow_file.name)
            if not match:
                continue
            try:
                first, last = read_arrow_clock_bounds(arrow_file)
            except (KeyError, ValueError):
                continue
            if first <= onix_ts_end and last >= onix_ts_start:
                overlapping.append(int(match.group(1)))
        return sorted(overlapping)

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
            overlapping.append(n)
    return sorted(overlapping)
