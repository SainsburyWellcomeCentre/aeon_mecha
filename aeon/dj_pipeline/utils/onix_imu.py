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
import pyarrow.feather as feather

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

IMU_arrow_to_DJ = {
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
}

# Map dotmap stream class name -> column prefix used in IMU_COLUMNS
_PREFIX_BY_CLASS: dict[str, str] = {
    "Bno055Euler": "euler",
    "Bno055GravityVector": "gravity_vector",
    "Bno055LinearAcceleration": "linear_acceleration",
    "Bno055Quaternion": "quaternion",
}


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
    clock_file = device_dir / f"{device_name}_Bno055_Clock_{chunk_index}.bin"

    if not clock_file.exists():
        clock_file = device_dir / f"{device_name}_Bno055_{chunk_index}.arrow"

    if not clock_file.exists():
        raise FileNotFoundError(clock_file)

    if clock_file.suffix == ".arrow":
        bno55_data = feather.read_table(clock_file)
        bno55_df = bno55_data.to_pandas()
        bno55_df.index = bno55_df["Clock"].values
        reduced_bno55 = bno55_df.rename(columns=IMU_arrow_to_DJ)[list(IMU_arrow_to_DJ.values())]
        return reduced_bno55

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
    clock_files = [
        *device_dir.glob(f"{device_name}_Bno055_Clock_*.bin"),
        *device_dir.glob(f"{device_name}_Bno055_*.arrow"),
    ]

    overlapping: list[int] = []
    for clock_file in clock_files:
        chunk_suffix = clock_file.stem.rsplit("_", 1)[-1]
        if not chunk_suffix.isdigit():
            continue
        n = int(chunk_suffix)
        if clock_file.suffix == ".arrow":
            table = feather.read_table(clock_file)
            clock_array = table["Clock"].to_numpy()
            first = clock_array[0]
            last = clock_array[-1]

        else:
            size = clock_file.stat().st_size
            # Floor to a uint64 boundary in case a trailing partial sample was
            # left behind by an interrupted acquisition. Skip files with no
            # whole samples.
            whole = (size // 8) * 8
            if whole < 8:
                continue
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
