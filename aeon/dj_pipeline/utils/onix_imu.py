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

from aeon.schema.ephys import social_ephys

IMU_COLUMNS: tuple[str, ...] = (
    "euler_x", "euler_y", "euler_z",
    "gravity_vector_x", "gravity_vector_y", "gravity_vector_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
)

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
    device_dir = Path(device_dir)
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
    """Return Bno055 chunk indices whose ONIX range overlaps [onix_ts_start, onix_ts_end].

    HarpSync CSVs (one per ``EphysSyncModel`` row) partition the ONIX clock on
    an hourly cadence set by the Bonsai workflow on the host. Bno055 binary
    files partition the same clock on a firmware-determined cadence (~10 min
    per file). The two partitions don't align, so each ``EphysSyncModel``
    window overlaps multiple Bno055 files and each Bno055 file may straddle
    multiple sync windows.

    Each returned chunk is one whose [first_sample, last_sample] window
    intersects [onix_ts_start, onix_ts_end].

    Args:
        device_dir: Path to the ONIX device directory containing the binaries.
        device_name: ``"NeuropixelsV2Beta"`` or ``"NeuropixelsV2"``.
        onix_ts_start: Start of the ONIX window of interest (inclusive).
        onix_ts_end: End of the ONIX window of interest (inclusive).

    Returns:
        Sorted list of chunk indices whose data spans into the window.
        Empty list if no Bno055 files exist or none overlap.
    """
    device_dir = Path(device_dir)
    pattern = f"{device_name}_Bno055_Clock_*.bin"
    overlapping: list[int] = []
    for clock_file in device_dir.glob(pattern):
        size = clock_file.stat().st_size
        if size < 8:
            continue  # need at least one sample
        match = re.search(r"_Clock_(\d+)\.bin$", clock_file.name)
        if not match:
            continue
        n = int(match.group(1))
        first = int(np.fromfile(clock_file, dtype=np.uint64, count=1)[0])
        if size >= 16:
            with open(clock_file, "rb") as f:
                f.seek(size - 8)
                last = int(np.frombuffer(f.read(8), dtype=np.uint64)[0])
        else:
            last = first
        # Standard half-open interval overlap test, treated as inclusive on both ends.
        if first <= onix_ts_end and last >= onix_ts_start:
            overlapping.append(n)
    return sorted(overlapping)
