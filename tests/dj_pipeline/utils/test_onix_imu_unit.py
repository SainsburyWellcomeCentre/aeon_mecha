"""Unit tests for aeon/dj_pipeline/utils/onix_imu.py.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection — before any fixtures run —
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_STREAMS = {
    "Euler": ("x", "y", "z"),
    "GravityVector": ("x", "y", "z"),
    "LinearAcceleration": ("x", "y", "z"),
    "Quaternion": ("w", "x", "y", "z"),
}


def _write_bno_chunk(tmp_path, device_name, n, n_samples, seed=0):
    """Write synthetic Clock + 4 stream binaries into tmp_path/<device_name>/."""
    device_dir = tmp_path / device_name
    device_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Offset clocks by chunk index so each chunk has a distinct first sample
    # (required by tests that locate a chunk by its starting ONIX timestamp).
    clock = (((np.arange(n_samples) + 1) * 100) + n * 1_000_000).astype(np.uint64)
    (device_dir / f"{device_name}_Bno055_Clock_{n}.bin").write_bytes(clock.tobytes())

    payloads = {}
    for stream, cols in _STREAMS.items():
        data = rng.standard_normal((n_samples, len(cols))).astype(np.float32)
        (device_dir / f"{device_name}_Bno055_{stream}_{n}.bin").write_bytes(data.tobytes())
        payloads[stream] = data
    return device_dir, clock, payloads


def test_imu_columns_canonical_order():
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    assert IMU_COLUMNS == (
        "euler_x", "euler_y", "euler_z",
        "gravity_vector_x", "gravity_vector_y", "gravity_vector_z",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    )


def test_load_and_merge_bno055_returns_canonical_columns(tmp_path):
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS, load_and_merge_bno055

    device_name = "NeuropixelsV2Beta"
    device_dir, clock, payloads = _write_bno_chunk(tmp_path, device_name, n=2, n_samples=50)

    df = load_and_merge_bno055(device_dir, device_name, chunk_index=2)

    assert tuple(df.columns) == IMU_COLUMNS
    assert len(df) == 50
    assert df.index.dtype == np.uint64
    np.testing.assert_array_equal(df.index.values, clock)

    np.testing.assert_array_equal(
        df[["euler_x", "euler_y", "euler_z"]].values,
        payloads["Euler"],
    )
    np.testing.assert_array_equal(
        df[["quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"]].values,
        payloads["Quaternion"],
    )


def test_load_and_merge_bno055_raises_on_missing_stream_file(tmp_path):
    from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=10)
    (device_dir / f"{device_name}_Bno055_Quaternion_0.bin").unlink()

    with pytest.raises(FileNotFoundError):
        load_and_merge_bno055(device_dir, device_name, chunk_index=0)


def test_load_and_merge_bno055_validates_columns(tmp_path, monkeypatch):
    """If load_and_merge_bno055 produces unexpected columns, raise ValueError."""
    from aeon.dj_pipeline.utils import onix_imu
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS, load_and_merge_bno055

    monkeypatch.setattr(onix_imu, "IMU_COLUMNS", IMU_COLUMNS + ("phantom_extra",))
    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=10)

    with pytest.raises(ValueError, match="Bno055 stream column mismatch"):
        load_and_merge_bno055(device_dir, device_name, chunk_index=0)


def test_find_overlapping_bno055_chunks_window_inside_one_chunk(tmp_path):
    """A window fully contained in one chunk returns just that chunk."""
    from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

    device_name = "NeuropixelsV2Beta"
    device_dir, clock_n5, _ = _write_bno_chunk(tmp_path, device_name, n=5, n_samples=20)

    # Window entirely inside chunk 5's range
    first, last = int(clock_n5[0]), int(clock_n5[-1])
    result = find_overlapping_bno055_chunks(
        device_dir, device_name,
        onix_ts_start=first + 100,
        onix_ts_end=last - 100,
    )
    assert result == [5]


def test_find_overlapping_bno055_chunks_window_spans_multiple_chunks(tmp_path):
    """A window straddling 3 sequential chunks returns all 3 indices, sorted."""
    from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

    device_name = "NeuropixelsV2Beta"
    # Three chunks with monotonically-increasing ONIX timestamps
    _, clock_n2, _ = _write_bno_chunk(tmp_path, device_name, n=2, n_samples=20)
    _write_bno_chunk(tmp_path, device_name, n=3, n_samples=20, seed=1)
    _, clock_n4, _ = _write_bno_chunk(tmp_path, device_name, n=4, n_samples=20, seed=2)
    device_dir = tmp_path / device_name

    result = find_overlapping_bno055_chunks(
        device_dir, device_name,
        onix_ts_start=int(clock_n2[5]),   # inside chunk 2
        onix_ts_end=int(clock_n4[-5]),    # inside chunk 4
    )
    assert result == [2, 3, 4]


def test_find_overlapping_bno055_chunks_no_overlap_returns_empty(tmp_path):
    """A window with no overlap returns []."""
    from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

    device_name = "NeuropixelsV2Beta"
    _, clock_n0, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=20)
    device_dir = tmp_path / device_name

    # Window entirely after the chunk
    result = find_overlapping_bno055_chunks(
        device_dir, device_name,
        onix_ts_start=int(clock_n0[-1]) + 1_000_000,
        onix_ts_end=int(clock_n0[-1]) + 2_000_000,
    )
    assert result == []


def test_find_overlapping_bno055_chunks_no_files_returns_empty(tmp_path):
    """A device dir with no Bno055 binaries returns []."""
    from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

    device_name = "NeuropixelsV2Beta"
    device_dir = tmp_path / device_name
    device_dir.mkdir()

    assert find_overlapping_bno055_chunks(device_dir, device_name, 0, 999999) == []


def test_find_overlapping_bno055_chunks_window_edge_inclusive(tmp_path):
    """Boundary touch (window edge equals chunk first/last sample) counts as overlap."""
    from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

    device_name = "NeuropixelsV2Beta"
    _, clock_n0, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=20)
    device_dir = tmp_path / device_name

    # Window's end exactly equals chunk's first sample
    assert find_overlapping_bno055_chunks(
        device_dir, device_name,
        onix_ts_start=int(clock_n0[0]) - 100,
        onix_ts_end=int(clock_n0[0]),
    ) == [0]


# ---------------------------------------------------------------------------
# End-to-end pattern test: overlap → concat → filter.
#
# Mirrors what OnixImuChunk.make does in production, against staggered
# synthetic data that reproduces the real-world cadence mismatch:
# HarpSync sync windows partition the ONIX clock on one cadence, Bno055
# binary chunks partition it on another. The middle sync window straddles
# two Bno055 chunks, so its IMU DataFrame must be built by concatenating
# both chunks and filtering to the window's [start, end] range.
# ---------------------------------------------------------------------------


def _write_staggered_bno_chunk(device_dir, device_name, n, clock_arr, seed=0):
    """Write Clock + 4 stream binaries for a Bno055 chunk with explicit timestamps."""
    rng = np.random.default_rng(seed)
    n_samples = len(clock_arr)
    (device_dir / f"{device_name}_Bno055_Clock_{n}.bin").write_bytes(
        clock_arr.astype(np.uint64).tobytes()
    )
    payloads = {}
    for stream, cols in _STREAMS.items():
        data = rng.standard_normal((n_samples, len(cols))).astype(np.float32)
        (device_dir / f"{device_name}_Bno055_{stream}_{n}.bin").write_bytes(data.tobytes())
        payloads[stream] = data
    return payloads


def test_overlap_concat_filter_pattern_against_staggered_chunks(tmp_path):
    """Full pattern: HarpSync window straddles two Bno055 chunks of mismatched cadence.

    Lays out:
      - Bno055 chunk 0 ONIX range: [1000, 5000]  (50 samples, 80 ticks apart)
      - Bno055 chunk 1 ONIX range: [5100, 9100]  (50 samples, 80 ticks apart)

    And tests three sync windows:
      - W_left  = [800, 3000]  → overlaps only chunk 0; filtered to [1000, 3000]
      - W_mid   = [4000, 7000] → overlaps BOTH chunks; filtered to [4000, 7000]
      - W_right = [7500, 9500] → overlaps only chunk 1; filtered to [7500, 9100]

    Asserts the merged+filtered DataFrame for each window has the right index
    range and sample count — exactly what OnixImuChunk.make produces.
    """
    import pandas as pd

    from aeon.dj_pipeline.utils.onix_imu import (
        find_overlapping_bno055_chunks,
        load_and_merge_bno055,
    )

    device_name = "NeuropixelsV2Beta"
    device_dir = tmp_path / device_name
    device_dir.mkdir(parents=True)

    chunk_0_clocks = np.arange(1000, 5001, 80, dtype=np.uint64)   # 51 samples
    chunk_1_clocks = np.arange(5100, 9101, 80, dtype=np.uint64)   # 51 samples
    _write_staggered_bno_chunk(device_dir, device_name, 0, chunk_0_clocks, seed=0)
    _write_staggered_bno_chunk(device_dir, device_name, 1, chunk_1_clocks, seed=1)

    def _materialize(onix_ts_start, onix_ts_end):
        """Repro of OnixImuChunk.make's data-loading flow."""
        idxs = find_overlapping_bno055_chunks(
            device_dir, device_name, onix_ts_start, onix_ts_end
        )
        if not idxs:
            return pd.DataFrame()
        df = pd.concat(
            [load_and_merge_bno055(device_dir, device_name, n) for n in idxs]
        )
        return df[(df.index >= onix_ts_start) & (df.index <= onix_ts_end)]

    # Window that touches only the left chunk
    df_left = _materialize(800, 3000)
    assert not df_left.empty
    assert df_left.index.min() >= 1000
    assert df_left.index.max() <= 3000
    expected_left = np.sum((chunk_0_clocks >= 800) & (chunk_0_clocks <= 3000))
    assert len(df_left) == expected_left, (
        f"left-only window: expected {expected_left} samples, got {len(df_left)}"
    )

    # Window that straddles both chunks — the discriminating case
    df_mid = _materialize(4000, 7000)
    assert df_mid.index.min() >= 4000
    assert df_mid.index.max() <= 7000
    expected_mid = (
        int(np.sum((chunk_0_clocks >= 4000) & (chunk_0_clocks <= 7000)))
        + int(np.sum((chunk_1_clocks >= 4000) & (chunk_1_clocks <= 7000)))
    )
    assert len(df_mid) == expected_mid, (
        f"straddle window: expected {expected_mid} samples (sum across chunks), "
        f"got {len(df_mid)}"
    )
    # Sanity: indices are monotonically increasing (chunks concat'd in order)
    assert df_mid.index.is_monotonic_increasing

    # Window that touches only the right chunk
    df_right = _materialize(7500, 9500)
    assert df_right.index.min() >= 7500
    assert df_right.index.max() <= 9100
    expected_right = int(np.sum((chunk_1_clocks >= 7500) & (chunk_1_clocks <= 9100)))
    assert len(df_right) == expected_right


def test_overlap_concat_filter_pattern_empty_window(tmp_path):
    """Sync window with no overlapping Bno055 chunks → empty DataFrame, no errors."""
    import pandas as pd

    from aeon.dj_pipeline.utils.onix_imu import (
        find_overlapping_bno055_chunks,
        load_and_merge_bno055,
    )

    device_name = "NeuropixelsV2Beta"
    device_dir = tmp_path / device_name
    device_dir.mkdir(parents=True)
    _write_staggered_bno_chunk(
        device_dir, device_name, 0, np.arange(1000, 2001, 50, dtype=np.uint64), seed=0
    )

    # Window way before the only chunk
    idxs = find_overlapping_bno055_chunks(device_dir, device_name, 0, 500)
    assert idxs == []
    # Calling make's flow on this should produce empty without exception
    if not idxs:
        df = pd.DataFrame()
    else:
        df = pd.concat([load_and_merge_bno055(device_dir, device_name, n) for n in idxs])
    assert df.empty
