"""Unit tests for aeon/dj_pipeline/utils/onix_imu.py.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection — before any fixtures run —
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

from pathlib import Path

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

    clock = ((np.arange(n_samples) + 1) * 100).astype(np.uint64)
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


def test_locate_bno055_chunk_index_finds_match(tmp_path):
    from aeon.dj_pipeline.utils.onix_imu import locate_bno055_chunk_index

    device_name = "NeuropixelsV2Beta"
    device_dir, clock_n5, _ = _write_bno_chunk(tmp_path, device_name, n=5, n_samples=20)
    _write_bno_chunk(tmp_path, device_name, n=6, n_samples=20, seed=1)

    expected_first = int(clock_n5[0])
    assert locate_bno055_chunk_index(device_dir, device_name, expected_first) == 5


def test_locate_bno055_chunk_index_returns_none_on_miss(tmp_path):
    from aeon.dj_pipeline.utils.onix_imu import locate_bno055_chunk_index

    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=20)

    assert locate_bno055_chunk_index(device_dir, device_name, 99999999) is None


def test_locate_bno055_chunk_index_handles_empty_clock(tmp_path):
    from aeon.dj_pipeline.utils.onix_imu import locate_bno055_chunk_index

    device_name = "NeuropixelsV2Beta"
    device_dir = tmp_path / device_name
    device_dir.mkdir()
    (device_dir / f"{device_name}_Bno055_Clock_0.bin").write_bytes(b"")

    assert locate_bno055_chunk_index(device_dir, device_name, 100) is None
