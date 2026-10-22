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
    """IMU_COLUMNS lists the BNO055 channels in canonical order."""
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    assert IMU_COLUMNS == (
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


def test_load_and_merge_bno055_returns_canonical_columns(tmp_path):
    """Merged BNO055 data has the canonical columns, indexed by ONIX clock."""
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
    """A missing BNO055 stream file raises FileNotFoundError."""
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
        device_dir,
        device_name,
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
        device_dir,
        device_name,
        onix_ts_start=int(clock_n2[5]),  # inside chunk 2
        onix_ts_end=int(clock_n4[-5]),  # inside chunk 4
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
        device_dir,
        device_name,
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
        device_dir,
        device_name,
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
    (device_dir / f"{device_name}_Bno055_Clock_{n}.bin").write_bytes(clock_arr.astype(np.uint64).tobytes())
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

    chunk_0_clocks = np.arange(1000, 5001, 80, dtype=np.uint64)  # 51 samples
    chunk_1_clocks = np.arange(5100, 9101, 80, dtype=np.uint64)  # 51 samples
    _write_staggered_bno_chunk(device_dir, device_name, 0, chunk_0_clocks, seed=0)
    _write_staggered_bno_chunk(device_dir, device_name, 1, chunk_1_clocks, seed=1)

    def _materialize(onix_ts_start, onix_ts_end):
        """Repro of OnixImuChunk.make's data-loading flow."""
        idxs = find_overlapping_bno055_chunks(device_dir, device_name, onix_ts_start, onix_ts_end)
        if not idxs:
            return pd.DataFrame()
        df = pd.concat([load_and_merge_bno055(device_dir, device_name, n) for n in idxs])
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
    expected_mid = int(np.sum((chunk_0_clocks >= 4000) & (chunk_0_clocks <= 7000))) + int(
        np.sum((chunk_1_clocks >= 4000) & (chunk_1_clocks <= 7000))
    )
    assert len(df_mid) == expected_mid, (
        f"straddle window: expected {expected_mid} samples (sum across chunks), got {len(df_mid)}"
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


ARROW_IMU_COLUMNS = (
    "HubClock",
    "Clock",
    "Calibration",
    "Temperature",
    "Gravity_X",
    "Gravity_Y",
    "Gravity_Z",
    "Acceleration_X",
    "Acceleration_Y",
    "Acceleration_Z",
    "Quaternion_X",
    "Quaternion_Y",
    "Quaternion_Z",
    "Quaternion_W",
    "EulerAngle_Yaw",
    "EulerAngle_Pitch",
    "EulerAngle_Roll",
)


def _write_bno055_arrow(device_dir, device_name, chunk_index, n=20, columns=ARROW_IMU_COLUMNS):
    """Write one Bno055_<n>.arrow with the real arrow column set."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc

    device_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for i, c in enumerate(columns):
        if c == "Clock":
            data[c] = (np.arange(n, dtype=np.uint64) + 1) * 100 + chunk_index * 10_000
        elif c == "HubClock":
            # Must differ from Clock, or reading the wrong clock column passes.
            data[c] = (np.arange(n, dtype=np.uint64) + 1) * 100 + chunk_index * 10_000 - 7
        elif c == "Calibration":
            data[c] = np.full(n, 15, dtype=np.uint8)
        elif c == "Temperature":
            data[c] = np.full(n, 87, dtype=np.int32)
        else:
            # Each float column gets its own value. An identical ramp everywhere
            # would make a Gravity <-> Acceleration <-> Quaternion swap in
            # _ARROW_TO_CANONICAL pass every assertion in this file.
            data[c] = np.full(n, 100.0 + i, dtype=np.float32)
    schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in data.items()])
    path = device_dir / f"{device_name}_Bno055_{chunk_index}.arrow"
    with ipc.new_file(path, schema) as w:
        w.write_batch(pa.RecordBatch.from_arrays([pa.array(v) for v in data.values()], schema=schema))
    return path


class TestBno055Arrow:
    """Arrow IMU chunks map onto the canonical column names, plus two new fields."""

    def test_maps_arrow_columns_to_canonical_names(self, tmp_path):
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS, load_and_merge_bno055

        _write_bno055_arrow(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        df = load_and_merge_bno055(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        for col in IMU_COLUMNS:
            assert col in df.columns, f"{col} missing from arrow-loaded frame"

    def test_euler_mapping_is_yaw_pitch_roll(self, tmp_path):
        """Verified empirically against real data: x=Yaw, y=Pitch, z=Roll."""
        import numpy as np

        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        d = tmp_path / "NeuropixelsV2"
        d.mkdir(parents=True, exist_ok=True)
        # written here with distinguishable per-column values
        import pyarrow as pa
        import pyarrow.ipc as ipc

        vals = {"EulerAngle_Yaw": 10.0, "EulerAngle_Pitch": 20.0, "EulerAngle_Roll": 30.0}
        data = {}
        for c in ARROW_IMU_COLUMNS:
            if c == "Clock":
                data[c] = np.arange(3, dtype=np.uint64) + 10
            elif c == "HubClock":
                data[c] = np.arange(3, dtype=np.uint64) + 1
            elif c == "Calibration":
                data[c] = np.full(3, 15, dtype=np.uint8)
            elif c == "Temperature":
                data[c] = np.full(3, 87, dtype=np.int32)
            else:
                data[c] = np.full(3, vals.get(c, 0.0), dtype=np.float32)
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in data.items()])
        with ipc.new_file(d / "NeuropixelsV2_Bno055_0.arrow", schema) as w:
            w.write_batch(pa.RecordBatch.from_arrays([pa.array(v) for v in data.values()], schema=schema))

        df = load_and_merge_bno055(d, "NeuropixelsV2", 0)
        assert df["euler_x"].iloc[0] == 10.0
        assert df["euler_y"].iloc[0] == 20.0
        assert df["euler_z"].iloc[0] == 30.0

    def test_index_is_clock_not_hubclock(self, tmp_path):
        """The ONIX index is Clock, never HubClock.

        HubClock is the hub's own counter and is offset from Clock, so indexing on it
        shifts every sample against the sync regression.
        """
        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        _write_bno055_arrow(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0, n=4)
        df = load_and_merge_bno055(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        assert list(df.index) == [100, 200, 300, 400]

    def test_every_arrow_column_lands_on_its_own_canonical_name(self, tmp_path):
        """Pins the whole mapping, not just the euler triple.

        A swap between gravity, linear_acceleration and quaternion is the
        mislabelled-channel anti-goal: the frame looks healthy and every summary
        stat is computed over the wrong axis.

        The expected pairs are written out here rather than read from
        _ARROW_TO_CANONICAL -- taking them from the dict under test makes the
        assertion follow any swap made to it, which is how the first version of
        this test passed a Gravity_X <-> Acceleration_X mutation.
        """
        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        expected_pairs = {
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
        _write_bno055_arrow(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0, n=4)
        df = load_and_merge_bno055(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        for arrow_name, canonical in expected_pairs.items():
            # _write_bno055_arrow fills float column i with 100.0 + i
            expected = 100.0 + ARROW_IMU_COLUMNS.index(arrow_name)
            got = df[canonical].iloc[0]
            assert got == expected, (
                f"{canonical} holds {got}, which is {ARROW_IMU_COLUMNS[int(got) - 100]}, not {arrow_name}"
            )

    def test_calibration_and_temperature_present(self, tmp_path):
        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        _write_bno055_arrow(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        df = load_and_merge_bno055(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)
        assert df["calibration"].iloc[0] == 15
        assert df["temperature"].iloc[0] == 87

    def test_unexpected_column_set_raises(self, tmp_path):
        """Review Focus #5: a frame matching neither schema must not be ingested silently."""
        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        _write_bno055_arrow(
            tmp_path / "NeuropixelsV2",
            "NeuropixelsV2",
            0,
            columns=("Clock", "HubClock", "Gravity_X", "Mystery_Column"),
        )
        with pytest.raises(ValueError, match="column"):
            load_and_merge_bno055(tmp_path / "NeuropixelsV2", "NeuropixelsV2", 0)

    def test_malformed_chunk_raises_rather_than_vanishing(self, tmp_path):
        """A chunk whose Clock cannot be read must not be silently dropped.

        Dropping it makes OnixImuChunk.make compute sample_count and every summary
        stat over a subset of the window and insert a row that looks healthy -- the
        spec's silent-wrong-data anti-goal, worse than an empty result.
        """
        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        for n in range(3):
            _write_bno055_arrow(d, "NeuropixelsV2", n)
        # chunk 1 loses its Clock column, as a truncated acquisition would
        _write_bno055_arrow(
            d, "NeuropixelsV2", 1, columns=tuple(c for c in ARROW_IMU_COLUMNS if c != "Clock")
        )
        with pytest.raises(KeyError, match="Clock"):
            find_overlapping_bno055_chunks(d, "NeuropixelsV2", 0, 100_000)

    def test_missing_clock_column_is_named_rather_than_a_bare_keyerror(self, tmp_path):
        """Clock is what everything downstream aligns on, so its absence is schema drift.

        Tolerating it in the column-set check trades the diagnostic for a bare
        KeyError raised four lines later, with no hint of which file or which schema.
        """
        from aeon.dj_pipeline.utils.onix_imu import load_and_merge_bno055

        d = tmp_path / "NeuropixelsV2"
        _write_bno055_arrow(
            d, "NeuropixelsV2", 0, columns=tuple(c for c in ARROW_IMU_COLUMNS if c != "Clock")
        )
        with pytest.raises(ValueError, match="column set"):
            load_and_merge_bno055(d, "NeuropixelsV2", 0)

    def test_empty_arrow_chunk_is_skipped_not_fatal(self, tmp_path):
        """A zero-row chunk contributes to no window, as the binary pass already assumes.

        Raising on it instead blocks every OnixImuChunk in the epoch, including sync
        windows nowhere near the empty chunk.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        _write_bno055_arrow(d, "NeuropixelsV2", 0)
        schema = pa.schema([(c, pa.uint64() if "lock" in c else pa.float32()) for c in ARROW_IMU_COLUMNS])
        with ipc.new_file(d / "NeuropixelsV2_Bno055_1.arrow", schema):
            pass
        assert find_overlapping_bno055_chunks(d, "NeuropixelsV2", 0, 100_000) == [0]

    def test_overlap_finder_sees_arrow_chunks(self, tmp_path):
        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        for n in range(3):
            _write_bno055_arrow(d, "NeuropixelsV2", n)
        # chunk n spans [100 + n*10000, 2000 + n*10000]
        assert find_overlapping_bno055_chunks(d, "NeuropixelsV2", 10_050, 10_200) == [1]
        assert find_overlapping_bno055_chunks(d, "NeuropixelsV2", 0, 100_000) == [0, 1, 2]


class TestMixedFormatImu:
    """An epoch written across a format change has arrow and binary chunks side by side."""

    def test_same_index_in_both_formats_raises(self, tmp_path):
        """A chunk present as both arrow and binary is ambiguous, not twice as much data.

        Scanning both globs (the round-1 mixed-format fix) made a same-index collision
        reachable. Without this, the chunk is returned twice, loaded twice via the arrow
        branch, and concatenated into a frame of double length with a non-monotonic index.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        _write_bno055_arrow(d, "NeuropixelsV2", 0)
        np.array([1, 10**9], dtype=np.uint64).tofile(d / "NeuropixelsV2_Bno055_Clock_0.bin")

        with pytest.raises(ValueError, match="both arrow and binary"):
            find_overlapping_bno055_chunks(d, "NeuropixelsV2", 0, 10**12)

    def test_concat_refuses_to_mix_column_sets(self, tmp_path):
        """13-column binary and 15-column arrow frames must not be concatenated.

        pd.concat pads the difference with NaN and returns a frame that looks whole,
        so OnixImuChunk would insert summary stats over half-NaN columns and the
        codec's column-count check would then read the padded width as correct.
        """
        from aeon.dj_pipeline.utils.onix_imu import concat_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        _write_bno055_arrow(d, "NeuropixelsV2", 0)
        _write_staggered_bno_chunk(d, "NeuropixelsV2", 1, np.arange(1, 21, dtype=np.uint64), seed=1)

        with pytest.raises(ValueError, match="different column"):
            concat_bno055_chunks(d, "NeuropixelsV2", [0, 1])

    def test_concat_accepts_a_homogeneous_set(self, tmp_path):
        from aeon.dj_pipeline.utils.onix_imu import ARROW_EXTRA_COLUMNS, IMU_COLUMNS, concat_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        for n in (0, 1):
            _write_bno055_arrow(d, "NeuropixelsV2", n)
        df = concat_bno055_chunks(d, "NeuropixelsV2", [0, 1])
        assert tuple(df.columns) == IMU_COLUMNS + ARROW_EXTRA_COLUMNS
        assert len(df) == 40

    def test_collision_raises_even_when_only_one_side_overlaps(self, tmp_path):
        """The ambiguity raise must key on file existence, not on window overlap.

        load_and_merge_bno055 picks arrow purely because the file exists. If the
        collision is only tested after the overlap filter, a window matched on the
        BINARY clock is then loaded from the ARROW file: the frame's index is
        somewhere else entirely, the window filter keeps nothing, and OnixImuChunk
        inserts sample_count=0 for a window whose data is sitting on disk.
        """
        import numpy as np

        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        # arrow chunk 0 lives far from the window; binary chunk 0 covers it
        _write_bno055_arrow(d, "NeuropixelsV2", 0)  # Clock 100..2000
        _write_staggered_bno_chunk(
            d, "NeuropixelsV2", 0, np.arange(900_000, 900_020, dtype=np.uint64), seed=0
        )

        with pytest.raises(ValueError, match="both arrow and binary"):
            find_overlapping_bno055_chunks(d, "NeuropixelsV2", 900_000, 900_019)

    def test_both_formats_are_found(self, tmp_path):
        import numpy as np

        from aeon.dj_pipeline.utils.onix_imu import find_overlapping_bno055_chunks

        d = tmp_path / "NeuropixelsV2"
        _write_bno055_arrow(d, "NeuropixelsV2", 0)  # chunk 0 arrow
        # chunk 1 binary: Clock_1.bin spanning a later window
        np.array([20_000, 30_000], dtype=np.uint64).tofile(d / "NeuropixelsV2_Bno055_Clock_1.bin")

        found = find_overlapping_bno055_chunks(d, "NeuropixelsV2", 0, 100_000)
        assert found == [0, 1], f"mixed-format epoch returned {found}; a half was dropped"
