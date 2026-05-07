# ONIX IMU Pipeline & Ephys SyncModel Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the per-chunk HARP↔ONIX sync regression to a first-class table (`EphysSyncModel`), then add a downstream `OnixImuChunk` table that surfaces Bno055 IMU streams from the ONIX-clocked binaries via a structural-only `<aeon_onix_stream>` codec and an explicit `synced_df` helper.

**Architecture:** Three-table dependency chain — `EphysEpoch` → `EphysSyncModel` → `EphysChunk`/`OnixImuChunk`. `EphysSyncModel` is `dj.Manual` populated by a re-runnable `ingest()` classmethod that walks `*_HarpSync_*.csv`. `EphysChunk` stays `dj.Manual` (1↔N to SyncModel for the straddling case), with its `SyncModel` Part downgraded to a pure FK link. `OnixImuChunk` is `dj.Imported` with default `key_source = EphysSyncModel`, one row per sync window with all four Bno055 streams merged on the shared sample index. The codec is structural-only (no regression); HARP conversion is opt-in via `OnixImuChunk.synced_df(key)`.

**Tech Stack:** Python 3.11+, DataJoint 2.x (`<blob>`/`<attach>` storage, codecs), scikit-learn `LinearRegression`, joblib serialization, pandas/numpy, dotmap (`social_ephys` reader hierarchy in `aeon/schema/ephys.py`), pytest + testcontainers MySQL.

**Spec:** `docs/specs/SPEC_ONIX_IMU_PIPELINE.md`

**Branch:** `tn/onix-imu-pipeline` (already created, base `tn/env-stream-tables`)

**Conventions:**
- All Python invocations via `uv run <cmd>` (never naked).
- Commits: no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" footer.
- Pre-commit hooks must pass before each commit.
- Test tiers per `docs/specs/SPEC_TESTING.md`: unit (no DB), integration (testcontainers MySQL), specialized (golden datasets).

---

## File Structure

| Path | Action | Responsibility |
|------|--------|----------------|
| `aeon/schema/ephys.py` | Modify | Extend `HarpSyncModel.Reader.read()` to surface observed `harp_start`/`harp_end`/`n_samples` |
| `aeon/dj_pipeline/utils/onix_imu.py` | Create | `IMU_COLUMNS` constant, `load_and_merge_bno055`, `locate_bno055_chunk_index` |
| `aeon/dj_pipeline/utils/codec.py` | Modify | Add `OnixStreamCodec` (registered as `<aeon_onix_stream>`) |
| `aeon/dj_pipeline/__init__.py` | Modify | Register `OnixStreamCodec` alongside `AeonStreamCodec` |
| `aeon/dj_pipeline/ephys.py` | Modify | Add `EphysSyncModel`. Refactor `EphysChunk.SyncModel` to link-only Part. Thin `ingest_chunks`. Add `OnixImuChunk` + `synced_df`. |
| `aeon/dj_pipeline/utils/ephys_utils.py` | Modify | Trim sync-fitting/dump logic out of `process_ephys_file` |
| `tests/schema/test_ephys_reader.py` | Create | Unit tests for `HarpSyncModel.Reader.read()` extension |
| `tests/dj_pipeline/utils/test_onix_imu.py` | Create | Unit tests for `IMU_COLUMNS`, `load_and_merge_bno055`, `locate_bno055_chunk_index` |
| `tests/dj_pipeline/utils/test_onix_codec.py` | Create | Unit + integration tests for `OnixStreamCodec` |
| `tests/dj_pipeline/test_onix_imu_pipeline.py` | Create | Integration tests for `EphysSyncModel`, `EphysChunk` (refactored), `OnixImuChunk` |

---

## Tasks

### Task 1: Extend `HarpSyncModel.Reader.read()` to expose observed HARP bounds and sample count

The new `EphysSyncModel` needs `harp_start`/`harp_end` (observed, not predicted) and `n_samples` from each HarpSync CSV. The existing reader returns only `clock_start`, `clock_end`, `model`, `r2`. Additive change — backward-compatible.

**Files:**
- Modify: `aeon/schema/ephys.py:35-58`
- Create: `tests/schema/test_ephys_reader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/schema/test_ephys_reader.py`:

```python
"""Unit tests for aeon/schema/ephys.py readers."""

import csv
from pathlib import Path

import numpy as np
import pytest

from aeon.schema.ephys import HarpSyncModel


@pytest.fixture
def synthetic_harp_csv(tmp_path):
    """Write a HarpSync CSV with three known sync samples and return its path."""
    csv_path = tmp_path / "NeuropixelsV2Beta_HarpSync_2024-06-04T11-00-00.csv"
    rows = [
        {"clock": 1000, "hub_clock": 0, "harp_time": 3000.5},
        {"clock": 2000, "hub_clock": 1, "harp_time": 3001.5},
        {"clock": 3000, "hub_clock": 2, "harp_time": 3002.5},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clock", "hub_clock", "harp_time"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_read_returns_observed_harp_bounds_and_sample_count(synthetic_harp_csv):
    reader = HarpSyncModel.Reader(pattern="NeuropixelsV2Beta")
    df = reader.read(synthetic_harp_csv)

    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["clock_start"]) == 1000
    assert int(row["clock_end"]) == 3000
    assert float(row["harp_start"]) == pytest.approx(3000.5)
    assert float(row["harp_end"]) == pytest.approx(3002.5)
    assert int(row["n_samples"]) == 3
    assert row["model"] is not None
    assert 0.0 <= float(row["r2"]) <= 1.0


def test_read_drops_na_rows_before_counting(synthetic_harp_csv, tmp_path):
    """Rows with NaN are dropped; n_samples reflects post-dropna count."""
    csv_path = tmp_path / "NeuropixelsV2Beta_HarpSync_2024-06-04T12-00-00.csv"
    with open(synthetic_harp_csv) as src, open(csv_path, "w") as dst:
        dst.write(src.read())
        # Append a row with empty harp_time → NaN after parse
        dst.write("4000,3,\n")

    reader = HarpSyncModel.Reader(pattern="NeuropixelsV2Beta")
    df = reader.read(csv_path)

    row = df.iloc[0]
    assert int(row["n_samples"]) == 3
    assert int(row["clock_end"]) == 3000  # last non-NaN row's clock
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/schema/test_ephys_reader.py -v
```

Expected: FAIL with `KeyError: 'harp_start'` (or similar) on the first assertion past the existing fields.

- [ ] **Step 3: Implement the reader extension**

Edit `aeon/schema/ephys.py:35-58`. Replace the existing `read()` method with:

```python
def read(self, file):
    data = super().read(file).dropna()
    onix_clock = data.clock.values.reshape(-1, 1)
    harp_time = data.harp_time.values.reshape(-1, 1)

    model = LinearRegression().fit(onix_clock, harp_time)
    r2 = model.score(onix_clock, harp_time)
    chunk_info = file.name.split("_")[-1]
    epoch = datetime.strptime(chunk_info, "%Y-%m-%dT%H-%M-%S.csv")
    return pd.DataFrame(
        index=[epoch],
        data={
            "clock_start": onix_clock[0],
            "clock_end": onix_clock[-1],
            "harp_start": harp_time[0],
            "harp_end": harp_time[-1],
            "n_samples": len(data),
            "model": [model],
            "r2": [r2],
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/schema/test_ephys_reader.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Verify no downstream breakage**

The existing `process_ephys_file` consumes `clock_start`/`clock_end`/`model`/`r2`. New fields are additive and won't break anything, but confirm:

```bash
uv run pytest tests/ -x -q 2>&1 | tail -20
```

Expected: existing tests pass (or at least don't regress on the reader change).

- [ ] **Step 6: Commit**

```bash
git add aeon/schema/ephys.py tests/schema/test_ephys_reader.py
git commit -m "feat(schema): expose observed harp bounds and sample count from HarpSyncModel reader"
```

---

### Task 2: Add `IMU_COLUMNS` constant and `load_and_merge_bno055` helper

The codec and `OnixImuChunk.make()` both need to load + prefix-rename + concat the four Bno055 streams. Single source of truth in a new utility module. Validates merged columns against `IMU_COLUMNS` and reorders to canonical order.

**Files:**
- Create: `aeon/dj_pipeline/utils/onix_imu.py`
- Create: `tests/dj_pipeline/utils/test_onix_imu.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dj_pipeline/utils/test_onix_imu.py`:

```python
"""Unit tests for aeon/dj_pipeline/utils/onix_imu.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeon.dj_pipeline.utils.onix_imu import (
    IMU_COLUMNS,
    load_and_merge_bno055,
)


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

    # ONIX timestamps (uint64), monotonically increasing
    clock = ((np.arange(n_samples) + 1) * 100).astype(np.uint64)
    (device_dir / f"{device_name}_Bno055_Clock_{n}.bin").write_bytes(clock.tobytes())

    payloads = {}
    for stream, cols in _STREAMS.items():
        data = rng.standard_normal((n_samples, len(cols))).astype(np.float32)
        (device_dir / f"{device_name}_Bno055_{stream}_{n}.bin").write_bytes(data.tobytes())
        payloads[stream] = data
    return device_dir, clock, payloads


def test_imu_columns_canonical_order():
    """IMU_COLUMNS is the canonical column order (matches OnixImuChunk schema)."""
    assert IMU_COLUMNS == (
        "euler_x", "euler_y", "euler_z",
        "gravity_vector_x", "gravity_vector_y", "gravity_vector_z",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    )


def test_load_and_merge_bno055_returns_canonical_columns(tmp_path):
    device_name = "NeuropixelsV2Beta"
    device_dir, clock, payloads = _write_bno_chunk(tmp_path, device_name, n=2, n_samples=50)

    df = load_and_merge_bno055(device_dir, device_name, chunk_index=2)

    assert tuple(df.columns) == IMU_COLUMNS
    assert len(df) == 50
    assert df.index.dtype == np.uint64
    np.testing.assert_array_equal(df.index.values, clock)

    # Spot-check that prefix-renaming preserved values
    np.testing.assert_array_equal(
        df[["euler_x", "euler_y", "euler_z"]].values,
        payloads["Euler"],
    )
    np.testing.assert_array_equal(
        df[["quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"]].values,
        payloads["Quaternion"],
    )


def test_load_and_merge_bno055_raises_on_missing_stream_file(tmp_path):
    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=10)
    (device_dir / f"{device_name}_Bno055_Quaternion_0.bin").unlink()

    with pytest.raises(FileNotFoundError):
        load_and_merge_bno055(device_dir, device_name, chunk_index=0)


def test_load_and_merge_bno055_validates_columns(tmp_path, monkeypatch):
    """If load_and_merge_bno055 produces unexpected columns, raise ValueError."""
    from aeon.dj_pipeline.utils import onix_imu

    monkeypatch.setattr(onix_imu, "IMU_COLUMNS", IMU_COLUMNS + ("phantom_extra",))
    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=10)

    with pytest.raises(ValueError, match="Bno055 stream column mismatch"):
        load_and_merge_bno055(device_dir, device_name, chunk_index=0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_imu.py -v
```

Expected: FAIL with `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Implement `aeon/dj_pipeline/utils/onix_imu.py`**

Create the file with this content:

```python
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
        chunk_index: Integer suffix of the binary file group (e.g., ``Clock_2.bin`` → 2).

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
        stream_suffix = cls_name.removeprefix("Bno055")  # "Euler", "GravityVector", ...
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


def locate_bno055_chunk_index(
    device_dir: Path,
    device_name: str,
    onix_ts_start: int,
) -> int | None:
    """Find the Bno055_Clock_N.bin file whose first sample matches ``onix_ts_start``.

    Args:
        device_dir: Path to the ONIX device directory.
        device_name: ``"NeuropixelsV2Beta"`` or ``"NeuropixelsV2"``.
        onix_ts_start: Expected first ONIX clock value (from ``EphysSyncModel.onix_ts_start``).

    Returns:
        The integer chunk index ``N``, or ``None`` if no Clock file's first
        sample matches.
    """
    device_dir = Path(device_dir)
    pattern = f"{device_name}_Bno055_Clock_*.bin"
    for clock_file in device_dir.glob(pattern):
        first_clock_arr = np.fromfile(clock_file, dtype=np.uint64, count=1)
        if len(first_clock_arr) == 0:
            continue
        if int(first_clock_arr[0]) == int(onix_ts_start):
            match = re.search(r"_Clock_(\d+)\.bin$", clock_file.name)
            if match:
                return int(match.group(1))
    return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_imu.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Add tests for `locate_bno055_chunk_index`**

Append to `tests/dj_pipeline/utils/test_onix_imu.py`:

```python
from aeon.dj_pipeline.utils.onix_imu import locate_bno055_chunk_index


def test_locate_bno055_chunk_index_finds_match(tmp_path):
    device_name = "NeuropixelsV2Beta"
    device_dir, clock_n5, _ = _write_bno_chunk(tmp_path, device_name, n=5, n_samples=20)
    _write_bno_chunk(tmp_path, device_name, n=6, n_samples=20, seed=1)

    # Read the actual first ONIX clock from chunk 5 to feed in
    expected_first = int(clock_n5[0])

    assert locate_bno055_chunk_index(device_dir, device_name, expected_first) == 5


def test_locate_bno055_chunk_index_returns_none_on_miss(tmp_path):
    device_name = "NeuropixelsV2Beta"
    device_dir, _, _ = _write_bno_chunk(tmp_path, device_name, n=0, n_samples=20)

    assert locate_bno055_chunk_index(device_dir, device_name, 99999999) is None


def test_locate_bno055_chunk_index_handles_empty_clock(tmp_path):
    device_name = "NeuropixelsV2Beta"
    device_dir = tmp_path / device_name
    device_dir.mkdir()
    (device_dir / f"{device_name}_Bno055_Clock_0.bin").write_bytes(b"")

    assert locate_bno055_chunk_index(device_dir, device_name, 100) is None
```

- [ ] **Step 6: Run all tests**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_imu.py -v
```

Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add aeon/dj_pipeline/utils/onix_imu.py tests/dj_pipeline/utils/test_onix_imu.py
git commit -m "feat(utils): add onix_imu helpers for Bno055 stream loading and chunk index lookup"
```

---

### Task 3: Add `OnixStreamCodec` and register it

Structural-only codec for ONIX-clocked IMU streams. Stored JSON: `experiment_name`, `epoch_start`, `sync_start`, `device_name`, `stream_group`. Decode loads + prefix-renames + concats Bno055 binaries via `load_and_merge_bno055`. Returns ONIX-clock-indexed DataFrame.

**Files:**
- Modify: `aeon/dj_pipeline/utils/codec.py`
- Modify: `aeon/dj_pipeline/__init__.py`
- Create: `tests/dj_pipeline/utils/test_onix_codec.py`

- [ ] **Step 1: Write the failing test (encode validation)**

Create `tests/dj_pipeline/utils/test_onix_codec.py`:

```python
"""Tests for OnixStreamCodec."""

import numpy as np
import pytest


class TestOnixStreamCodecEncode:
    def test_encodes_valid_dict(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        ref = {
            "experiment_name": "exp01",
            "epoch_start": "2024-06-04 10:24:07",
            "sync_start": "2024-06-04 11:00:00",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055",
        }
        encoded = codec.encode(ref)
        assert encoded == ref

    def test_rejects_non_dict(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(TypeError, match="OnixStreamCodec expects a dict"):
            codec.encode("not-a-dict")

    def test_rejects_dict_missing_keys(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"experiment_name": "exp01"})

    def test_codec_name(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        assert OnixStreamCodec.name == "aeon_onix_stream"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_codec.py -v
```

Expected: FAIL with `ImportError: cannot import name 'OnixStreamCodec'`.

- [ ] **Step 3: Implement `OnixStreamCodec`**

Append to `aeon/dj_pipeline/utils/codec.py`:

```python
class OnixStreamCodec(dj.Codec):
    """Structural-only codec for ONIX-clocked stream groups (e.g., Bno055).

    Stores a self-contained JSON reference. On fetch, loads the referenced
    binaries via the dotmap reader hierarchy in ``aeon/schema/ephys.py``,
    prefix-renames stream columns, concats on the shared sample index, and
    returns an **ONIX-clock-indexed** DataFrame.

    The codec deliberately does NOT apply the HARP sync regression — that's
    the caller's responsibility (typically via ``OnixImuChunk.synced_df``).

    Stored JSON format::

        {
            "experiment_name": "...",
            "epoch_start": "...",
            "sync_start": "...",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055"
        }
    """

    name = "aeon_onix_stream"

    _REQUIRED_KEYS = {
        "experiment_name",
        "epoch_start",
        "sync_start",
        "device_name",
        "stream_group",
    }

    def get_dtype(self, is_store: bool) -> str:
        return "json"

    def encode(self, value, *, key=None, store_name=None):
        if not isinstance(value, dict):
            raise TypeError(f"OnixStreamCodec expects a dict, got {type(value).__name__}")
        missing = self._REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(f"OnixStreamCodec missing required keys: {missing}")
        return value

    def decode(self, stored, *, key=None):
        """Load + merge the referenced ONIX-clocked stream group as an ONIX-indexed DataFrame."""
        # Lazy imports to avoid circular references at module load time.
        import pandas as pd

        from aeon.dj_pipeline import acquisition, ephys
        from aeon.dj_pipeline.utils.onix_imu import (
            IMU_COLUMNS,
            load_and_merge_bno055,
            locate_bno055_chunk_index,
        )

        sm_key = {
            "experiment_name": stored["experiment_name"],
            "epoch_start": pd.Timestamp(stored["epoch_start"]),
            "sync_start": pd.Timestamp(stored["sync_start"]),
        }
        sm = (ephys.EphysSyncModel & sm_key).fetch1()

        epoch_dir = (acquisition.Epoch & sm_key).fetch1("epoch_dir")
        raw_dir = acquisition.Experiment.get_data_directory(
            {"experiment_name": stored["experiment_name"]}, "raw"
        )
        device_dir = raw_dir / epoch_dir / stored["device_name"]

        chunk_index = locate_bno055_chunk_index(
            device_dir, stored["device_name"], int(sm["onix_ts_start"])
        )
        if chunk_index is None:
            # No-data row — return empty DataFrame with canonical columns and uint64 index.
            return pd.DataFrame(columns=list(IMU_COLUMNS), index=pd.Index([], dtype=np.uint64))

        if stored["stream_group"] != "Bno055":
            raise NotImplementedError(
                f"stream_group={stored['stream_group']!r} not supported. "
                "Only 'Bno055' is wired today."
            )
        return load_and_merge_bno055(device_dir, stored["device_name"], chunk_index)
```

Add `import numpy as np` near the top of the file (next to existing imports).

- [ ] **Step 4: Run encode tests**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_codec.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Register the codec in `__init__.py`**

Edit `aeon/dj_pipeline/__init__.py:11-12`:

```python
# Register codecs BEFORE any schema activation
from aeon.dj_pipeline.utils.codec import (  # pyright: ignore[reportUnusedImport]
    AeonStreamCodec,
    OnixStreamCodec,
)
```

- [ ] **Step 6: Commit**

```bash
git add aeon/dj_pipeline/utils/codec.py aeon/dj_pipeline/__init__.py tests/dj_pipeline/utils/test_onix_codec.py
git commit -m "feat(codec): add OnixStreamCodec for ONIX-clocked stream groups"
```

---

### Task 4: Add `EphysSyncModel` table definition

`dj.Manual` table keyed by `(EphysEpoch, sync_start)`. Stores observed HARP and ONIX bounds plus the joblib regression model as `attach`. No ingest logic yet — that's Task 5.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (insert before existing `EphysChunk` definition, around line 211)
- Create: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
"""Integration tests for the ONIX IMU pipeline (EphysSyncModel, refactored EphysChunk, OnixImuChunk)."""

import datajoint as dj
import pytest


pytestmark = pytest.mark.integration


def test_ephys_sync_model_table_exists(dj_config_integration):
    """EphysSyncModel can be activated against an integration DB."""
    from aeon.dj_pipeline import ephys

    assert hasattr(ephys, "EphysSyncModel")
    table = ephys.EphysSyncModel()
    assert "sync_start" in table.primary_key
    assert "experiment_name" in table.primary_key
    assert "epoch_start" in table.primary_key
    # Attribute presence
    attrs = set(table.heading.attributes.keys())
    assert {"sync_end", "onix_ts_start", "onix_ts_end", "sync_model", "r2", "n_samples"} <= attrs
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_sync_model_table_exists -v
```

Expected: FAIL with `AttributeError: module 'aeon.dj_pipeline.ephys' has no attribute 'EphysSyncModel'`.

- [ ] **Step 3: Add the table definition**

Edit `aeon/dj_pipeline/ephys.py`. Locate the `@schema class EphysChunk(dj.Manual):` line (around line 212), and insert this block immediately before it:

```python
@schema
class EphysSyncModel(dj.Manual):
    """Per-chunk HARP↔ONIX sync regression for an ephys epoch.

    One row per ``HarpSync_*.csv`` (one per ONIX chunk window). Both HARP and
    ONIX bounds are observed values from the CSV (not predicted) — stable
    across re-ingestion.
    """

    definition = """
    -> EphysEpoch
    sync_start: datetime(6)            # PK — observed harp_time[0] from CSV (master clock)
    ---
    sync_end: datetime(6)              # observed harp_time[-1] from CSV
    onix_ts_start: bigint              # observed clock[0] from CSV
    onix_ts_end: bigint                # observed clock[-1] from CSV
    sync_model: attach                 # joblib-serialized LinearRegression (onix→harp)
    r2: float                          # regression fit quality
    n_samples: int                     # rows in CSV after dropna()
    unique index (experiment_name, epoch_start, onix_ts_start)
    """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_sync_model_table_exists -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "feat(ephys): add EphysSyncModel table"
```

---

### Task 5: Implement `EphysSyncModel.ingest(experiment_name)` classmethod

Cron-driven discovery: walks `*_HarpSync_*.csv` across the experiment's raw dir, reads each via the extended `HarpSyncModel.Reader`, joblib-dumps the model, and inserts a row. Idempotent — skips already-inserted CSVs by `(experiment_name, epoch_start, sync_start)`.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (add classmethod inside `EphysSyncModel`)
- Modify: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
import csv
from pathlib import Path

import numpy as np


def _make_synthetic_epoch(raw_dir: Path, experiment_name: str, epoch_dir_name: str,
                          device_name: str, n_chunks: int):
    """Create an epoch directory with `n_chunks` HarpSync CSVs and matching Clock binaries."""
    epoch_dir = raw_dir / epoch_dir_name
    device_dir = epoch_dir / device_name
    device_dir.mkdir(parents=True)

    for n in range(n_chunks):
        # HarpSync CSV: one row per second, ONIX clock advancing at 1000 ticks/sec
        ts_str = f"2024-06-04T1{n}-00-00"
        csv_path = device_dir / f"{device_name}_HarpSync_{ts_str}.csv"
        rows = []
        for s in range(60):
            rows.append({
                "clock": 1000 * (n * 60 + s) + 1,
                "hub_clock": s,
                "harp_time": float(3000 + n * 60 + s),
            })
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clock", "hub_clock", "harp_time"])
            writer.writeheader()
            writer.writerows(rows)


def test_ephys_sync_model_ingest_inserts_one_row_per_csv(
    dj_config_integration, tmp_path, monkeypatch
):
    """EphysSyncModel.ingest walks HarpSync CSVs and inserts one row each."""
    from aeon.dj_pipeline import acquisition, ephys

    # Set up a synthetic raw_dir + register experiment + epoch
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    experiment_name = "test_onix_exp"
    epoch_dir_name = "2024-06-04T10-24-07"
    device_name = "NeuropixelsV2Beta"

    _make_synthetic_epoch(raw_dir, experiment_name, epoch_dir_name, device_name, n_chunks=3)

    # Insert minimal Experiment + Epoch + EphysEpoch fixtures
    # (helper TBD per existing test patterns; reuse fixtures from conftest.py)
    # ... pseudocode:
    # acquisition.Experiment.insert1({...})
    # acquisition.Experiment.Directory.insert1({...raw_dir})
    # acquisition.Epoch.insert1({"experiment_name": experiment_name,
    #                            "epoch_start": ..., "epoch_dir": epoch_dir_name, ...})
    # ephys.EphysEpoch.insert1({..., "has_ephys": True, "n_probes": 0})

    ephys.EphysSyncModel.ingest(experiment_name)

    rows = (ephys.EphysSyncModel & {"experiment_name": experiment_name}).to_dicts()
    assert len(rows) == 3
    for row in rows:
        assert row["n_samples"] == 60
        assert 0.0 <= row["r2"] <= 1.0


def test_ephys_sync_model_ingest_is_idempotent(
    dj_config_integration, tmp_path, monkeypatch
):
    """Re-running ingest doesn't insert duplicates."""
    # (Set up same as above)
    # ...
    ephys.EphysSyncModel.ingest(experiment_name)
    initial_count = len(ephys.EphysSyncModel & {"experiment_name": experiment_name})
    ephys.EphysSyncModel.ingest(experiment_name)
    assert len(ephys.EphysSyncModel & {"experiment_name": experiment_name}) == initial_count
```

> NOTE for the implementer: the fixture setup (`acquisition.Experiment` etc.) follows the patterns in `tests/dj_pipeline/test_full_ingestion.py` and `tests/dj_pipeline/utils/conftest.py`. Reuse the `dj_config_integration` fixture and any existing `experiment_factory`/`epoch_factory` helpers. If those helpers don't exist, write them in `tests/dj_pipeline/conftest.py` as a side helper (not part of this task's commit; flag for follow-up if needed).

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_sync_model_ingest_inserts_one_row_per_csv -v
```

Expected: FAIL with `AttributeError: ... EphysSyncModel has no attribute 'ingest'`.

- [ ] **Step 3: Implement `EphysSyncModel.ingest`**

Add to `EphysSyncModel` (the class added in Task 4). Insert after the `definition` string (still inside the class):

```python
@classmethod
def ingest(cls, experiment_name: str) -> None:
    """Discover new HarpSync CSVs across all epochs of the experiment and insert sync model rows.

    Idempotent: skips CSVs whose ``(experiment_name, epoch_start, sync_start)``
    is already present.
    """
    import re
    import tempfile
    from pathlib import Path

    import joblib
    from swc.aeon.io import api as io_api

    from aeon.schema.ephys import social_ephys

    exp_key = {"experiment_name": experiment_name}
    raw_dir = acquisition.Experiment.get_data_directory(exp_key, directory_type="raw")
    if raw_dir is None:
        logger.error(f"Raw data directory not found for {experiment_name}")
        return

    # Build epoch_dir → epoch_start lookup
    epochs = (acquisition.Epoch & exp_key).proj("epoch_start", "epoch_dir").to_dicts()
    epoch_dir_to_start = {}
    for ep in epochs:
        if ep["epoch_dir"]:
            top_dir = Path(ep["epoch_dir"]).parts[0]
            epoch_dir_to_start[top_dir] = ep["epoch_start"]

    csvs = sorted(raw_dir.rglob("*_HarpSync_*.csv"))
    for csv_path in csvs:
        rel_parts = csv_path.relative_to(raw_dir).parts
        if len(rel_parts) < 3:
            continue
        epoch_dir_name = rel_parts[0]
        device_name = rel_parts[1]

        epoch_start = epoch_dir_to_start.get(epoch_dir_name)
        if epoch_start is None:
            continue

        device_reader = social_ephys.get(device_name)
        if device_reader is None or "HarpSyncModel" not in device_reader:
            continue

        df_row = device_reader["HarpSyncModel"].read(csv_path).iloc[0]
        sync_start = io_api.to_datetime(float(df_row["harp_start"]))
        # Strip tz if present (DJ datetime(6) is naive)
        if hasattr(sync_start, "tz_localize"):
            sync_start = sync_start.tz_localize(None)

        existing = cls & {"experiment_name": experiment_name,
                          "epoch_start": epoch_start, "sync_start": sync_start}
        if existing:
            continue

        sync_end = io_api.to_datetime(float(df_row["harp_end"]))
        if hasattr(sync_end, "tz_localize"):
            sync_end = sync_end.tz_localize(None)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"{csv_path.stem}.joblib"
            joblib.dump(df_row["model"], model_path)
            cls.insert1({
                "experiment_name": experiment_name,
                "epoch_start": epoch_start,
                "sync_start": sync_start,
                "sync_end": sync_end,
                "onix_ts_start": int(df_row["clock_start"]),
                "onix_ts_end": int(df_row["clock_end"]),
                "sync_model": str(model_path),
                "r2": float(df_row["r2"]),
                "n_samples": int(df_row["n_samples"]),
            })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v -k "ingest"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "feat(ephys): implement EphysSyncModel.ingest classmethod"
```

---

### Task 6: Refactor `EphysChunk.SyncModel` Part to a link-only table

Old definition has `onix_ts_start`/`onix_ts_end`/`sync_model: attach`/`harp_start`. New definition is FK-only: `-> master` and `-> EphysSyncModel`. Schema-breaking change — existing data must be dropped before applying.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py:212-239` (the `EphysChunk` class block)
- Modify: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
def test_ephys_chunk_sync_model_part_is_link_only(dj_config_integration):
    """EphysChunk.SyncModel exposes only FK columns, no attach or onix bounds."""
    from aeon.dj_pipeline import ephys

    sync_part = ephys.EphysChunk.SyncModel()
    attrs = set(sync_part.heading.attributes.keys())

    # FK to master + EphysSyncModel
    expected = {"experiment_name", "subject", "insertion_number", "chunk_start", "epoch_start", "sync_start"}
    assert expected <= attrs

    # No more attach / onix bounds / harp_start
    assert "sync_model" not in attrs
    assert "onix_ts_start" not in attrs
    assert "onix_ts_end" not in attrs
    assert "harp_start" not in attrs
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_chunk_sync_model_part_is_link_only -v
```

Expected: FAIL — current Part has `onix_ts_start` etc.

- [ ] **Step 3: Modify `EphysChunk.SyncModel` Part**

Edit `aeon/dj_pipeline/ephys.py:231-239`. Replace:

```python
    class SyncModel(dj.Part):
        definition = """
        -> master
        onix_ts_start: bigint  # ONIX timestamp at the start of the sync
        ---
        onix_ts_end: bigint  # ONIX timestamp at the end of the sync
        sync_model: attach  # serialized file containing the sync model
        harp_start: datetime(6)  # HARP start time of the sync
        """
```

with:

```python
    class SyncModel(dj.Part):
        """Link-only: each EphysChunk references 1+ EphysSyncModel rows.

        The actual model bytes and ONIX bounds live on EphysSyncModel.
        """

        definition = """
        -> master
        -> EphysSyncModel
        """
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_chunk_sync_model_part_is_link_only -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "refactor(ephys): convert EphysChunk.SyncModel Part to link-only FK"
```

---

### Task 7: Refactor `EphysChunk.ingest_chunks` to use DB-backed sync lookup

Replace the in-memory `sync_models = {}` cache + lazy `io_api.load(HarpSyncModel)` with a DB query against `EphysSyncModel`. Compute `chunk_start`/`chunk_end` via the fast path (`onix_ts == matched.onix_ts_start` → use `matched.sync_start` directly) or slow path (download attach + predict). Insert link Part rows for the 1-or-2 matched SyncModels.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (the `ingest_chunks` classmethod, around line 241)
- Modify: `aeon/dj_pipeline/utils/ephys_utils.py` (slim `process_ephys_file`)

- [ ] **Step 1: Write the failing test**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
def test_ephys_chunk_ingest_uses_sync_model_from_db(dj_config_integration, tmp_path):
    """EphysChunk.ingest_chunks reads sync info from EphysSyncModel rows, not in-memory cache."""
    # 1. Set up synthetic epoch with HarpSync CSVs + AmplifierData binaries
    # 2. EphysSyncModel.ingest(...)
    # 3. EphysChunk.ingest_chunks(...)
    # 4. Verify EphysChunk.SyncModel link rows reference EphysSyncModel rows by sync_start
    # (Detailed setup pattern: see Task 5's test for fixture sketch)

    from aeon.dj_pipeline import ephys

    # ... setup ...
    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.EphysChunk.ingest_chunks(experiment_name)

    chunk_rows = (ephys.EphysChunk & {"experiment_name": experiment_name}).to_dicts()
    assert len(chunk_rows) >= 1

    link_rows = (ephys.EphysChunk.SyncModel & {"experiment_name": experiment_name}).to_dicts()
    sync_starts_in_links = {r["sync_start"] for r in link_rows}
    sync_starts_in_table = {
        r["sync_start"]
        for r in (ephys.EphysSyncModel & {"experiment_name": experiment_name}).to_dicts()
    }
    assert sync_starts_in_links <= sync_starts_in_table
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_ephys_chunk_ingest_uses_sync_model_from_db -v
```

Expected: FAIL — current `ingest_chunks` still tries to insert old-shape SyncModel rows.

- [ ] **Step 3: Refactor `ingest_chunks`**

Replace the body of `EphysChunk.ingest_chunks` (`aeon/dj_pipeline/ephys.py:241-372`). Key changes:

```python
@classmethod
def ingest_chunks(cls, experiment_name: str) -> None:
    """Discover new AmplifierData files and create EphysChunk rows.

    For each new file, looks up the matching ``EphysSyncModel`` rows in DB
    (no in-memory cache, no model fitting — that's now upstream), computes
    HARP ``chunk_start``/``chunk_end`` (fast path uses the SyncModel's
    observed ``sync_start``/``sync_end`` directly; slow path downloads the
    attach and predicts), and inserts master + File + SyncModel link rows.
    """
    import re
    import joblib
    import numpy as np
    from pathlib import Path
    from swc.aeon.io import api as io_api

    exp_key = {"experiment_name": experiment_name}
    raw_dir = acquisition.Experiment.get_data_directory(exp_key, directory_type="raw")
    if raw_dir is None:
        logger.error(f"Raw data directory not found for {experiment_name}")
        return

    epochs = (acquisition.Epoch & exp_key).proj("epoch_start", "epoch_dir").to_dicts()
    epoch_dir_to_start = {}
    for ep in epochs:
        if ep["epoch_dir"]:
            top_dir = Path(ep["epoch_dir"]).parts[0]
            epoch_dir_to_start[top_dir] = ep["epoch_start"]

    insertion_lookup = {}
    for entry in (EphysEpoch.Insertion & exp_key).to_dicts():
        insertion_lookup[(entry["epoch_start"], entry["probe_label"])] = {
            "experiment_name": entry["experiment_name"],
            "subject": entry["subject"],
            "insertion_number": entry["insertion_number"],
        }
    if not insertion_lookup:
        logger.warning(f"No EphysEpoch.Insertion entries for {experiment_name}; run EphysEpoch.populate first.")
        return

    all_files = sorted(raw_dir.rglob("*_AmplifierData*.bin"), key=lambda p: p.as_posix())
    for ephys_file in all_files:
        rel_path = ephys_file.relative_to(raw_dir).as_posix()
        if cls.File & exp_key & {"file_path": rel_path}:
            continue

        rel_parts = ephys_file.relative_to(raw_dir).parts
        if len(rel_parts) < 3:
            continue
        epoch_dir_name = rel_parts[0]
        epoch_start = epoch_dir_to_start.get(epoch_dir_name)
        if epoch_start is None:
            continue

        name_match = re.search(r"_(Probe[A-Z])_AmplifierData", ephys_file.name)
        if not name_match:
            continue
        probe_label = name_match.group(1)

        insertion_key = insertion_lookup.get((epoch_start, probe_label))
        if insertion_key is None:
            logger.warning(f"No subject-probe mapping for {probe_label} in {epoch_start}; skipping {rel_path}")
            continue

        clock_file = ephys_file.with_name(ephys_file.name.replace("AmplifierData", "Clock"))
        onix_ts = np.memmap(clock_file, mode="r", dtype=np.uint64)
        first_ts, last_ts = int(onix_ts[0]), int(onix_ts[-1])

        matched = (EphysSyncModel & {"experiment_name": experiment_name, "epoch_start": epoch_start}
                   & f"({first_ts} BETWEEN onix_ts_start AND onix_ts_end) "
                     f"OR ({last_ts} BETWEEN onix_ts_start AND onix_ts_end)").fetch(order_by="sync_start", as_dict=True)
        if not matched:
            logger.warning(f"No EphysSyncModel covers {ephys_file.name}; skipping")
            continue

        chunk_start = _resolve_harp(matched[0], first_ts)
        chunk_end = _resolve_harp(matched[-1], last_ts)

        # Resolve electrode_config (existing pattern)
        probe_name = (ProbeInsertion & insertion_key).fetch1("probe")
        probe_type = (Probe & {"probe": probe_name}).fetch1("probe_type")
        configs = (ElectrodeConfig & {"probe_type": probe_type}).to_arrays("electrode_config_name")
        if len(configs) == 0:
            raise ValueError(f"No electrode configs for probe_type={probe_type}")
        if len(configs) > 1:
            raise ValueError(f"Multiple electrode configs for {probe_type}: {configs}")
        electrode_config_name = configs[0]

        chunk_entry = {
            **insertion_key,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "epoch_start": epoch_start,
            "electrode_config_name": electrode_config_name,
        }
        cls.insert1(chunk_entry)
        cls.File.insert([
            {**chunk_entry, "directory_type": "raw",
             "file_name": f.name, "file_path": f.relative_to(raw_dir).as_posix()}
            for f in (ephys_file, clock_file)
        ], ignore_extra_fields=True)
        cls.SyncModel.insert([
            {**chunk_entry, "sync_start": m["sync_start"]}
            for m in matched
        ], ignore_extra_fields=True)


def _resolve_harp(sync_row: dict, onix_ts: int):
    """Compute the HARP equivalent of ``onix_ts`` from a SyncModel row.

    Fast path: if onix_ts matches the row's bound exactly, return the observed harp value.
    Slow path: download the attach, joblib.load, predict.
    """
    import joblib
    import numpy as np
    from swc.aeon.io import api as io_api

    if onix_ts == int(sync_row["onix_ts_start"]):
        return sync_row["sync_start"]
    if onix_ts == int(sync_row["onix_ts_end"]):
        return sync_row["sync_end"]
    model = joblib.load(sync_row["sync_model"])
    harp_seconds = float(model.predict(np.array([[onix_ts]])).flatten()[0])
    harp_dt = io_api.to_datetime(harp_seconds)
    if hasattr(harp_dt, "tz_localize"):
        harp_dt = harp_dt.tz_localize(None)
    return harp_dt
```

Add the `_resolve_harp` helper as a module-level function in `aeon/dj_pipeline/ephys.py` (next to the other module-level helpers).

- [ ] **Step 4: Trim `process_ephys_file` (or remove)**

Edit `aeon/dj_pipeline/utils/ephys_utils.py:212-321`. The function is now obsolete — its model-fitting half moved into `EphysSyncModel.ingest`, its insert half into the refactored `ingest_chunks`. Delete the function entirely. Also remove its import from `aeon/dj_pipeline/ephys.py` (line 15).

- [ ] **Step 5: Run tests to verify pass**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v
```

Expected: all task-1-through-7 tests pass.

- [ ] **Step 6: Commit**

```bash
git add aeon/dj_pipeline/ephys.py aeon/dj_pipeline/utils/ephys_utils.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "refactor(ephys): drive EphysChunk.ingest_chunks from EphysSyncModel rows"
```

---

### Task 8: Add `OnixImuChunk` table definition

Single-row table per sync window with 13 per-column JSON summary attributes plus the `<aeon_onix_stream>` codec column. Default `key_source` = `EphysSyncModel`.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (add at end of file, after `EphysBlockInfo`)
- Modify: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
def test_onix_imu_chunk_table_shape(dj_config_integration):
    """OnixImuChunk has 13 IMU column attrs + sample_count + timestamps + stream_df."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    table = ephys.OnixImuChunk()
    attrs = set(table.heading.attributes.keys())

    assert {"experiment_name", "epoch_start", "sync_start"} <= set(table.primary_key)
    assert {"sample_count", "timestamps", "stream_df"} <= attrs
    for col in IMU_COLUMNS:
        assert col in attrs, f"Missing per-column stat field: {col}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_onix_imu_chunk_table_shape -v
```

Expected: FAIL with `AttributeError: ... no attribute 'OnixImuChunk'`.

- [ ] **Step 3: Add the table definition**

Append to `aeon/dj_pipeline/ephys.py`:

```python
@schema
class OnixImuChunk(dj.Imported):
    """One row per ONIX sync window with all four Bno055 streams merged on sample index.

    The codec column ``stream_df`` returns an ONIX-clock-indexed DataFrame.
    For HARP-indexed data, use :meth:`OnixImuChunk.synced_df`.
    """

    definition = """
    -> EphysSyncModel
    ---
    sample_count: int32
    timestamps: json
    euler_x: json
    euler_y: json
    euler_z: json
    gravity_vector_x: json
    gravity_vector_y: json
    gravity_vector_z: json
    linear_acceleration_x: json
    linear_acceleration_y: json
    linear_acceleration_z: json
    quaternion_w: json
    quaternion_x: json
    quaternion_y: json
    quaternion_z: json
    stream_df: <aeon_onix_stream>
    """
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py::test_onix_imu_chunk_table_shape -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "feat(ephys): add OnixImuChunk table definition"
```

---

### Task 9: Implement `OnixImuChunk.make`

Loads + merges Bno055 streams via `load_and_merge_bno055`, applies regression for HARP timestamp summary only, computes per-column stats, inserts a single row. No-IMU rigs get an empty row (`sample_count=0`, empty stat dicts).

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (add `make` to `OnixImuChunk`)
- Modify: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
def test_onix_imu_chunk_populate_with_data(dj_config_integration, tmp_path):
    """OnixImuChunk.populate() ingests Bno055 streams when files exist."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    # Set up: synthetic epoch with HarpSync + Bno055_Clock + 4 stream binaries
    # ... (use _make_synthetic_epoch + extension that adds Bno055 binaries)
    ephys.EphysSyncModel.ingest(experiment_name)

    ephys.OnixImuChunk.populate()

    rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).to_dicts()
    assert len(rows) >= 1
    for r in rows:
        assert r["sample_count"] > 0
        assert isinstance(r["timestamps"], dict)
        assert "min" in r["timestamps"] or "start" in r["timestamps"]
        for col in IMU_COLUMNS:
            assert isinstance(r[col], dict)


def test_onix_imu_chunk_populate_no_imu_rig(dj_config_integration, tmp_path):
    """When Bno055 files are absent, OnixImuChunk inserts an empty row."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    # Set up: synthetic epoch with HarpSync but NO Bno055 files
    # ...
    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    rows = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).to_dicts()
    assert len(rows) >= 1
    for r in rows:
        assert r["sample_count"] == 0
        assert r["timestamps"] == {}
        for col in IMU_COLUMNS:
            assert r[col] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v -k "populate"
```

Expected: FAIL — `make` not implemented.

- [ ] **Step 3: Implement `make`**

Inside the `OnixImuChunk` class (added in Task 8), add:

```python
def make(self, key):
    """Populate one OnixImuChunk row per EphysSyncModel key.

    No-IMU rigs (Bno055 files absent) get a row with ``sample_count=0`` and empty stat dicts.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from swc.aeon.io import api as io_api

    from aeon.dj_pipeline.utils.onix_imu import (
        IMU_COLUMNS,
        load_and_merge_bno055,
        locate_bno055_chunk_index,
    )
    from aeon.dj_pipeline.utils.stats import column_stats, timestamp_stats

    sm = (EphysSyncModel & key).fetch1()
    epoch_dir = (acquisition.Epoch & key).fetch1("epoch_dir")
    raw_dir = acquisition.Experiment.get_data_directory(
        {"experiment_name": key["experiment_name"]}, "raw",
    )
    epoch_path = raw_dir / epoch_dir

    device_name = None
    for candidate in ("NeuropixelsV2Beta", "NeuropixelsV2"):
        if (epoch_path / candidate).is_dir():
            device_name = candidate
            break
    if device_name is None:
        # Should not happen if EphysSyncModel exists, but defensively:
        self._insert_empty_row(key, IMU_COLUMNS, device_name="UNKNOWN")
        return

    device_dir = epoch_path / device_name
    chunk_index = locate_bno055_chunk_index(device_dir, device_name, int(sm["onix_ts_start"]))

    stream_df_ref = {
        "experiment_name": key["experiment_name"],
        "epoch_start": str(key["epoch_start"]),
        "sync_start": str(key["sync_start"]),
        "device_name": device_name,
        "stream_group": "Bno055",
    }

    if chunk_index is None:
        self.insert1({
            **key,
            "sample_count": 0,
            "timestamps": {},
            **{col: {} for col in IMU_COLUMNS},
            "stream_df": stream_df_ref,
        })
        return

    df = load_and_merge_bno055(device_dir, device_name, chunk_index)

    # Apply regression for HARP timestamp summary only (codec stays ONIX-indexed)
    model = joblib.load(sm["sync_model"])
    harp_seconds = model.predict(df.index.values.reshape(-1, 1)).flatten()
    harp_index = pd.to_datetime(io_api.to_datetime(harp_seconds))

    self.insert1({
        **key,
        "sample_count": len(df),
        "timestamps": timestamp_stats(harp_index),
        **{col: column_stats(df[col].values) for col in IMU_COLUMNS},
        "stream_df": stream_df_ref,
    })

def _insert_empty_row(self, key, IMU_COLUMNS, device_name):
    self.insert1({
        **key,
        "sample_count": 0,
        "timestamps": {},
        **{col: {} for col in IMU_COLUMNS},
        "stream_df": {
            "experiment_name": key["experiment_name"],
            "epoch_start": str(key["epoch_start"]),
            "sync_start": str(key["sync_start"]),
            "device_name": device_name,
            "stream_group": "Bno055",
        },
    })
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v -k "populate"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "feat(ephys): implement OnixImuChunk.make"
```

---

### Task 10: Add `OnixImuChunk.synced_df` classmethod

Explicit opt-in helper for HARP-indexed data. Fetches `stream_df` (ONIX-indexed via codec), downloads sync model attach, applies regression, returns HARP-indexed DataFrame.

**Files:**
- Modify: `aeon/dj_pipeline/ephys.py` (add classmethod to `OnixImuChunk`)
- Modify: `tests/dj_pipeline/test_onix_imu_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dj_pipeline/test_onix_imu_pipeline.py`:

```python
def test_synced_df_returns_harp_indexed_dataframe(dj_config_integration, tmp_path):
    """synced_df fetches stream_df and applies the SyncModel regression."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    # Set up: synthetic epoch with Bno055 binaries; ingest sync model + IMU chunk
    # ...
    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate()

    key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
    df = ephys.OnixImuChunk.synced_df(key)

    assert tuple(df.columns) == IMU_COLUMNS
    assert len(df) > 0
    # HARP-indexed → datetime dtype, not uint64
    assert df.index.dtype.kind == "M"


def test_synced_df_raises_on_ambiguous_key(dj_config_integration):
    """A non-PK restriction that matches multiple rows raises (via fetch1)."""
    import datajoint as dj
    from aeon.dj_pipeline import ephys

    # Assume multiple OnixImuChunk rows exist for the experiment
    with pytest.raises(dj.errors.DataJointError):
        ephys.OnixImuChunk.synced_df({"experiment_name": "test_onix_exp"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v -k "synced_df"
```

Expected: FAIL — `AttributeError: ... no attribute 'synced_df'`.

- [ ] **Step 3: Implement `synced_df`**

Add inside the `OnixImuChunk` class:

```python
@classmethod
def synced_df(cls, key):
    """Fetch stream_df for a single chunk and apply its HARP sync regression.

    Args:
        key: A complete OnixImuChunk primary key dict — must resolve to
            exactly one row. PK fields: ``experiment_name``, ``epoch_start``,
            ``sync_start``. (``fetch1()`` raises if the key resolves to zero
            or multiple rows.)

    Returns:
        HARP-time-indexed DataFrame with the columns in
        :data:`aeon.dj_pipeline.utils.onix_imu.IMU_COLUMNS`.

    For raw ONIX-clock-indexed data, fetch ``stream_df`` directly instead
    of using this helper.
    """
    import joblib
    import pandas as pd
    from swc.aeon.io import api as io_api

    df = (cls & key).fetch1("stream_df")
    sync_attach = (EphysSyncModel & key).fetch1("sync_model")
    model = joblib.load(sync_attach)
    harp_seconds = model.predict(df.index.values.reshape(-1, 1)).flatten()
    df.index = pd.to_datetime(io_api.to_datetime(harp_seconds))
    return df
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py -v -k "synced_df"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add aeon/dj_pipeline/ephys.py tests/dj_pipeline/test_onix_imu_pipeline.py
git commit -m "feat(ephys): add OnixImuChunk.synced_df helper for HARP-indexed fetch"
```

---

### Task 11: Codec round-trip integration test (encode → DB insert → fetch decode)

Verify end-to-end that an `OnixImuChunk` row's `stream_df` survives a DB round-trip and the codec returns an ONIX-indexed DataFrame matching the original.

**Files:**
- Modify: `tests/dj_pipeline/utils/test_onix_codec.py`

- [ ] **Step 1: Write the test**

Append to `tests/dj_pipeline/utils/test_onix_codec.py`:

```python
class TestOnixStreamCodecDecode:
    def test_round_trip_returns_onix_indexed_dataframe(self, dj_config_integration, tmp_path):
        """Insert a row with stream_df ref, fetch it back, codec returns merged ONIX-indexed DF."""
        import numpy as np
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        # Setup: synthetic epoch w/ Bno055 binaries, ingest sync model, populate IMU chunk
        # (reuse helpers from test_onix_imu_pipeline.py)
        # ...

        key = (ephys.OnixImuChunk & {"experiment_name": "test_onix_exp"}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        # Codec returns ONIX-indexed (uint64), NOT HARP datetimes
        assert df.index.dtype == np.uint64
        assert len(df) > 0

    def test_decode_no_data_returns_empty_dataframe(self, dj_config_integration, tmp_path):
        """For no-IMU rigs, decode returns an empty 13-column DataFrame."""
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        # Setup: epoch w/o Bno055 binaries
        # ...
        key = (ephys.OnixImuChunk & {"experiment_name": "no_imu_exp"}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        assert len(df) == 0
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/dj_pipeline/utils/test_onix_codec.py -v
```

Expected: 6 passed (4 encode tests from Task 3 + 2 decode tests).

- [ ] **Step 3: Commit**

```bash
git add tests/dj_pipeline/utils/test_onix_codec.py
git commit -m "test(codec): add OnixStreamCodec round-trip decode tests"
```

---

### Task 12: Migration note for existing `EphysChunk.SyncModel` rows

Old `EphysChunk.SyncModel` Part stored `attach`/`onix_ts_start`/`onix_ts_end`/`harp_start`. The new schema is link-only. **Schema-breaking change** — DataJoint cannot migrate the Part definition in-place.

**Action:** for development DBs, drop the old `EphysChunk` rows and re-ingest. Production migration is out of scope (separate ticket).

- [ ] **Step 1: Document the migration in the PR description**

Add to the PR description (when raising the PR for this branch):

```markdown
### Breaking schema change

`EphysChunk.SyncModel` Part definition changes from `attach`+ONIX-bounds to FK-only.
DataJoint cannot migrate this in place. Operators must:

1. Pause any running ingestion processes.
2. Drop `EphysChunk` and all dependent tables (`EphysBlock`, `EphysBlockInfo`, `OnixImuChunk`).
3. Drop `EphysSyncModel` if it exists from a prior partial deploy.
4. Pull this branch and let `aeon_ingest` re-ingest from raw files.

The model bytes for existing chunks are reproducible from the HarpSync CSVs, so no data is lost — just CPU time to re-fit regressions.
```

- [ ] **Step 2: Add a CLI helper for the drop step**

Create `aeon/dj_pipeline/scripts/drop_ephys_chunk_for_migration.py`:

```python
"""One-shot helper: drop EphysChunk and dependents to enable migration to link-only SyncModel.

Run once per database before deploying the ONIX IMU pipeline branch.
"""

import argparse

from aeon.dj_pipeline import ephys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="Confirm the drop (otherwise dry-run).")
    args = parser.parse_args()

    targets = [
        ephys.EphysBlockInfo,
        ephys.EphysBlock,
        ephys.EphysChunk,
    ]
    if hasattr(ephys, "EphysSyncModel"):
        targets.insert(0, ephys.EphysSyncModel)

    for table in targets:
        n = len(table())
        print(f"Would drop {table.__name__}: {n} rows")
        if args.yes:
            table.drop_quick()
            print(f"  → dropped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add aeon/dj_pipeline/scripts/drop_ephys_chunk_for_migration.py
git commit -m "chore(scripts): add EphysChunk drop helper for ONIX IMU migration"
```

---

### Task 13: Final verification — full test suite + lint

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass. Pre-existing failures unrelated to this work should be triaged but not block merge.

- [ ] **Step 2: Run pre-commit hooks**

```bash
uv run pre-commit run --all-files 2>&1 | tail -20
```

Expected: clean (or fix any reported issues and re-commit).

- [ ] **Step 3: Diff review against the spec**

Open `docs/specs/SPEC_ONIX_IMU_PIPELINE.md` and `git log --oneline tn/env-stream-tables..HEAD` side-by-side. Confirm every "Concrete change set" row in the spec has a corresponding commit. Any gaps → follow-up task.

- [ ] **Step 4: Push**

```bash
git push
```

- [ ] **Step 5: Open the PR**

```bash
gh pr create --base tn/env-stream-tables --title "ONIX IMU pipeline + ephys SyncModel refactor" --body "$(cat <<'EOF'
## Summary

Promotes per-chunk HARP↔ONIX sync regression to a first-class table (`EphysSyncModel`) and adds `OnixImuChunk` for Bno055 IMU streams via a structural-only `<aeon_onix_stream>` codec.

See [SPEC_ONIX_IMU_PIPELINE.md](docs/specs/SPEC_ONIX_IMU_PIPELINE.md) for full design rationale.

## Breaking schema change

`EphysChunk.SyncModel` Part is now FK-only. Operators must drop `EphysChunk` and dependents before deploying — see PR comment for runbook.

## Test plan

- [ ] Unit tests pass: `uv run pytest tests/schema/ tests/dj_pipeline/utils/`
- [ ] Integration tests pass: `uv run pytest tests/dj_pipeline/test_onix_imu_pipeline.py`
- [ ] Full suite green: `uv run pytest tests/`
- [ ] Manual: ingest one synthetic epoch end-to-end and verify `OnixImuChunk.synced_df` returns HARP-indexed data
EOF
)"
```

---

## Self-Review Checklist (run before marking plan complete)

- [x] Spec coverage: every section of `SPEC_ONIX_IMU_PIPELINE.md` mapped to a task (reader extension → T1, helpers → T2, codec → T3+T11, EphysSyncModel → T4+T5, EphysChunk refactor → T6+T7, OnixImuChunk → T8+T9+T10, migration → T12, validation → T13).
- [x] No placeholders. Every step has runnable commands and concrete code.
- [x] Type consistency: `IMU_COLUMNS` referenced consistently as a tuple. `EphysSyncModel` PK = `(experiment_name, epoch_start, sync_start)` everywhere. `OnixImuChunk` inherits same PK shape (one row per SyncModel). `synced_df(cls, key)` signature consistent across spec + tests + impl.
- [x] Spec out-of-scope items honored: no Pydantic migration, no `EphysChunk.populate()` upgrade, no `read_probe_assignments` work, no `EphysEpoch` re-population fix.

---

## Notes for the implementer

1. **Test fixtures.** The integration tests reference `_make_synthetic_epoch` and a `dj_config_integration` fixture. The latter exists; the former needs to be written in `tests/dj_pipeline/conftest.py` if not already there. Pattern: build a `tmp_path/raw/<epoch_dir>/<device_name>/` tree, write CSVs and binaries, register an `acquisition.Experiment` + `Epoch` + `EphysEpoch.Insertion` row pointing at it. See `tests/dj_pipeline/test_full_ingestion.py` for the existing convention.

2. **`io_api.to_datetime`** lives in `swc.aeon.io.api` (already used in current `process_ephys_file`). Confirm it accepts both scalar floats and arrays — most tasks pass scalars, `synced_df` and `make` pass arrays.

3. **`column_stats` and `timestamp_stats`** are imported from `aeon.dj_pipeline.utils.stats`. Verify their output shape during Task 9 implementation; spec assumes JSON-serializable dicts.

4. **DataJoint `attach` field** stores model bytes via the project's blob store (see `repository_config` in `__init__.py`). For tests, ensure the testcontainer config has a writable attach store path.

5. **`pytestmark = pytest.mark.integration`** at the top of `test_onix_imu_pipeline.py` ensures the file is auto-marked per project convention. Unit-only files (codec encode tests, onix_imu helper tests) should not have this marker.

6. **`streams.py` test pollution.** Per project memory, never commit regenerated `streams.py` from test runs. Confirm working tree before each commit.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-onix-imu-pipeline.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration. Best for a 13-task plan where some tasks (like fixture setup in T5) may surface refactoring opportunities.

2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
