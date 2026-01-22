# Testing Specification for streams_maker & load_new_metadata

Testing strategy for the Pydantic-based metadata loading and DataJoint table generation pipeline.

---

## Architecture

**Three tiers:**

| Tier | Scope | Database | Data | CI |
|------|-------|----------|------|-----|
| Unit | Pure functions | None | Synthetic dicts | Yes |
| Integration | Pipeline tables, catalog population | MySQL (testcontainers) | Real Pydantic + Real `@data_reader` + Real Readers | Yes |
| Specialized | Full populate(), streams_maker.main() | MySQL (testcontainers) | Golden datasets | Manual |

**Key patterns:**
- Testcontainers (zero-config MySQL)
- Real Pydantic classes from `swc.aeon.schema` (has `_resolve_pattern_prefix()` for `@data_reader`)
- Real `@data_reader` decorator from `swc.aeon.schema` (no mocking needed)
- Real Reader classes from `swc-aeon` (no file I/O until `load()`)
- Graceful skipping when optional data unavailable

---

## Design Decisions

### Why testcontainers?
- Zero configuration - no "start MySQL first" instructions
- Isolated - each test session gets fresh database
- CI-friendly - works in GitHub Actions without setup
- Same code works locally and in CI

### Why real Pydantic classes + real @data_reader?

| Package | Stability | Test Dependency? | Usage |
|---------|-----------|------------------|-------|
| `swc-aeon` (aeon_api) | Stable | Yes (existing) | `@data_reader`, `BaseSchema`, Reader classes, Device classes |
| `swc-aeon-rigs` (aeon_swc_rigs) | Stable | **No** | Pydantic config classes (different from `swc.aeon.schema`) |
| `aeon_exp_foragingABC` | **Volatile** | **No** | Experiment-specific, changes frequently |

**Rationale:**
- Use `swc.aeon.schema` classes (from `swc-aeon`) which have `_resolve_pattern_prefix()` required by `@data_reader`
- Use real `@data_reader` decorator - no mocking needed, tests actual production patterns
- Reader classes (`Video`, `Harp`, `Csv`) are stable and don't do file I/O until `load()` is called
- Test fixtures mirror production code (e.g., `aeon_exp_foragingABC/rig.py`)

**Important:** Do NOT use `swc.aeon.rigs` classes for `@data_reader` tests - they lack `_resolve_pattern_prefix()`.

### Why session-scoped DB fixtures?
- Creating/dropping schemas is expensive
- Integration tests build on each other (insert StreamType → insert DeviceType.Stream)
- Matches production flow where tables persist across operations

---

## Directory Structure

```
tests/
├── conftest.py                           # Root: testcontainers, markers
├── dj_pipeline/
│   ├── conftest.py                       # DB fixtures, pipeline loading
│   ├── test_pipeline_instantiation.py   # Existing: schema instantiation
│   ├── test_acquisition.py              # Existing: epoch/chunk ingestion
│   ├── test_qc.py                        # Existing: QC tables
│   ├── test_tracking.py                  # Existing: tracking tables
│   └── utils/
│       ├── conftest.py                   # Fixtures for load_new_metadata tests
│       ├── test_load_new_metadata_unit.py    # Unit tests (no DB)
│       └── test_load_new_metadata_integration.py  # Integration tests (DB)
└── fixtures/
    └── metadata/                         # Sample Metadata.json files (optional)
```

---

## Test Categories

### 1. Unit Tests (`test_load_new_metadata_unit.py`)

Pure function tests - no database, no external packages required:

| Function | Test Cases |
|----------|------------|
| `to_pascal_case()` | `"video"` → `"Video"`, `"beam_break"` → `"BeamBreak"` |
| `_flatten_rig_devices()` | Flatten nested rig config to device dict |
| `_infer_device_type_from_rig()` | Infer device type from rig structure position |
| `_extract_device_mapper_from_rig()` | Extract device type mapper and serial numbers |
| `extract_active_regions()` | Extract ActiveRegion data from rig config |

**Example:**
```python
@pytest.mark.unit
class TestToPascalCase:
    def test_single_word(self):
        assert to_pascal_case("video") == "Video"

    def test_two_words(self):
        assert to_pascal_case("beam_break") == "BeamBreak"

    def test_three_words(self):
        assert to_pascal_case("weight_raw_data") == "WeightRawData"

    def test_already_pascal(self):
        assert to_pascal_case("Video") == "Video"


@pytest.mark.unit
class TestFlattenRigDevices:
    def test_extracts_cameras(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "CameraTop" in result
        assert result["CameraTop"]["serialNumber"] == "21053810"

    def test_extracts_feeders(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "Feeder1" in result
        assert result["Feeder1"]["portName"] == "COM3"
```

### 2. Integration Tests (`test_load_new_metadata_integration.py`)

Database operations using real Pydantic classes with real `@data_reader` decorator:

| Function | Test Cases |
|----------|------------|
| `extract_stream_types_from_device()` | Extract from Device class with real @data_reader |
| `get_device_info()` | Extract device/stream info from test Rig |
| `get_stream_entries()` | Generate StreamType entries from test Rig |
| `get_device_mapper_from_rig()` | Extract device type and serial number mappings |
| `insert_stream_types()` | Insert into StreamType, handle duplicates |
| `insert_device_types()` | Insert DeviceType, DeviceType.Stream, Device; FK handling |

**Example:**
```python
@pytest.mark.integration
class TestInsertStreamTypes:
    def test_inserts_stream_types(self, pipeline, test_rig):
        """Verify StreamType entries are inserted from Rig."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)
        initial_count = len(streams.StreamType())

        insert_stream_types(test_rig)

        assert len(streams.StreamType()) > initial_count

    def test_handles_duplicates(self, pipeline, test_rig):
        """Verify duplicate insertions are handled gracefully (skip_duplicates)."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)

        insert_stream_types(test_rig)
        count_after_first = len(streams.StreamType())

        # Second call should not raise or create duplicates
        insert_stream_types(test_rig)

        assert len(streams.StreamType()) == count_after_first


@pytest.mark.integration
class TestInsertDeviceTypesFKHandling:
    def test_fk_constraint_triggers_stream_type_insert(self, pipeline, test_rig, tmp_path):
        """Verify FK failure triggers insert_stream_types()."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)

        # Ensure StreamType is empty to force FK failure
        # (DeviceType.Stream references StreamType)

        # This should succeed by calling insert_stream_types() on FK failure
        metadata_filepath = tmp_path / "Metadata.json"
        insert_device_types(test_rig, metadata_filepath)

        # Verify both StreamType and DeviceType.Stream are populated
        assert len(streams.StreamType()) > 0
        assert len(streams.DeviceType.Stream()) > 0

    def test_non_fk_errors_are_reraised(self, pipeline, test_rig, tmp_path, monkeypatch):
        """Verify non-FK DataJointErrors are re-raised."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)

        def mock_insert(entries):
            raise dj.DataJointError("Connection refused")

        monkeypatch.setattr(streams.DeviceType.Stream, "insert", mock_insert)

        with pytest.raises(dj.DataJointError, match="Connection refused"):
            insert_device_types(test_rig, tmp_path / "Metadata.json")
```

---

## Key Fixtures

### Testcontainers MySQL

```python
# tests/conftest.py

import pytest
from testcontainers.mysql import MySqlContainer


@pytest.fixture(scope="session")
def mysql_container():
    """Auto-provision MySQL via testcontainers."""
    with MySqlContainer("mysql:8.0") as mysql:
        yield mysql


@pytest.fixture(scope="session")
def dj_config(mysql_container):
    """Configure DataJoint to use testcontainers MySQL."""
    import datajoint as dj

    dj.config["database.host"] = mysql_container.get_container_host_ip()
    dj.config["database.port"] = mysql_container.get_exposed_port(3306)
    dj.config["database.user"] = "root"
    dj.config["database.password"] = mysql_container.MYSQL_ROOT_PASSWORD
    dj.config["safemode"] = False
    dj.config["custom"] = {}
    dj.config["custom"]["database.prefix"] = "test_"

    yield dj.config

    # Teardown: drop test schemas
    # (implementation depends on pipeline structure)
```

### Test Rig with Real @data_reader

We use **real `@data_reader` decorator** and **real Reader classes** from `swc-aeon` - no mocking needed.

```python
# tests/dj_pipeline/utils/conftest.py

import pytest

# Real Reader classes from swc-aeon (no file I/O until load() is called)
from swc.aeon.io.reader import Csv, Harp, Video

# Real Pydantic base classes from swc-aeon (has _resolve_pattern_prefix)
from swc.aeon.schema import BaseSchema, data_reader
from swc.aeon.schema.foraging import UndergroundFeeder
from swc.aeon.schema.video import SpinnakerCamera


class TestCamera(SpinnakerCamera):
    """Test camera using real @data_reader decorator."""

    @data_reader
    def video(self, pattern) -> Video:
        return Video(f"{pattern}")

    @data_reader
    def position(self, pattern) -> Harp:
        return Harp(f"{pattern}_200", columns=["x", "y"])


class TestFeeder(UndergroundFeeder):
    """Test feeder using real @data_reader decorator."""

    @data_reader
    def beam_break(self, pattern) -> Harp:
        return Harp(f"{pattern}_32", columns=["state"])

    @data_reader
    def encoder(self, pattern) -> Csv:
        return Csv(f"{pattern}_90", columns=["angle", "intensity"])


class TestRig(BaseSchema):
    """Test Rig using real Pydantic base classes with real @data_reader."""

    cameras: dict[str, TestCamera] = {}
    feeders: dict[str, TestFeeder] = {}


@pytest.fixture
def test_rig():
    """Create a test Rig with real Pydantic structure and real @data_reader."""
    return TestRig(
        cameras={
            "CameraTop": TestCamera(serial_number="21053810"),
            "CameraSide": TestCamera(serial_number="21053811"),
        },
        feeders={
            "Feeder1": TestFeeder(port_name="COM3"),
            "Feeder2": TestFeeder(port_name="COM4"),
        },
    )


@pytest.fixture
def sample_rig_config():
    """Sample nested rig configuration dict for unit tests."""
    return {
        "cameras": {
            "CameraTop": {
                "serialNumber": "21053810",
                "trigger": "Trigger0",
                "cameraTracking": {"blobTracking": {"Arena": {"x": 0, "y": 0}}},
            },
            "CameraSide": {"serialNumber": "21053811", "trigger": "Trigger1"},
        },
        "feeders": {
            "Feeder1": {"portName": "COM3"},
            "Feeder2": {"portName": "COM4"},
        },
        "nest": {"Nest": {"portName": "COM5"}},
        "cameraSynchronizer": {"portName": "COM6"},
        "clockSynchronizer": {"portName": "COM7"},
    }
```

**Note:** Import from `swc.aeon.schema` (not `swc.aeon.rigs`) - these classes have `_resolve_pattern_prefix()` required by `@data_reader`.

### Auto-Marking Hook

```python
# tests/conftest.py

def pytest_collection_modifyitems(items):
    """Auto-mark tests that use integration fixtures."""
    for item in items:
        # If test uses mysql_container or pipeline fixture, mark as integration
        if "mysql_container" in item.fixturenames or "pipeline" in item.fixturenames:
            item.add_marker(pytest.mark.integration)
```

---

## Pytest Markers

Register in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests (no database required)",
    "integration: Integration tests (requires MySQL via testcontainers)",
    "slow: Tests that take >10 seconds",
]
testpaths = ["tests"]
```

---

## CI Strategy

### Trigger Matrix

| Event | Unit Tests | Integration Tests |
|-------|------------|-------------------|
| PR opened/updated | Yes | Yes |
| Merge to main | Yes | Yes |
| Scheduled (weekly) | Yes | Yes |

### GitHub Actions Workflow

```yaml
name: Tests

on:
  push:
    branches: [datajoint_pipeline, main]
  pull_request:
    branches: [datajoint_pipeline, main]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install testcontainers[mysql]

      - name: Run unit tests
        run: pytest -m unit -v --tb=short

      - name: Run integration tests
        run: pytest -m integration -v --tb=short
```

**Note:** Testcontainers automatically handles MySQL provisioning - no `services:` block needed.

---

## Commands

```bash
# Unit tests only (fast, no database)
pytest -m unit -v

# Integration tests only (auto-provisions MySQL)
pytest -m integration -v

# All tests
pytest -v

# Specific test file
pytest tests/dj_pipeline/utils/test_load_new_metadata_unit.py -v

# With coverage
pytest --cov=aeon.dj_pipeline.utils --cov-report=html

# Debug single test
pytest tests/dj_pipeline/utils/test_load_new_metadata_unit.py::TestToPascalCase -v --pdb
```

---

## Implementation Plan

### Phase 2.1: Setup & Unit Tests

1. Add `testcontainers[mysql]` to test dependencies (swc-aeon is already a main dependency)
2. Create `tests/dj_pipeline/utils/conftest.py` with fixtures
3. Create `tests/dj_pipeline/utils/test_load_new_metadata_unit.py`:
   - `TestToPascalCase`
   - `TestFlattenRigDevices`
   - `TestInferDeviceTypeFromRig`
   - `TestExtractDeviceMapperFromRig`
   - `TestExtractActiveRegions`

### Phase 2.2: Integration Tests

1. Update root `tests/conftest.py` with testcontainers MySQL fixture
2. Create `tests/dj_pipeline/utils/test_load_new_metadata_integration.py`:
   - `TestExtractStreamTypesFromDevice`
   - `TestGetDeviceInfo`
   - `TestGetStreamEntries`
   - `TestInsertStreamTypes`
   - `TestInsertDeviceTypes`
   - `TestInsertDeviceTypesFKHandling`

### Phase 2.3: CI Integration

1. Add pytest markers to `pyproject.toml`
2. Update GitHub Actions workflow
3. Verify tests pass in CI

---

## Test Coverage Targets

| Module | Target | Priority |
|--------|--------|----------|
| `load_new_metadata.py` | 80% | High |
| `streams_maker.py` | 60% | Medium |
| FK handling logic | 100% | Critical |

---

## Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "pytest-cov",
    "testcontainers[mysql]>=3.7",
]
```

**Note:** `swc-aeon` (aeon_api) is already a main dependency and provides `@data_reader`, `BaseSchema`, device classes, and Reader classes needed for all tests.

---

## Bug Fixes During Implementation

### `extract_stream_types_from_device()` closure handling

The `@data_reader` decorator wraps the original function in a closure:

```python
# Original function has (self, pattern) signature
@data_reader
def video(self, pattern) -> Video:
    return Video(f"{pattern}")

# After decoration, cached_property.func is the wrapper with (self) signature
# The original function is in func.__closure__[0].cell_contents
```

The detection logic was updated to check both direct signature AND closure contents:

```python
def extract_stream_types_from_device(device_class: type) -> list[str]:
    # ... check direct signature first ...

    # Check closure for original function (real @data_reader wraps in closure)
    closure = getattr(func, '__closure__', None)
    if closure:
        for cell in closure:
            orig_func = cell.cell_contents
            if callable(orig_func):
                orig_sig = inspect.signature(orig_func)
                orig_params = list(orig_sig.parameters.keys())
                if len(orig_params) == 2 and orig_params[1] == 'pattern':
                    stream_types.append(name)
                    break
```
