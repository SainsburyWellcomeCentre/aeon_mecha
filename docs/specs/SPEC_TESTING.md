# Testing Specification for streams_maker & load_metadata

Testing strategy for the Pydantic-based metadata loading and DataJoint table generation pipeline.

**Last Updated:** 2026-02-19

---

## Architecture

**Three tiers:**

| Tier | Scope | Database | Data | CI |
|------|-------|----------|------|-----|
| Unit | Pure functions | None | Synthetic dicts + sample fixtures | Yes |
| Integration | DB operations, `populate()` | MySQL (testcontainers) | Synthetic Pydantic OR golden datasets | Yes |
| Specialized | GPU/high-RAM `populate()` | MySQL (testcontainers) | Golden datasets | No |

**Key patterns:**
- Golden datasets (real data, strict assertions) for integration/specialized tests
- Testcontainers (zero-config MySQL)
- Auto-marking (no manual `@pytest.mark` needed)
- Graceful skipping when data unavailable

---

## Test Classification

### What makes a test "integration"?

Integration tests verify that components work together correctly. They require external systems (database, file system) and can use different data sources:

| Test Type | Database | Data Source | What it tests |
|-----------|----------|-------------|---------------|
| Unit | No | Synthetic/fixtures | Pure functions, logic |
| Integration | Yes | Synthetic OR golden datasets | DB operations, `populate()` |
| Specialized | Yes | Golden datasets | GPU/high-RAM operations |

**Integration tests have two sub-categories:**

1. **Schema/catalog integration** - Tests DB operations with synthetic data
   - Example: `insert_stream_types()`, FK constraint handling
   - Data: Synthetic Pydantic fixtures
   - No file I/O

2. **Data ingestion integration** - Tests `populate()` with real files
   - Example: `acquisition.Chunk.populate()`, `streams.Video.populate()`
   - Data: Golden datasets (real experiment data)
   - Reads Harp binary, video, CSV files

### Current test coverage

| Category | What it tests | Data Source | Status |
|----------|--------------|-------------|--------|
| Unit tests | Pure functions | Synthetic | ✅ Implemented |
| Integration (schema) | `insert_stream_types()`, FK handling | Synthetic Pydantic | ✅ Implemented |
| Integration (ingestion) | `EpochConfig.make()`, `Chunk.populate()`, `streams.*.populate()` | Golden datasets | 🔄 In Progress |
| Specialized | SLEAP/DLC tracking | Golden datasets | ❌ Needs GPU |

**Note:** Current tests in `test_load_metadata_integration.py` test schema/catalog operations with synthetic data. Golden dataset tests in `test_full_ingestion.py` test full `populate()` flow.

---

## Data Types for Testing

### Sample Fixtures (in git)

Small configuration files checked into `tests/fixtures/`:

```
tests/fixtures/
├── metadata/
│   └── ForagingABC_Metadata.json   # ~15KB sample metadata config
└── manifests/                       # File manifests for golden datasets
```

**Characteristics:**
- Small enough for git (< 1MB typically)
- Configuration samples, schema examples, manifests
- Used for unit and integration tests of parsing logic
- Can be modified when format changes

### Golden Datasets (external storage)

**A golden dataset is a frozen, representative sample of real production data designated for testing.**

Unlike synthetic test data or sample fixtures, golden datasets:

- Come from actual experiment sessions with known, validated outputs
- Are **never modified** once designated - any data issues are documented, not fixed
- Enable **strict assertions** (exact expected values, not ranges or approximations)
- Are **versioned by session** (e.g., `exp02-aeon3/2024.05.15`) so tests are reproducible
- Are **large** (GBs) - video files, Harp binary streams, pose estimation outputs
- Stored outside git (S3, NFS, or similar)
- Tests gracefully skip when data is unavailable

**Why golden datasets instead of synthetic data for `populate()` tests?**
- DataJoint `populate()` reads real files with complex formats (Harp binary, video, CSV)
- Mocking file I/O at this scale is fragile and misleading
- Real data catches edge cases synthetic data misses

### Comparison

| Aspect | Sample Fixtures | Golden Datasets |
|--------|-----------------|-----------------|
| Size | KB-MB | GB+ |
| Storage | In git (`tests/fixtures/`) | External (S3, NFS) |
| Purpose | Test parsing, config validation | Test full `populate()` flow |
| Mutability | Can be updated | Never modified |
| CI availability | Always | Optional / scheduled |

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
├── conftest.py                           # Root: DJ env vars, DJ mocking, testcontainers, auto-marking
├── dj_pipeline/
│   ├── conftest.py                       # Golden dataset registry, DB fixtures, integration setup
│   ├── test_full_ingestion.py            # Golden dataset integration tests
│   └── utils/
│       ├── conftest.py                   # Test Rig fixtures (real @data_reader)
│       ├── test_load_metadata_unit.py    # Unit tests (no DB)
│       └── test_load_metadata_integration.py  # Schema integration tests (DB)
└── fixtures/
    └── metadata/                         # Sample Metadata.json files
```

---

## Test Categories

### 1. Unit Tests (`test_load_metadata_unit.py`)

Pure function tests - no database, no external packages required:

| Function | Test Cases |
|----------|------------|
| `to_pascal_case()` | `"video"` → `"Video"`, `"beam_break"` → `"BeamBreak"` |
| `_flatten_rig_devices()` | Flatten nested rig config to device dict |
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

### 2. Integration Tests (`test_load_metadata_integration.py`)

Database operations using real Pydantic classes with real `@data_reader` decorator:

| Function | Test Cases |
|----------|------------|
| `get_data_reader_methods()` | Extract @data_reader methods from Device class |
| `get_device_info()` | Extract device/stream info from test Rig |
| `get_stream_entries()` | Generate StreamType entries from test Rig |
| `get_device_mapper_from_rig()` | Extract device type and serial number mappings |
| `insert_stream_types()` | Insert into StreamType, handle duplicates |
| `insert_device_types()` | Insert DeviceType, DeviceType.Stream, Device; FK handling |

**Example:**
```python
@pytest.mark.integration
class TestInsertStreamTypes:
    def test_inserts_stream_types(self, pipeline_integration, test_rig):
        """Verify StreamType entries are inserted from Rig."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)
        initial_count = len(streams.StreamType())

        insert_stream_types(test_rig)

        assert len(streams.StreamType()) > initial_count

    def test_handles_duplicates(self, pipeline_integration, test_rig):
        """Verify duplicate insertions are handled gracefully (skip_duplicates)."""
        streams = dj.VirtualModule("streams", streams_maker.schema_name)

        insert_stream_types(test_rig)
        count_after_first = len(streams.StreamType())

        # Second call should not raise or create duplicates
        insert_stream_types(test_rig)

        assert len(streams.StreamType()) == count_after_first


@pytest.mark.integration
class TestInsertDeviceTypesFKHandling:
    def test_fk_constraint_triggers_stream_type_insert(self, pipeline_integration, test_rig, tmp_path):
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

    def test_non_fk_errors_are_reraised(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
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

import os
import pytest


@pytest.fixture(scope="session")
def mysql_container():
    """Auto-provision MySQL via testcontainers."""
    from testcontainers.mysql import MySqlContainer

    container = MySqlContainer(
        image="mysql:8.0",
        username="root",
        password="test_password",
        dbname="test_db",
    )
    container.start()

    # Update environment variables with container details
    host = container.get_container_host_ip()
    port = container.get_exposed_port(3306)
    os.environ["DJ_HOST"] = host
    os.environ["DJ_PORT"] = str(port)
    os.environ["DJ_USER"] = "root"
    os.environ["DJ_PASS"] = "test_password"

    yield container
    container.stop()
```

The `dj_config_integration` fixture (in `tests/dj_pipeline/conftest.py`) reads from these env vars and configures DataJoint:

```python
@pytest.fixture(scope="session")
def dj_config_integration(mysql_container):
    """Configure DataJoint to use testcontainers MySQL."""
    import datajoint as dj

    dj.config.update({
        "safemode": False,
        "database.host": os.environ.get("DJ_HOST", "localhost"),
        "database.port": int(os.environ.get("DJ_PORT", "3306")),
        "database.user": os.environ.get("DJ_USER", "root"),
        "database.password": os.environ.get("DJ_PASS", "test_password"),
    })
    dj.config["custom"]["database.prefix"] = "test_integration_"
    return dj.config
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
    integration_fixtures = {"mysql_container", "pipeline_integration", "dj_config_integration"}
    for item in items:
        if integration_fixtures & set(item.fixturenames):
            item.add_marker(pytest.mark.integration)
```

---

## Pytest Markers

Register in `pyproject.toml`:

```toml
[tool.pytest]
markers = [
    "unit: Unit tests (no database, synthetic data)",
    "integration: Integration tests (DB via testcontainers, synthetic or golden datasets)",
    "specialized: Specialized tests (GPU/high-RAM, golden datasets)",
]
testpaths = ["tests"]
```

---

## CI Strategy

### Trigger Matrix

| Event | Unit Tests | Integration Tests (Schema) | Integration Tests (Golden) |
|-------|------------|---------------------------|---------------------------|
| PR to `datajoint_pipeline` | Yes | Yes | Skip (no data) |
| `workflow_dispatch` | Yes | Yes | Skip (no data) |

### GitHub Actions Workflow

Two parallel jobs using `uv` (not `pip`):

```yaml
name: test_dj_pipeline

on:
  pull_request:
    branches: [datajoint_pipeline]
    types: [opened, reopened, synchronize]
  workflow_dispatch:

jobs:
  unit-tests:
    name: Unit tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    if: github.event.pull_request.draft == false
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync --extra test
      - run: uv run pytest -m unit --tb=short -q

  integration-tests:
    name: Integration tests (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    if: github.event.pull_request.draft == false
    strategy:
      matrix:
        python-version: ["3.11"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install ${{ matrix.python-version }}
      - run: uv sync --extra test
      - run: uv run pytest -m integration --tb=short -q
```

**Note:** Testcontainers automatically handles MySQL provisioning - no `services:` block needed. Golden dataset tests skip automatically when data is unavailable.

---

## Commands

```bash
# Unit tests only (fast, no database)
uv run pytest -m unit -v

# Integration tests only (auto-provisions MySQL)
uv run pytest -m integration -v

# All tests
uv run pytest -v

# Specific test file
uv run pytest tests/dj_pipeline/utils/test_load_metadata_unit.py -v

# With coverage
uv run pytest --cov=aeon.dj_pipeline.utils --cov-report=html

# Debug single test
uv run pytest tests/dj_pipeline/utils/test_load_metadata_unit.py::TestToPascalCase -v --pdb
```

---

## Implementation Plan

### Phase 2.1: Setup & Unit Tests

1. Add `testcontainers[mysql]` to test dependencies (swc-aeon is already a main dependency)
2. Create `tests/dj_pipeline/utils/conftest.py` with fixtures
3. Create `tests/dj_pipeline/utils/test_load_metadata_unit.py`:
   - `TestToPascalCase`
   - `TestFlattenRigDevices`
   - `TestInferDeviceTypeFromRig`
   - `TestExtractDeviceMapperFromRig`
   - `TestExtractActiveRegions`

### Phase 2.2: Integration Tests

1. Update root `tests/conftest.py` with testcontainers MySQL fixture
2. Create `tests/dj_pipeline/utils/test_load_metadata_integration.py`:
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
| `load_metadata.py` | 80% | High |
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

Install with: `uv sync --extra test`

**Note:** `swc-aeon` (aeon_api) is already a main dependency and provides `@data_reader`, `BaseSchema`, device classes, and Reader classes needed for all tests.

---

## Implementation Notes

### `get_data_reader_methods()` closure handling

The `@data_reader` decorator wraps the original function in a closure. The `get_data_reader_methods()` function checks both the direct signature AND closure contents to detect @data_reader methods:

```python
# After decoration, cached_property.func is the wrapper
# The original function with (self, pattern) signature is in func.__closure__
closure = getattr(func, '__closure__', None)
if closure:
    for cell in closure:
        orig_func = cell.cell_contents
        if callable(orig_func):
            orig_sig = inspect.signature(orig_func)
            orig_params = list(orig_sig.parameters.keys())
            if len(orig_params) == 2 and orig_params[1] == 'pattern':
                # Found @data_reader method
                break
```

---

## Golden Dataset Integration Tests

This section details integration tests that validate the full ingestion pipeline using golden datasets.

### Overview

Golden dataset tests validate:
1. `EpochConfig.make()` - Parse real Metadata.json with experiment-specific Pydantic schema
2. `Chunk.ingest_chunks()` - Detect and ingest chunk files from filesystem
3. `DeviceDataStream.make()` - Load ALL stream types with `populate(limit=10)`

### Golden Dataset Registry

Datasets are configured via a plain dict registry to support multiple datasets:

```python
# tests/dj_pipeline/conftest.py

from pathlib import Path

GOLDEN_DATASETS = {
    "foraging_abc_2025_11_18": {
        "experiment_name": "abcBehav0-aeon3",
        "experiment_path": "AEON3/abcBehav0",
        "epoch_dir": "2025-11-18T10-13-15",
        "devices_schema": "swc.aeon.exp.foragingABC.experiment:Experiment",
        "arena_name": "arena-aeon3",
        "lab": "SWC",
        "location": "room-0",
        "experiment_type": "foraging",
        "required_files": [
            "Metadata.json",
            "CameraTop/CameraTop_2025-11-18T10-00-00.csv",
            "Feeder1/Feeder1_32_2025-11-18T10-00-00.bin",
        ],
        "expected_camera_count": 13,
        "expected_feeder_count": 6,
    },
    # Future datasets can be added here
}

DEFAULT_GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"
```

### Path Configuration

Golden dataset tests use aeon_mecha's existing `repository_config` pattern:

```python
# In fixture - configure dj.config the same way production does
dj.config["custom"]["repository_config"] = {
    "ceph_aeon": str(golden_data_root)
}

# In production code (unchanged)
from aeon.dj_pipeline.utils.paths import get_repository_path
repo_path = get_repository_path("ceph_aeon")
```

This maintains consistency with how paths are configured in `dj_local_conf.json` for production deployments.

### Key Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `golden_dataset_config` | session | Active dataset configuration from registry |
| `dj_config_with_golden_data` | session | Configure `repository_config` for golden data |
| `require_golden_data` | session | Skip tests if data unavailable, validate files |
| `full_pipeline` | session | All schema modules (lab, subject, acquisition, streams) + catalog + table creation |
| `test_experiment` | session | Insert experiment, arena, directories from config |
| `test_epochs` | session | Ingest epochs using `Epoch.ingest_epochs()` |

### Test Classes

| Class | Tests | Description |
|-------|-------|-------------|
| `TestStep1CatalogPopulation` | 3 | Verify StreamType, DeviceType, DeviceType.Stream populated |
| `TestStep2TableCreation` | 2 | Verify ExperimentDevice and DeviceDataStream tables exist |
| `TestEpochIngestion` | 2 | `Epoch.ingest_epochs()` filesystem detection |
| `TestEpochConfigMake` | 3 | `EpochConfig.populate()` with metadata assertions |
| `TestChunkIngestion` | 2 | `Chunk.ingest_chunks()` file detection |
| `TestStreamDataIngestion` | 3 | ALL streams with `populate(limit=10)` |

### Stream Testing Strategy

**Key insight:** Test ALL stream types, but limit entries per stream.

```python
POPULATE_LIMIT = 10

def test_all_stream_tables_populate(self, ...):
    """Verify ALL stream tables can populate with limit."""
    stream_tables = [...]  # All DeviceDataStream tables

    for table in stream_tables:
        table.populate(limit=POPULATE_LIMIT, display_progress=False)

    # Assert at least some tables have data
    populated = [t for t in stream_tables if len(t) > 0]
    assert len(populated) > 0
```

This validates:
- Every `DeviceDataStream` table can be created
- Every reader can parse its file format
- FK relationships are correct

Without:
- Ingesting gigabytes of data
- Long test execution times

### Directory Structure

```
tests/
├── conftest.py                              # Root: testcontainers, markers
├── dj_pipeline/
│   ├── conftest.py                          # Golden dataset registry + DB fixtures
│   ├── test_full_ingestion.py               # Golden dataset integration tests
│   └── utils/
│       ├── conftest.py                      # Test Rig fixtures
│       ├── test_load_metadata_unit.py       # Unit tests
│       └── test_load_metadata_integration.py # Schema integration tests
└── fixtures/
    └── metadata/                            # Sample fixtures (in git)
```

### Adding New Golden Datasets

1. **Copy data** to golden data root:
   ```bash
   rsync -avP aeon-hpc:/ceph/aeon/aeon/data/raw/EXPERIMENT/epoch/ \
       ~/sciops-data/project_aeon/aeon/data/raw/EXPERIMENT/epoch/
   ```

2. **Add to registry** in `tests/dj_pipeline/conftest.py`:
   ```python
   GOLDEN_DATASETS["new_dataset_name"] = {
       "experiment_name": "expName-aeonN",
       "experiment_path": "AEONN/expName",
       "epoch_dir": "YYYY-MM-DDTHH-MM-SS",
       "devices_schema": "swc.aeon.exp.expName.experiment:Experiment",
       "arena_name": "arena-aeonN",
       "lab": "SWC",
       "location": "room-0",
       "experiment_type": "foraging",
       "required_files": ["Metadata.json", ...],
       "expected_camera_count": N,
       "expected_feeder_count": N,
   }
   ```

3. **Update active dataset** (optional) or **parametrize** to test multiple:
   ```python
   @pytest.mark.parametrize("dataset_name", list(GOLDEN_DATASETS.keys()))
   def test_with_all_datasets(dataset_name, ...):
       cfg = GOLDEN_DATASETS[dataset_name]
   ```

### CI Strategy (Golden Datasets)

| Event | Unit | Integration (Schema) | Integration (Golden) |
|-------|------|---------------------|---------------------|
| PR to `datajoint_pipeline` | Yes | Yes | Skip (no data) |
| `workflow_dispatch` | Yes | Yes | Skip (no data) |
| Local dev | Yes | Yes | Yes (if available) |

Golden dataset tests gracefully skip with clear messages when data is unavailable.

### Implementation Phases

#### Phase 3.1: Infrastructure

1. Add `aeon_exp_foragingABC` as test dependency in `pyproject.toml`
2. Add `GoldenDataset` dataclass and registry to `tests/dj_pipeline/conftest.py`
3. Add `dj_config_with_golden_data` and `require_golden_data` fixtures

#### Phase 3.2: Experiment Setup

1. Add `full_pipeline` fixture with all schemas
2. Add `test_experiment` fixture (generic, from config)
3. Add `test_epoch` fixture (generic, from config)

#### Phase 3.3: Core Tests

1. Create `tests/dj_pipeline/test_full_ingestion.py`
2. Implement `TestEpochConfigMake` (4 tests)
3. Implement `TestChunkIngestion` (2 tests)
4. Implement `TestStreamDataIngestion` (3 tests, uses `populate(limit=10)`)

### Dependencies (Additional)

Add to `pyproject.toml`:

```toml
[tool.uv.sources]
swc-aeon-exp-foragingabc = { git = "https://github.com/SainsburyWellcomeCentre/aeon_exp_foragingABC.git", branch = "data-api" }
```

### Commands (Golden Dataset Tests)

```bash
# Run golden dataset tests (skips if data unavailable)
uv run pytest tests/dj_pipeline/test_full_ingestion.py -v

# Run all integration tests (schema + golden, golden skips if no data)
uv run pytest -m integration -v
```

### Success Criteria

- [ ] `EpochConfig.make()` completes with golden Metadata.json
- [ ] `Chunk.ingest_chunks()` detects and ingests chunk
- [ ] ALL stream tables can `populate(limit=10)`
- [ ] Tests skip gracefully when data unavailable
- [ ] No modification to production `streams.py`
- [ ] Existing tests still pass
