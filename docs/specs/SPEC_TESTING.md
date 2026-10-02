# Testing Specification

How tests are organized, what they cover, how to run them, and how to add new ones.

---

## Test tiers

Three tiers, distinguished by what they need and what they cover:

| Tier | Database | Data | Marker | Runs in CI? |
|---|---|---|---|---|
| Unit | None | Synthetic / sample fixtures | `@pytest.mark.unit` | All OSes |
| Integration | MySQL (testcontainers) | Synthetic or golden datasets | `@pytest.mark.integration` | Linux only |
| Specialized | MySQL | Golden datasets + GPU | `@pytest.mark.specialized` | No (manual) |

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`. Tests using integration fixtures (`mysql_container`, `dj_config_integration`, `pipeline_integration`) are **auto-marked** as integration by a `pytest_collection_modifyitems` hook in `tests/conftest.py` — no manual marker needed for those.

---

## Where tests live

```
tests/
├── conftest.py                                       # testcontainers MySQL, auto-marking
├── fixtures/
│   ├── metadata/                                     # sample Metadata.json files (in git)
│   └── ephys/                                        # ProbeInterface JSON, synthetic factories
├── dj_pipeline/
│   ├── conftest.py                                   # GOLDEN_DATASETS registry, golden-data fixtures
│   ├── test_full_ingestion.py                        # behavior golden dataset
│   ├── test_ephys_ingestion.py                       # ephys golden dataset
│   ├── test_ephys_synthetic_integration.py           # ephys schema-invariant tests (no golden data)
│   ├── test_onix_imu_pipeline_integration.py         # synthetic IMU
│   ├── test_acquisition_environment_integration.py
│   └── utils/
│       ├── conftest.py                               # test Rig fixtures (real @data_reader)
│       ├── test_*_unit.py
│       └── test_*_integration.py
└── schema/
    └── test_ephys_reader_unit.py
```

---

## Test infrastructure

**Testcontainers MySQL.** `tests/conftest.py:mysql_container` spins up a fresh MySQL container per test session and exports its host/port/user/pass as env vars. The `dj_config_integration` fixture (in `tests/dj_pipeline/conftest.py`) reads those and configures DataJoint with `TEST_DB_PREFIX = "test_aeon_"`. Zero local setup; works the same in CI.

**Session-scoped DB fixtures.** Test schemas are expensive to create/drop, and most integration tests build on each other (insert StreamType → insert DeviceType.Stream → ...). The schema lives for the whole session and is dropped at teardown via `_drop_test_schemas()` (which guards against unsafe prefixes — see `tests/dj_pipeline/test_conftest_unit.py`).

**Real Pydantic + real `@data_reader`.** The test Rig classes in `tests/dj_pipeline/utils/conftest.py` use the actual `swc-aeon` base classes (`BaseSchema`, `SpinnakerCamera`, `UndergroundFeeder`) and the real `@data_reader` decorator — no mocking. Reader classes do no I/O until `load()` is called, so unit tests stay fast.

---

## Sample fixtures vs golden datasets

| | Sample fixtures | Golden datasets |
|---|---|---|
| Where | In git (`tests/fixtures/`) | On disk, configured per machine |
| Size | < 2 MB | 100+ GB |
| Purpose | Test parsing, config validation, schema invariants | Test full `populate()` flow against real files |
| Mutability | Edit freely | Never modified — issues get documented |
| Available in CI | Always | No (graceful skip) |

Golden datasets enable **exact-equality assertions** (e.g. expected unit count, expected spike count). Synthetic data tests structural correctness; golden data tests numerical correctness.

---

## Golden dataset registry

All golden datasets are configured in one dict in `tests/dj_pipeline/conftest.py`:

```python
GOLDEN_DATASETS = {
    "<dataset_key>": {
        "experiment_name": ...,
        "experiment_path": ...,
        "epoch_dir": ...,
        "required_files": [...],   # tests skip if any of these are missing
        # ...dataset-specific keys (expected counts, probe info, etc.)
    },
}
DEFAULT_GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"
```

Override the root with the `DJ_REPOSITORY_CONFIG` env var (used on HPC). The `_check_golden_data` helper validates `required_files` and calls `pytest.skip(...)` if anything's missing — so tests just don't run when the data isn't there.

### Adding a new golden dataset

1. `rsync` the data into `~/sciops-data/project_aeon/aeon/data/...` (or wherever your `DEFAULT_GOLDEN_DATA_ROOT` points).
2. Add an entry to `GOLDEN_DATASETS` with `required_files` and any expected-value constants.
3. Add a fixture chain in `tests/dj_pipeline/conftest.py` modeled on the existing ones (`*_test_experiment` → `*_test_epochs` → ...).
4. Add a test module that consumes those fixtures and asserts against the expected constants.

---

## Behavior golden datasets

Tests parametrize the `golden_dataset_config` fixture over both registered
datasets — every behavior test runs once per dataset:

| Key | Experiment | Duration | On-disk profile | Pair |
|---|---|---|---|---|
| `foraging_abc_2025_11_18` | `abcBehav0-aeon3` | ~1 hour | Full 13 cameras + 6 feeders writing data; rich stream samples (`FeederEncoder` 848k, `CameraPosition` 12k) | Behavior only |
| `foraging_abc_2026_05_11` | `abcGolden01-aeon3` | ~2 hours | 5 cameras + 4 feeders writing data; sparser samples; paired with the ephys golden | Paired with `foraging_abc_ephys_2026_05_11` |

**Locations** (relative to `DEFAULT_GOLDEN_DATA_ROOT`):
- `<data-root>/raw/AEON3/abcBehav0/2025-11-18T10-13-15/`
- `<data-root>/raw/AEON3/abcGolden01/2026-05-11T075134Z/`

**Test modules:** `tests/dj_pipeline/test_full_ingestion.py`,
`tests/dj_pipeline/test_acquisition_environment_integration.py`

**Reference device:** `CameraTop` by default; `abcGolden01-aeon3` overrides
to `CameraNest` via `_ref_device_mapping` in `acquisition.py` (that rig has
no `CameraTop` on disk).

**Mixed file-name formats** in abcGolden01 — CSVs use `T07-00-00`, newer
bins use `T070000Z`. Both parse via `swc.aeon.io.api.chunk_key`.

**Covers:**
- `Epoch.ingest_epochs()` — filesystem detection of epoch directories
- `EpochConfig.populate()` — Metadata.json parsing via foragingABC Pydantic schema
- `Chunk.ingest_chunks()` — chunk file detection
- All `DeviceDataStream` tables — `populate(max_calls=10)` per stream
- The 3 `acquisition.Environment*` tables (`EnvironmentState`, `MessageLog`, `LightEvents`)
- `TestStreamPopulationInventory` — diagnostic inventory of which streams populate

**Run:**
```bash
uv run pytest -m integration tests/dj_pipeline/test_full_ingestion.py -v
```

**Required deps:** `uv sync --group test-golden` (installs `swc-aeon-rigs-foragingabc`).
Without it, the test module skips at import time via `pytest.importorskip`.

---

## Ephys golden dataset

**Active dataset:** `foraging_abc_ephys_2026_05_11` — `abcGolden01-aeonx1`, NeuropixelsV2 ProbeB, all 96 active contacts on shank 3. Two overlapping ephys blocks (chunks 0-6 and 4-11, sharing 4-6), each with a real Kilosort4 sorting under `golden_test_sorting/<block>/shank3/kilosort4_400/`.

**Location:** `~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/2026-05-11T07-50-11/`

**Test modules:**
- `tests/dj_pipeline/test_ephys_ingestion.py` — golden-data ingestion through `SyncedSpikes` (`integration`)
- `tests/dj_pipeline/test_ephys_curation_integration.py` — auto-approved curation (`specialized`)
- `tests/dj_pipeline/test_unit_matching_integration.py` — `UnitMatching` / `GlobalUnit` (`specialized`)
- `tests/dj_pipeline/test_unit_matching_helpers_unit.py` — pure matching helpers (`unit`, no DB)
- `tests/dj_pipeline/test_ephys_synthetic_integration.py` — schema invariants (no golden data)

The curation and matching modules are `specialized` because they drive the same ~18-minute
fixture chain; running them in one invocation builds it once.

**Covers (golden):**
- `EphysEpoch.ingest_epochs()` + `EphysEpochConfig.populate()` — probe discovery
- `EphysSyncModel.ingest()` — HARP↔ONIX regression per HarpSync CSV
- `EphysChunk.ingest_chunks()` — chunk file detection + per-epoch-probe config resolution
- `EphysBlockInfo.populate()` — block-level channel mapping (multi-config validation enforced)
- `OnixImuChunk.populate()` — Bno055 IMU streams (overlap-based chunk selection)
- `PreProcessing.populate()` — bandpass + CAR
- `PostProcessing` / `SortedSpikes` / `SyncedSpikes` (when `golden_test_sorting/` is present — currently skipped on most setups)

**Probe-electrode-config JSON fixture:** `tests/fixtures/ephys/M81_ProbeB_4Shanks_1000_to_1700_um.json` — copy of the per-epoch ProbeInterface JSON, source of truth for the probe geometry (5120 contacts) and active electrode subset (384). Used by `create_electrode_config` in tests.

### HPC setup

The full ephys suite runs on the HPC against `/ceph` data. From `aeon-hpc`:

1. **Get onto a compute node** (the suite is heavy; run it off the shared gateway):
   ```bash
   srun --cpus-per-task=2 --mem=8G --time=2:00:00 --pty bash
   ```
2. **Get the golden data** (rsync from Ceph to your home; or symlink if you trust the path):
   ```bash
   rsync -avP /ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/ \
       ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/
   ```
3. **Set test DB prefix** to your username so tests don't collide with others:
   ```bash
   export TEST_DB_PREFIX=u_${USER}_golden_tests_
   ```
4. **Run:**
   ```bash
   module load uv
   # ingestion through SyncedSpikes
   uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py -v --tb=short
   # curation + unit matching (shares one fixture build; ~18 min)
   uv run pytest -m specialized \
     tests/dj_pipeline/test_ephys_curation_integration.py \
     tests/dj_pipeline/test_unit_matching_integration.py -v --tb=short
   ```

Without `TEST_DB_PREFIX`, tests fall back to testcontainers (Docker), which isn't available on HPC.

### Refreshing golden data

Unit and spike counts are **derived from the artifacts at fixture time**, not hardcoded, so a
regenerated sorting needs no constants updated.

What is hardcoded is geometry, in `GOLDEN_DATASETS["foraging_abc_ephys_2026_05_11"]` and
`ephys_test_blocks`: the two `golden_blocks` directory names, `n_channels: 96`, `shank_id: "3"`,
and the chunk slices each block must select (`0..6` and `4..11`). Re-sorting on different block
bounds or a different channel subset means updating those in the **same PR** that touches the data.

`ephys_block_info_populated` asserts the exact chunk identities each block links, so a mismatch
between bounds and artifact fails at fixture setup rather than as an `IndexError` inside
`SyncedSpikes` fifteen minutes later.

---

## CI

Workflow: `.github/workflows/lint_and_test.yml`.

| Job | Matrix | Triggers |
|---|---|---|
| **lint** | Ubuntu + Python 3.12 (ruff + pyright) | All pushes, all PRs, manual, `v*` tags |
| **tests** (full) | Ubuntu + Python 3.11/3.12 — unit + integration + coverage | Same as above |
| **tests** (smoke) | macOS + Windows, Python 3.12 — `-m unit` only | Same as above |

Coverage is uploaded to codecov from the Ubuntu + Python 3.12 run. Concurrency cancels in-flight runs on the same ref except for `main` and tags.

Golden dataset tests skip cleanly on CI (no data) — no need to disable them by marker.

---

## Common commands

```bash
# Unit tests only (fast, no DB)
uv run pytest -m unit

# Integration tests only (testcontainers spins up MySQL automatically)
uv run pytest -m integration

# Full ephys suite
uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py -v

# Single test class
uv run pytest tests/dj_pipeline/test_ephys_ingestion.py::TestEphysSyncModel -v

# With coverage
uv run pytest --cov=aeon --cov-report=html
```

---

## Troubleshooting

**"Cannot connect to Docker daemon" on local runs.** Testcontainers needs Docker. On HPC use `TEST_DB_PREFIX` against an external DB instead (see HPC setup).

**"Cannot touch /ceph/..." / Ceph access problems.** Ceph is now mounted on the gateway as well as the compute nodes, so this is rare; if you do hit it, `srun` onto a compute node.

**Stale test schemas from a crashed run.** Manually drop:
```bash
uv run python -c "
import datajoint as dj
dj.config.safemode = False
for s in dj.list_schemas():
    if s.startswith('u_${USER}_golden_tests_'):
        print(f'Dropping {s}')
        dj.Schema(s).drop()
"
```

**Schema-drop teardown warnings.** The teardown drops schemas in arbitrary order, which can hit FK constraints. The retry loop handles it but emits warnings — cosmetic, can be ignored.

**`ephys_test_epochs` fixture takes a long time.** First run sets up the whole DB + downloads `<attach>` columns. Subsequent runs reuse the session-scoped fixture; only the actual test bodies re-run.
