# Ephys Pipeline Testing Specification

Testing strategy for the electrophysiology (spike sorting) pipeline, covering epoch discovery through synced spike times.

**Last Updated:** 2026-05-26

---

## Overview

The ephys test suite parallels the existing behavior tests in `test_full_ingestion.py`. It exercises the full ephys ingestion pipeline from epoch discovery through spike time synchronization, using a golden dataset — a reduced 8-channel extract of real NeuropixelsV2 data.

Tests are integration tests that require a database and golden dataset files. They gracefully skip when the data is unavailable.

---

## Golden Dataset

### What it is

An 8-channel subset of a real NeuropixelsV2 ProbeB recording from the `social-ephys0.1-aeon3` experiment, plus pre-computed KS4 sorting output. The reduction from 384 to 8 channels makes the data small enough for fast test runs (~2 min) while still exercising every pipeline stage.

### Contents

| Component | Size | Description |
|-----------|------|-------------|
| `AmplifierData_0.bin` | 1,008 MB | 63M samples × 8 channels × uint16 |
| `Clock_0.bin` | 504 MB | 63M samples × int64 (ONIX timestamps) |
| `HarpSync_*.csv` (3) | ~KB | ONIX↔HARP clock regression data |
| `Metadata.yml` | ~KB | Epoch metadata (device config, gains) |
| `M81_ProbeB_*.json` | ~KB | Probeinterface channel map (remapped to 0-7) |
| `golden_test_sorting/` | ~50 MB | Pre-computed KS4 output (14 units) |

**Source recording:** `social-ephys0.1-aeon3`, epoch `2026-05-11T07-50-11`, subject `IAA-1147881`, electrodes 3982–3989 on shank 3. Duration: 35 minutes continuous.

**Sorting details:** Kilosort4 with modified parameters for small channel count (`nblocks=0`, `Th_universal=8`, `Th_learned=7`). Produced 14 units (7 good, 7 MUA), 357,480 total spikes.

### Storage locations

**Source of truth (Ceph on HPC):**
```
/ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/
├── 2026-05-11T07-50-11/          # epoch data
│   ├── Metadata.yml
│   ├── M81_ProbeB_4Shanks_1000_to_1700_um.json
│   └── NeuropixelsV2/
│       ├── NeuropixelsV2_ProbeB_AmplifierData_0.bin
│       ├── NeuropixelsV2_ProbeB_Clock_0.bin
│       └── NeuropixelsV2_HarpSync_*.csv (3 files)
└── golden_test_sorting/          # pre-computed KS4 output
    ├── recording/recording.dat
    └── sorting_output/
        ├── si_sorting.pkl
        ├── in_container_sorting/
        └── sorter_output/
```

**Local test copy (per-user):**
```
~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/
```

Note: the local path uses `raw/` while Ceph uses `golden_tests/`. This is because the test code expects data under `DEFAULT_GOLDEN_DATA_ROOT / "raw"`, and the golden data is a modified extract (not production data) so it lives in a separate Ceph directory.

---

## Setup Instructions

### 1. Get the golden data

From the HPC (or any machine with Ceph access), rsync the golden dataset to your local home directory:

```bash
mkdir -p ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01
rsync -avP /ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/ \
    ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/
```

If running from outside the HPC (e.g., your local machine):
```bash
rsync -avP aeon-hpc:/ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/ \
    ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/
```

### 2. Clone the branch

```bash
git clone -b es/ephys-tests https://github.com/SainsburyWellcomeCentre/aeon_mecha.git aeon_mecha_ephys-tests
cd aeon_mecha_ephys-tests
```

### 3. Install dependencies

```bash
module load uv
uv sync --all-extras
```

### 4. Database configuration

The tests need access to a DataJoint-compatible database. Set the `TEST_DB_PREFIX` environment variable to use an existing database connection (via `datajoint.json` + `.secrets/`):

```bash
export TEST_DB_PREFIX=u_<username>_golden_tests_
```

This creates test schemas prefixed with that string (e.g., `u_thinh_golden_tests_ephys`). Schemas are dropped automatically after the test session.

If `TEST_DB_PREFIX` is not set, the tests attempt to use testcontainers (Docker-based ephemeral MySQL). This works locally but not on HPC where Docker is unavailable.

**DataJoint config files:** If your clone doesn't have `datajoint.json` and `.secrets/`, symlink from an existing working directory:

```bash
ln -s ~/ProjectAeon/foragingABC_analysis/datajoint.json .
ln -s ~/ProjectAeon/foragingABC_analysis/.secrets .
```

### 5. Run tests

```bash
# Get onto a compute node (Ceph is not visible from the gateway)
srun --cpus-per-task=2 --mem=8G --time=2:00:00 --pty bash

cd ~/ProjectAeon/aeon_mecha_ephys-tests
export TEST_DB_PREFIX=u_<username>_golden_tests_
module load uv

# Run all ephys integration tests
uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py -v --tb=short

# Run a single test class
uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py::TestPreProcessing -v --tb=short
```

Tests take approximately 2 minutes for the full suite.

---

## Test Coverage

### Pipeline stages tested

The tests follow the ephys pipeline cascade in order. Each stage depends on the previous.

| Test Class | Pipeline Stage | What it verifies |
|------------|---------------|------------------|
| `TestEphysEpochDiscovery` | EphysEpoch, ProbeInsertion | Epoch creation, probe discovery, subject linking |
| `TestEphysChunkIngestion` | EphysChunk | Chunk file registration, timestamp validity |
| `TestEphysBlockInfo` | EphysBlockInfo | Block timing, chunk association, channel mappings |
| `TestPreProcessing` | PreProcessing | Binary recording creation, file registration |
| `TestPostProcessing` | PostProcessing | Sorting analyzer creation from golden KS4 output |
| `TestSortedSpikes` | SortedSpikes | Unit extraction, spike counts, quality labels |
| `TestSyncedSpikes` | SyncedSpikes | ONIX→HARP timestamp synchronization |

### Test list (25 tests)

**TestEphysEpochDiscovery** (6 tests):
- Epoch entry exists in DB
- `has_ephys` flag is set
- Correct number of probes discovered (1 — ProbeB only)
- ProbeInsertion entries created
- ProbeInsertion links to correct subject
- `discover_epoch_probes()` returns expected probe labels

**TestEphysChunkIngestion** (3 tests):
- Chunks ingested for the experiment
- Chunk timestamps are valid datetimes
- Chunk files registered in DB

**TestEphysBlockInfo** (4 tests):
- Block info populated
- Block duration matches expected value
- Block-chunk associations exist
- Channel mappings created (8 channels)

**TestPreProcessing** (3 tests):
- PreProcessing entry populated (runs `write_binary_recording`)
- Recording files registered in DB
- `recording.dat` binary file exists on disk

**TestPostProcessing** (2 tests):
- PostProcessing entry populated (runs quality metrics)
- Sorting analyzer artifact created

**TestSortedSpikes** (4 tests):
- Sorted spikes populated
- Unit count matches expected (14 units)
- Spike counts are reasonable (> 0 for each unit)
- Quality labels assigned (good/mua/noise)

**TestSyncedSpikes** (3 tests):
- Synced spikes populated
- Spike times have datetime64 dtype (confirms sync model ran)
- All spike times fall within the sync model's calibrated HARP time range

### How SpikeSorting is handled

The tests do NOT run Kilosort4 (requires GPU + Singularity container). Instead, the `ephys_sorting_injected` fixture:

1. Runs PreProcessing.populate() normally (creates `recording.dat`)
2. Copies the pre-computed golden sorting output into the pipeline's expected directory
3. Force-inserts a SpikeSorting entry with `allow_direct_insert=True`
4. PostProcessing, SortedSpikes, and SyncedSpikes then populate normally through the real pipeline code

---

## Key Design Decisions

### External DB support (`TEST_DB_PREFIX`)

HPC has no Docker, so testcontainers can't run there. When `TEST_DB_PREFIX` is set, the tests use the host's `datajoint.json` configuration and prefix all schema names with the given string. This isolates test schemas from production data.

### Golden data is modified, not production

Unlike the behavior golden dataset (which is unmodified production data rsynced directly from `/ceph/aeon/aeon/data/raw/`), the ephys golden dataset is a modified extract:

- 8 channels extracted from 384 (reduces data from ~50 GB to ~1.5 GB)
- Probeinterface JSON `device_channel_indices` remapped from 384-channel positions to 0-7
- ProbeA stub files removed (only ProbeB retained)

This is why it lives in `/ceph/aeon/aeon/data/golden_tests/` rather than `/ceph/aeon/aeon/data/raw/`.

### No `specialized` test tier

All ephys tests are marked `integration`. The `specialized` tier (for GPU-dependent tests) was considered but deferred — KS4 sorting is injected from pre-computed output rather than run live.

---

## File Structure

```
tests/
├── conftest.py                              # Root: testcontainers, markers, TEST_DB_PREFIX
├── dj_pipeline/
│   ├── conftest.py                          # Golden dataset registry + ephys fixtures
│   ├── test_ephys_ingestion.py              # Ephys integration tests (25 tests)
│   ├── test_full_ingestion.py               # Behavior integration tests (existing)
│   └── utils/
│       └── test_ephys_utils_unit.py         # Ephys utility unit tests
└── fixtures/
    └── ephys/
        └── probe_assignments/               # Test fixture for read_probe_assignments()
```

---

## Troubleshooting

**Tests skip with "Golden data root not found":**
Run the rsync command from the setup instructions. The tests expect data at `~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/`.

**Tests skip with "Golden sorting output not found":**
Make sure you rsynced the full `golden_tests/` directory, including `golden_test_sorting/`.

**"Cannot touch /ceph/..." or Ceph not visible:**
You're on the gateway (`hpc-gw2`). Ceph is only accessible from compute nodes. Use `srun` to get onto one first.

**Schema drop warnings in teardown:**
The teardown drops schemas in arbitrary order, which can hit foreign key constraints. These warnings are cosmetic — the schemas get cleaned up on the next test run.

**Stale test schemas from a crashed run:**
If a previous run crashed without cleanup, manually drop the schemas:
```bash
.venv/bin/python << 'EOF'
import datajoint as dj
dj.config.safemode = False
for s in dj.list_schemas():
    if s.startswith('u_<username>_golden_tests_'):
        print(f'Dropping {s}')
        dj.Schema(s).drop()
EOF
```

---

## Updating Golden Data

The golden dataset (`/ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/`) is
the source of truth for the exact-equality assertions in `test_ephys_ingestion.py`
(`expected_unit_count`, `expected_total_spikes`, and the per-quality-label
counts). Any change to the upstream data — Kilosort version bump, parameter
change, channel-subset change, regeneration of `golden_test_sorting/` — MUST be
paired with an update to the matching constants in the
`foraging_abc_ephys_2026_05_11` entry of `GOLDEN_DATASETS` (in
`tests/dj_pipeline/conftest.py`).

Workflow:

1. Regenerate `golden_test_sorting/sorting_output/` on the HPC compute node.
2. Read the new counts:
   ```python
   import spikeinterface as si
   s = si.load(
       "/ceph/aeon/aeon/data/golden_tests/AEONX1/abcGolden01/"
       "golden_test_sorting/sorting_output/in_container_sorting"
   )
   units = s.get_unit_ids()
   print(f"unit_count={len(units)}")
   print(f"total_spikes={sum(len(s.get_unit_spike_train(u)) for u in units)}")
   ```
3. Update `expected_unit_count`, `expected_total_spikes`, and the quality-label
   counts in the same PR that touches upstream data.

### Probe-electrode-config JSON fixture

The integration tests use `tests/fixtures/ephys/M81_ProbeB_4Shanks_1000_to_1700_um.json`
— a copy of the per-epoch probeinterface JSON from the `2026-05-11T07-50-11`
golden epoch. It's the source of truth for the probe geometry (5120 contacts on
NP 2.0 multishank) and the active electrode subset (384 contacts where
`device_channel_indices != -1`).

The conftest uses
`aeon.dj_pipeline.utils.ephys_utils.create_electrode_config()` to populate the
test DB's `ProbeType` and `ElectrodeConfig` tables from this JSON. The helper
is idempotent — re-running tests doesn't duplicate rows.

If a future epoch uses a different probe configuration, copy its JSON into
`tests/fixtures/ephys/` and update the conftest path / config to reference it.

> **Schema architecture:** the ephys schema was restructured (#583 + #584)
> so that `EphysEpoch` is a peer of `acquisition.Epoch`, with its own
> HARP-native `epoch_start`, and `ElectrodeConfig` carries a structured
> `config_file_name` for per-epoch-per-probe resolution. See
> `SPEC_EPHYS_PIPELINE.md` for the full architecture.
>
> The `ephys_test_epochs` fixture drives the new ingest chain end-to-end:
> `EphysEpoch.ingest_epochs() → EphysEpochConfig.populate() →
> EphysSyncModel.ingest()`. No manual `acquisition.Epoch` insert needed.

> **Note on `golden_test_sorting/`:** the pre-computed Kilosort output
> directory used by `ephys_sorting_injected` is not yet uploaded to S3. When
> the local copy is missing, the 9 spike-sorting-cascade tests
> (PostProcessing/SortedSpikes/SyncedSpikes) skip cleanly via the existing
> `pytest.skip` in the fixture. The remaining 16 tests cover EphysEpoch
> through PreProcessing.
