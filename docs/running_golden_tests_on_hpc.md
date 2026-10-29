# Running the ephys golden-dataset tests on the SWC HPC

How to run the ephys golden integration suite
(`tests/dj_pipeline/test_ephys_ingestion.py`) against the golden dataset on the
SWC HPC. The suite force-injects the pre-computed sorting, so it needs **no
GPU** — a CPU node is enough. Most of the runtime is PreProcessing and
PostProcessing reading the amplifier data off Ceph, so the full suite takes
hours.

There are two ways to run it:

- **[Batch job](#running-the-full-suite-batch)** — recommended for the full suite.
- **[Interactive session](#running-subsets-interactively)** — for
  debugging and for running subsets of the suite.

## Prerequisites

- HPC access (e.g. `ssh aeon-hpc`).
- A clone of the code (see [Code](#code)).
- A database account on `aeon-db` or `aeondj` (either works — pick one), and the
  credentials set up in your clone (see [Database](#database)). If you don't
  have an account yet, set one up via the team's DB onboarding or ask Thinh/Elissa.
- Access to the golden data, which lives on Ceph (see
  [Golden data](#golden-data)).

### Code

If you do not already have a local clone of aeon_mecha, create one:

```
git clone git@github.com:SainsburyWellcomeCentre/aeon_mecha.git
```

Then navigate to the cloned repository:
```
cd aeon_mecha
```

### Database

The integration tests normally start a throwaway MySQL instance via
[testcontainers](https://testcontainers.com/), which requires a Docker daemon.
Since Docker is not available on the HPC, the tests must connect to an external
database instead.

Create a `datajoint.json` and `.secrets/` in the root of your clone (where you
run everything) with your own credentials, as shown below. The test harness uses them instead of
testcontainers when `TEST_DB_PREFIX` is set.

`datajoint.json` (set `host` to whichever server you're using):

```json
{
  "database": {
    "host": "aeondj",
    "port": 3306,
    "reconnect": true,
    "database_prefix": "u_<your_user>_test_"
  },
  "stores": {
    "dj_store": {
      "protocol": "file",
      "location": "/ceph/aeon/aeon/dj_store",
      "stage": "/ceph/aeon/aeon/dj_store"
    }
  }
}
```

`.secrets/` with two files holding just the credential values:

```
.secrets/database.user       # your DB username
.secrets/database.password   # your DB password (for aeondj, your aeondj password)
```

(The test fixtures override the `dj_store` to a per-session temp dir, so the
`stores` entry above is just to make the config valid — the tests don't write to
the production store.)

### Golden data

The golden dataset lives on Ceph under `/ceph/aeon`. Confirm the ephys golden
epoch and its required files are visible there before running. The dataset's
`experiment_path`, `epoch_dir`, and `required_files` are listed in the
`GOLDEN_DATASETS` registry in `tests/dj_pipeline/conftest.py` (the ephys entry is
`foraging_abc_ephys_2026_05_11`). A quick `ls` of the epoch directory is a good
check.

> [!IMPORTANT]
> If the data isn't found at the configured root, `_check_golden_data` calls
> `pytest.skip(...)` rather than failing — so a run full of **skips** means the
> data root isn't set right, not that everything passed.

## Running the full suite (batch)

> [!NOTE]
> The full suite takes hours. To run only
> part of it, see [Running subsets interactively](#running-subsets-interactively).

Submit from the repo root:

```
sbatch tests/run_full_test_suite.sh
```

The script ([tests/run_full_test_suite.sh](../tests/run_full_test_suite.sh)) does the
following:

1. Sets `TEST_DB_PREFIX` to `u_${USER}_golden_tests_` and `DJ_REPOSITORY_CONFIG`
   to the Ceph copy of the golden data.
2. Loads `uv` and installs the required dependencies (e.g. 
   spikeinterface/probeinterface for the ephys tests,
   `swc-aeon-rigs-foragingabc` for the golden fixtures).
3. Runs a DB preflight before starting pytest. It prints the host, user and
   prefix, and stops the job if the DB is unreachable or if `TEST_DB_PREFIX`
   doesn't end in `_` and contain `test` (teardown drops every schema starting
   with the prefix, so a broad one could delete schemas that aren't yours).
4. Runs pytest.
5. When the job ends, restores `aeon/dj_pipeline/streams.py`. The tests
   regenerate this file and only restore it on a clean teardown, so a cancelled
   job would otherwise leave it modified.

Requested resources (CPUs, memory, walltime) are in the `#SBATCH` header of the
script. Output goes to `slurm_output/aeon-tests_<jobid>.out` and `.err`.

## Running subsets interactively

Use this for debugging or for running part of the suite. The full suite is long
and I/O heavy, so prefer the batch job for that.

### 1. Request a compute node

Run on a compute node rather than on the gateway. Match the resources to what
you're running:

```
srun --cpus-per-task=8 --time=04:00:00 --mem=16G --pty bash -i
```

### 2. Install dependencies and set the environment

From the root of your clone:

```
module load uv
uv sync --extra spike_sorting --group test-golden
```

Both flags matter. `--extra spike_sorting` pulls in spikeinterface/probeinterface
(without it the ephys tests skip and some unit tests hard-fail). `--group
test-golden` pulls in `swc-aeon-rigs-foragingabc`, needed by the golden fixtures.

Set the following environment variables:

| Variable | Purpose |
|---|---|
| `TEST_DB_PREFIX` | Schema prefix for your test schemas. Must end in `_` and contain `test`; make it unique to you so you don't collide with anyone else's schemas on the shared server, e.g. `u_${USER}_golden_tests_`. |
| `DJ_REPOSITORY_CONFIG` | A DataJoint config whose `repository_config` maps `ceph_aeon` to the repository root (as in prod; data is under `<root>/aeon/data/raw/...`). |

```
export TEST_DB_PREFIX=u_${USER}_golden_tests_
export DJ_REPOSITORY_CONFIG='{"ceph_aeon": "/ceph/aeon"}'
```

> [!NOTE]
> Without `DJ_REPOSITORY_CONFIG`, the tests use `~/sciops-data/project_aeon` as
> the data root (see `DEFAULT_GOLDEN_DATA_ROOT` in `tests/dj_pipeline/conftest.py`),
> expecting data under `<root>/aeon/data/raw/...`. Alternatively, stage a local
> copy there (rsync from Ceph — see the staging pattern in `docs/specs/SPEC_TESTING.md`).

### 3. Sanity-check before the long run

Confirm the database connection and namespace are what you expect. This actually
connects, so it catches host/password (and, for aeondj, cryptography) problems
before you commit to a long run:

```
uv run python -c "import os, datajoint as dj; dj.conn(); print('host:', dj.config.database.host); print('user:', dj.config.database.user); print('prefix:', os.environ.get('TEST_DB_PREFIX'))"
```

### 4. Run the tests

Unit tests (no database or golden data needed):

```
uv run pytest -m unit -v --tb=short -ra 2>&1 | tee unit_test_output.txt
```

All integration tests, including the `specialized` ones (the specialized tests are
auto-marked `integration`). A mix of synthetic-data and golden-dataset tests:

```
uv run pytest -m integration -v --tb=short -ra 2>&1 | tee integration_test_output.txt
```

Only the `specialized` tier (curation and unit matching), which share one expensive fixture build:

```
uv run pytest tests/dj_pipeline/test_ephys_curation_integration.py tests/dj_pipeline/test_unit_matching_integration.py -v --tb=short -ra 2>&1 | tee matching_test_output.txt
```

All integration tests, excluding the `specialized` tier:

```
uv run pytest -m "integration and not specialized" -v --tb=short -ra 2>&1 | tee integration_no_specialized_output.txt
```

Integration tests for the ephys ingestion module (golden dataset):

```
uv run pytest tests/dj_pipeline/test_ephys_ingestion.py -v --tb=short -ra 2>&1 | tee golden_test_output.txt
```

## Reading the output

- **Skipped** tests mean a dependency or the golden data wasn't found. A run
  that's mostly skips is NOT a pass — recheck
  [Install dependencies](#2-install-dependencies-and-set-the-environment) (the extra/group) and
  [Golden data](#golden-data) (the data root).
- **Failed** tests are a real signal; the `--tb=short` traceback shows where. The
  ones most worth watching for ephys/zarr changes are `test_recording_zarr_exists`
  (PreProcessing's recording output) and `test_sorting_analyzer_created`
  (PostProcessing's analyzer output).

## Dropping stale schemas

When `TEST_DB_PREFIX` is set (using an **external DB**), the `streams_schema` fixture
automatically drops any schemas created under your prefix during session teardown. After a
normal test run, whether it passes or fails, no manual cleanup should be required.

You only need to remove stale schemas manually if the process exits before teardown can run,
for example due to an OOM kill, `scancel`, or node failure. Before dropping any schemas,
**verify that the namespace belongs to you**:

```
uv run python -c "import os, datajoint as dj; pref=os.environ['TEST_DB_PREFIX']; print('host:', dj.config.database.host, 'prefix:', pref); [print('would drop:', s) for s in dj.list_schemas() if s.startswith(pref)]"
```

Then drop them:

```
uv run python -c "import os, datajoint as dj; dj.config.safemode=False; pref=os.environ['TEST_DB_PREFIX']; assert pref.endswith('_') and 'test' in pref; todo=[s for s in dj.list_schemas() if s.startswith(pref)]; exec('for _ in range(len(todo)+1):\n rest=[]\n for s in todo:\n  try: dj.Schema(s).drop()\n  except Exception: rest.append(s)\n todo=rest'); print('left:', todo)"
```

Schemas reference each other. The above command retries until the dependency order resolves.
