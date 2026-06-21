# Running the ephys golden-dataset tests on the SWC HPC

How to run the ephys golden integration suite
(`tests/dj_pipeline/test_ephys_ingestion.py`) against the golden dataset on the
SWC HPC. The suite force-injects the pre-computed sorting, so it needs **no
GPU** — a CPU node is enough, and a full run takes ~35 minutes (most of that is
PreProcessing reading the amplifier data off Ceph).

## Prerequisites

- HPC access (`ssh aeon-hpc`).
- A database account on `aeon-db` or `aeondj` (either works — pick one). If you
  don't have one yet, set one up via the team's DB onboarding or ask Thinh/Elissa.
- Access to the golden data, which lives on Ceph (see step 4).

## 1. Get a compute node

The suite is heavy (~35 minutes, lots of Ceph I/O), so run it on a compute node
rather than on the gateway. Grab a CPU node with a generous walltime:

```
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p cpu --time=04:00:00 --mem=16G --pty bash -i
```

## 2. Check out the code and install dependencies

Use your own checkout directory:

```
module load uv
cd <your_aeon_mecha_checkout>
uv sync --extra spike_sorting --group test-golden
```

Both flags matter. `--extra spike_sorting` pulls in spikeinterface/probeinterface
(without it the ephys tests skip and some unit tests hard-fail). `--group
test-golden` pulls in `swc-aeon-rigs-foragingabc`, needed by the golden fixtures.

If you connect to **aeondj** (MySQL), also install the auth dependency it needs:

```
uv pip install cryptography
```

(Not needed for `aeon-db`.)

## 3. Point at your database

The integration tests need an external database (there is no Docker on the HPC).
Setting `TEST_DB_PREFIX` tells the test harness to skip testcontainers and use a
`datajoint.json` + `.secrets/` in your checkout directory. Create them with your
own credentials.

`datajoint.json` (set `host` to whichever server you're using):

```json
{
  "database": {
    "host": "aeon-db",
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

Then set a **distinct** schema prefix. It must end in `_`, and it should be
unique to you so you don't collide with anyone else's test schemas on the shared
server:

```
export TEST_DB_PREFIX=u_${USER}_golden_tests_
```

(The test fixtures override the `dj_store` to a per-session temp dir, so the
`stores` entry above is just to make the config valid — the tests don't write to
the production store.)

## 4. Make sure the golden data is reachable

By default the tests look for the data under
`~/sciops-data/project_aeon/aeon/data` (per-user, see
`DEFAULT_GOLDEN_DATA_ROOT` in `tests/dj_pipeline/conftest.py`). The source of
truth lives on Ceph, so you have two options:

- **Point at the Ceph copy** by setting the `DJ_REPOSITORY_CONFIG` env var to a
  DataJoint config whose `repository_config` maps `ceph_aeon` to the golden-data
  root, or
- **Stage a local copy** into `~/sciops-data/...` (rsync from Ceph — see the
  staging pattern in `docs/specs/SPEC_TESTING.md`).

Either way, confirm the ephys golden epoch and its required files are visible at
whichever root you use before running. The dataset's `experiment_path`,
`epoch_dir`, and `required_files` are listed in the `GOLDEN_DATASETS` registry in
`tests/dj_pipeline/conftest.py` (the ephys entry is
`foraging_abc_ephys_2026_05_11`). A quick `ls` of the epoch directory is a good
check.

Important: if the data isn't found at the configured root, `_check_golden_data`
calls `pytest.skip(...)` rather than failing — so a run full of **skips** means
the data root isn't set right, not that everything passed.

## 5. Sanity-check before the long run

Confirm the database connection and namespace are what you expect. This actually
connects, so it catches host/password (and, for aeondj, cryptography) problems
before you commit to a 35-minute run:

```
uv run python -c "import os, datajoint as dj; dj.conn(); print('host:', dj.config.database.host); print('user:', dj.config.database.user); print('prefix:', os.environ.get('TEST_DB_PREFIX'))"
```

## 6. Run the tests

```
uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py -v --tb=short -ra 2>&1 | tee golden_test_output.txt
```

## 7. Reading the output

- **Expected result: `29 passed`.** That is the golden baseline for this module.
- **Skipped** tests mean a dependency or the golden data wasn't found. A run
  that's mostly skips is NOT a pass — recheck step 2 (the extra/group) and step 4
  (the data root).
- **Failed** tests are a real signal; the `--tb=short` traceback shows where. The
  ones most worth watching for ephys/zarr changes are `test_recording_zarr_exists`
  (PreProcessing's recording output) and `test_sorting_analyzer_created`
  (PostProcessing's analyzer output).

## Re-running

The integration fixtures don't drop their schemas on an external DB, so stale
schemas from a previous run can cause confusing failures. Before re-running,
drop your test schemas. **Check the namespace first** so you only ever drop your
own prefix:

```
uv run python -c "import os, datajoint as dj; pref=os.environ['TEST_DB_PREFIX']; print('host:', dj.config.database.host, 'prefix:', pref); [print('would drop:', s) for s in dj.list_schemas() if s.startswith(pref)]"
```

Then drop them:

```
uv run python -c "import os, datajoint as dj; dj.config.safemode=False; pref=os.environ['TEST_DB_PREFIX']; assert pref.endswith('_') and 'test' in pref; [dj.Schema(s).drop() for s in dj.list_schemas() if s.startswith(pref)]; print('dropped')"
```
