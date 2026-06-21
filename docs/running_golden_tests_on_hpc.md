# Running the ephys golden-dataset tests on the SWC HPC

How to run the ephys golden integration suite (`tests/dj_pipeline/test_ephys_ingestion.py`)
against the staged golden dataset on the SWC HPC. The suite force-injects the
pre-computed sorting, so it needs **no GPU** — a CPU node is enough and a full
run takes ~35 minutes (the time is PreProcessing reading the amplifier data off
Ceph).

## Prerequisites

- HPC access as your SWC user (`ssh aeon-hpc`).
- The golden data is already staged in shared home at
  `~/sciops-data/project_aeon/aeon/data/` (this is the default
  `DEFAULT_GOLDEN_DATA_ROOT`, so the tests find it automatically). You do not
  need to stage anything.
- A database account on `aeondj` (or `aeon-db`). You only need a `datajoint.json`
  plus a `.secrets/` directory; the steps below copy a known-good pair.

## 1. Get a compute node

Ceph is only visible from compute nodes, not the gateway, so grab a CPU node
first. Give it a generous walltime so the run isn't cut off:

```
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p cpu --time=04:00:00 --mem=16G --pty bash -i
```

## 2. Check out the code and install dependencies

Use your own checkout directory so you don't collide with other sessions:

```
module load uv
cd <your_aeon_mecha_checkout>
uv sync --extra spike_sorting --group test-golden
```

Both flags matter. `--extra spike_sorting` pulls in spikeinterface/probeinterface
(without it the ephys tests skip and some unit tests hard-fail). `--group
test-golden` pulls in `swc-aeon-rigs-foragingabc`, needed by the golden fixtures.

If your DB is `aeondj` (MySQL), also install the auth dependency it needs:

```
uv pip install cryptography
```

## 3. Point at the database

The integration tests need an external database (there is no Docker on the HPC).
Setting `TEST_DB_PREFIX` tells the test harness to skip testcontainers and use
your `datajoint.json` + `.secrets/` instead. Copy a known-good config:

```
cp ~/ProjectAeon/foragingABC_analysis/datajoint.json .
cp -r ~/ProjectAeon/foragingABC_analysis/.secrets .
```

Confirm `datajoint.json` points at the server you want (`aeon-db` or `aeondj`).
To switch it to aeondj, set the host and your aeondj password:

```
uv run python -c "import json; d=json.load(open('datajoint.json')); d['database']['host']='aeondj'; json.dump(d,open('datajoint.json','w'),indent=2)"
printf '%s' 'YOUR_AEONDJ_PASSWORD' > .secrets/database.password
```

Then set a **distinct** schema prefix. It must end in `_`, and it should be
unique to you so you don't collide with anyone else's test schemas on the shared
server:

```
export TEST_DB_PREFIX=u_${USER}_golden_tests_
```

## 4. Sanity-check before the long run

Confirm the connection and namespace are what you expect. This actually connects,
so it's your early check that the host, password, and (for aeondj) cryptography
are all good before committing to a 35-minute run:

```
uv run python -c "import os, datajoint as dj; dj.conn(); print('host:', dj.config.database.host); print('user:', dj.config.database.user); print('prefix:', os.environ.get('TEST_DB_PREFIX'))"
```

Confirm the golden data is visible from this node:

```
ls ~/sciops-data/project_aeon/aeon/data/raw/AEONX1/abcGolden01/2026-05-11T07-50-11/
```

## 5. Run the tests

```
uv run pytest -m integration tests/dj_pipeline/test_ephys_ingestion.py -v --tb=short -ra 2>&1 | tee golden_test_output.txt
```

## 6. Reading the output

- **Expected result: `29 passed`.** That is the golden baseline for this module.
- A test that **skips** rather than runs usually means a dependency or the golden
  data isn't found. If the whole module skips, recheck step 2 (the extra/group)
  and step 4 (the `ls`).
- A test that **fails** is a real signal — the `--tb=short` traceback tells you
  where. The ones most worth watching for ephys/zarr changes are
  `test_recording_zarr_exists` (PreProcessing's recording output) and
  `test_sorting_analyzer_created` (PostProcessing's analyzer output).

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
