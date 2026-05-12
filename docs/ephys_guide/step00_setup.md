# Step 0 -- Setup and Prerequisites

Before running any ephys pipeline code, you need three things: access to the
SWC HPC, a working Python environment, and database credentials. This page
walks through each one.

---

## Prerequisites

- **SSH access to the SWC HPC.** If you don't have an account, ask your PI or
  the SWC IT team.
- **Database credentials** for `aeon-db`. Ask the pipeline team for a username
  and password.
- **Basic familiarity with DataJoint.** The guide explains each pipeline step,
  but it assumes you know what tables, schemas, and `populate()` calls are. If
  you're new to DataJoint, start with the tutorials at
  <https://docs.datajoint.org/>.
- **Basic familiarity with the terminal** (SSH, running commands, editing
  files).

---

## HPC Access

### Connecting to the gateway

SSH into the SWC HPC gateway:

```bash
ssh aeon-hpc
```

This lands you on `hpc-gw2`, one of the gateway nodes. The gateway is fine for
editing files and submitting jobs, but **Ceph (the shared filesystem where raw
data lives) is not visible from the gateway.** To access data, you need a
compute node.

### Getting a compute node

Request an interactive compute node with SLURM:

```bash
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p cpu --time=04:00:00 --mem=16G --pty bash -i
```

What the flags mean:

| Flag | Purpose |
|------|---------|
| `--nodes=1` | One compute node |
| `--ntasks-per-node=1` | One task (interactive shell) |
| `--cpus-per-task=8` | 8 CPU cores -- enough for most pipeline work |
| `-p cpu` | Use the CPU partition (no GPU needed for setup) |
| `--time=04:00:00` | 4-hour time limit -- the session ends automatically when this expires |
| `--mem=16G` | 16 GB RAM -- sufficient for data exploration and most populate calls |
| `--pty bash -i` | Start an interactive bash shell |

Once the node is allocated, you'll have access to `/ceph/aeon` and can run
pipeline code directly.

### About the `slurm_get_node_energy` errors

When you connect to a compute node, you may see errors like this:

```
slurm_get_node_energy: Can't get energy data. Check if the acct_gather_energy plugin is configured.
```

**These are cosmetic.** They come from SLURM's energy accounting plugin failing
to communicate -- it has nothing to do with your session. The shell is working
normally even though the error text may print over your prompt. Just press
**Enter** to get a clean prompt. Do **not** press Ctrl+C, as that will kill
your session.

### Session timeouts

Your compute node session ends automatically when the `--time` allocation
expires (4 hours in the example above). Save your work before that happens. You
can request a longer session by increasing `--time`, but be considerate of
shared cluster resources.

---

## Environment Setup

The HPC uses environment modules to manage software. Load `uv` (the Python
package manager used by this project):

```bash
module load uv
```

Then verify your environment by importing the `aeon` package:

```bash
uv run python -c "import aeon; print(aeon.__version__)"
```

If this prints a version number (e.g., `0.4.0`), your environment is ready.

**Important:** Always use `uv run python` instead of bare `python`. The `uv
run` prefix ensures you're using the project's virtual environment with the
correct dependencies, regardless of what other Python installations exist on
the system.

---

## Database Configuration

The pipeline uses DataJoint to connect to a MariaDB database. DataJoint 2.x
reads configuration from two sources in the repository root:

- **`datajoint.json`** -- database host, prefix, and store definitions
- **`.secrets/`** -- credentials (username and password), kept out of git

### Creating the config files

From the repo root on a compute node, generate both files in one command:

```bash
uv run python -c "import datajoint as dj; dj.config.save_template()"
```

This creates a `datajoint.json` template and a `.secrets/` directory with
placeholder credential files. The `.secrets/` directory is automatically
gitignored.

### Editing `datajoint.json`

Open the generated file and set these values:

```json
{
  "database.host": "aeon-db",
  "database.database_prefix": "u_<yourname>_aeon_ephys_v2_test_",
  "stores": {
    "dj_store": {
      "protocol": "file",
      "location": "/ceph/aeon/datajoint_stores"
    }
  }
}
```

What each field does:

- `database.host` -- Use `aeon-db`. This is the primary database server. Do
  **not** use `aeon-db2`, which only hosts old historical data from a previous
  experiment.
- `database.database_prefix` -- Prepended to every schema name the pipeline
  creates. Using a test prefix like `u_yourname_aeon_ephys_v2_test_` keeps your
  test schemas completely separate from production schemas (which use the
  `aeon_` prefix). You can drop your test schemas without affecting anyone else.
- `stores.dj_store` -- The ephys pipeline stores spike data and sorting output
  files in this location on Ceph. This entry is required for spike sorting
  tables to work.

### Setting credentials

Edit the files in the `.secrets/` directory:

```bash
echo "your_username" > .secrets/database.user
echo "your_password" > .secrets/database.password
```

DataJoint automatically reads these files at import time. Never commit the
`.secrets/` directory to git (the generated `.gitignore` inside it prevents
this).

### Verifying the connection

After setting up the config and credentials, verify the connection:

```bash
uv run python -c "import datajoint as dj; dj.conn()"
```

You should see a message confirming a successful connection to `aeon-db`.

---

## Data Location

Raw experimental data lives on Ceph, the shared filesystem. The directory
structure follows a consistent pattern:

```
/ceph/aeon/aeon/data/raw/<ARENA>/<EXPERIMENT_TAG>/
```

For the golden baseline dataset used in this guide:

```
/ceph/aeon/aeon/data/raw/AEONX1/abcEphysPilot02/
```

### Directory structure

Inside each experiment directory, data is organized into **epochs**. Each epoch
represents a continuous recording period and is named with an ISO 8601
timestamp:

```
abcEphysPilot02/
  2026-05-05T15-15-51/          <-- epoch directory
    NeuropixelsV2/              <-- device directory (ephys data)
      ProbeB/                   <-- probe directory
        ...                     <-- binary data files (10-minute chunks)
    recording_configurations/   <-- channel mapping files
    ...                         <-- other device directories
  2026-05-06T...                <-- next epoch
  ...
```

- **Epoch directories** are named with the timestamp when recording started.
  Each one contains all data acquired during that continuous recording period.
- **Device directories** (like `NeuropixelsV2/`) contain the raw binary data
  from each hardware device.
- **Probe directories** (like `ProbeB/`) sit inside the device directory, one
  per probe. The golden dataset has only ProbeB (ProbeA was disabled).
- **Data files** within each probe directory are split into chunks (typically
  10 minutes each in this dataset).

---

## Local Setup (On-Site at SWC)

If you are physically at SWC and have Ceph mounted on your local machine, you
can run the pipeline locally without the HPC. Ask Adrian for help with the
local Ceph mount configuration.

This guide assumes HPC access. The commands and paths are the same either way
-- only the connection method differs.

---

## Next Steps

Once you have a compute node session, your environment loaded, your
`datajoint.json` and `.secrets/` in place, and you've verified your database
connection, you're ready to
start processing data. Continue to
[Step 1: Register Experiment](step01_register_experiment.py).
