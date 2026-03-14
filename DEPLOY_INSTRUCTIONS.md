# Production Ephys v2 Deployment — HPC Instructions

## Branch Layout

```
dj_pipeline_ephys (main ephys branch)
  └─ v0.2.0 tag (tested template — do not modify)
       └─ es/prod-social-ephys0.1 (this branch — production deployment scripts)
```

## Files Added on This Branch

| File | Location | Purpose |
|------|----------|---------|
| `prod_ephys_deploy.py` | `aeon/dj_pipeline/scripts/` | Main deployment script (28 steps, 5 phases) |
| `prod_spike_sorting.py` | `aeon/dj_pipeline/scripts/` | SLURM worker for spike sorting |
| `prod_spike_sorting.sh` | repo root | SLURM submission wrapper |
| `prod_dj_local_conf.json` | repo root | Template DB config for aeon-db2 |
| `DEPLOY_INSTRUCTIONS.md` | repo root | This file |

---

## Workspace Setup on HPC

```bash
# Connect to HPC
ssh sgw2
ssh aeon-hpc

# Clone the production branch directly
cd ~/ProjectAeon
git clone --branch es/prod-social-ephys0.1 https://github.com/SainsburyWellcomeCentre/aeon_mecha.git aeon_mecha_prod_ephys
cd aeon_mecha_prod_ephys

# Install dependencies
module load uv
uv sync

# Create dj_local_conf.json from template
cp prod_dj_local_conf.json dj_local_conf.json
# EDIT dj_local_conf.json: replace REPLACE_WITH_PASSWORD with actual password

# Verify connection
uv run python -c "import datajoint as dj; print(dj.config['database.host'], dj.config['custom']['database.prefix'])"
# Should print: aeon-db2 aeon_
```

---

## Phase 0: Reconnaissance (read-only)

Run this FIRST to discover subjects, probes, and time ranges before any destructive action.

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv

# Run all recon steps
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --phase 0
```

This will show:
- Current production table contents and row counts
- Available epoch time range
- Probe labels (ProbeA, ProbeB, etc.)
- Subject information from the DB

**After recon, edit `aeon/dj_pipeline/scripts/prod_ephys_deploy.py` constants:**
- `SUBJECT` — real subject name
- `BLOCK_START` — first block start time
- `PROBE_LABELS` — single or dual probe
- `N_BLOCKS` — number of blocks for your time window

---

## Phase 1: Drop Old Tables (DESTRUCTIVE)

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv

# Preview first
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --dry-run --phase 1

# Execute (DataJoint will prompt for confirmation on each drop)
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --phase 1
```

Each `drop()` call will ask you to type "yes" — this is the safety net.

---

## Phase 2: Ingestion

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv

# Preview
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --dry-run --phase 2

# Execute
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --phase 2
```

This creates: Lookup entries, Experiment, ProbeInsertions, EphysEpochs, EphysChunks, EphysBlocks, EphysBlockInfo.

Note: Step 15 (ingest_chunks) may take a while for 294 epochs.

---

## Phase 3: Spike Sorting Setup + PreProcessing

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv

# Execute setup steps (electrode groups, params, tasks)
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --phase 3
```

This creates SortingTasks and runs PreProcessing. Step 22 will show you the pending SLURM jobs.

---

## Phase 3b: SpikeSorting via SLURM

**Submit ONE block at a time** (30h blocks are very large).

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys

# Edit aeon/dj_pipeline/scripts/prod_spike_sorting.py:
#   1. Set correct subject name
#   2. Set block_start and block_end for this job
#   3. Set insertion_number (1=ProbeA, 2=ProbeB)

# Submit
sbatch prod_spike_sorting.sh

# Monitor
squeue -u $USER
tail -f slurm_output/*.out
```

**DO NOT change any code while SLURM jobs are queued!**

For dual-probe setup, you need separate jobs for each probe:
- insertion_number=1 (ProbeA): 7 SLURM jobs
- insertion_number=2 (ProbeB): 7 SLURM jobs
- Total: 14 SLURM jobs

Wait for ALL spike sorting jobs to complete before Phase 4.

---

## Phase 4: Post-Sorting

After all SLURM jobs are done:

```bash
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv

# Execute all post-sorting steps
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --phase 4
```

This runs: PostProcessing, SortedSpikes, Waveform, SortingQuality, auto-curation, SyncedSpikes, and UnitMatching.

---

## Spot Checks

After everything completes, do manual verification:

```python
# Quick check script
import datajoint as dj
from aeon.dj_pipeline import ephys, spike_sorting, spike_sorting_curation as curation

exp = "social-ephys0.1-aeon3"

print("=== Row Counts ===")
print(f"EphysEpoch:       {len(ephys.EphysEpoch & {'experiment_name': exp})}")
print(f"EphysChunk:       {len(ephys.EphysChunk & {'experiment_name': exp})}")
print(f"EphysBlock:       {len(ephys.EphysBlock & {'experiment_name': exp})}")
print(f"SortingTask:      {len(spike_sorting.SortingTask & {'experiment_name': exp})}")
print(f"SpikeSorting:     {len(spike_sorting.SpikeSorting & {'experiment_name': exp})}")
print(f"SortedSpikes:     {len(spike_sorting.SortedSpikes & {'experiment_name': exp})}")
print(f"SyncedSpikes:     {len(spike_sorting.SyncedSpikes & {'experiment_name': exp})}")
print(f"GlobalUnit:       {len(spike_sorting.GlobalUnit & {'experiment_name': exp})}")
print(f"UnitMatching:     {len(spike_sorting.UnitMatching & {'experiment_name': exp})}")

# Fetch some spike times to verify datetime64[ns]
import numpy as np
spikes = (spike_sorting.SyncedSpikes.Unit & {'experiment_name': exp}).fetch(
    'synced_spike_times', limit=1
)[0]
print(f"\nSpike times dtype: {spikes.dtype}")
print(f"Sample: {spikes[:5]}")
```

---

## Reconnect Preamble

If you get disconnected from HPC:

```bash
ssh sgw2
ssh aeon-hpc
cd ~/ProjectAeon/aeon_mecha_prod_ephys
module load uv
# Continue from where you left off, e.g.:
uv run python -m aeon.dj_pipeline.scripts.prod_ephys_deploy --step 17
```

---

## If Something Goes Wrong

- **Wrong DB / prefix**: The script checks `aeon-db2` + `aeon_` prefix before every phase
- **Drop failed**: Run individual steps with `--step N` to retry
- **Sorting failed**: Check SLURM logs in `slurm_output/`, clear error jobs, resubmit
- **Can always re-clone**: Raw data on ceph is untouched; only DB metadata changes
- **Ceph sorting data**: Previous sorting results at `/ceph/aeon/aeon/dj_store/ephys-processed/` are preserved
