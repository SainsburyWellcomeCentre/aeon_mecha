# SLURM Pipeline Automation Specifications

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Orchestrator](#orchestrator)
5. [Workers](#workers)
6. [Error Tracking and History](#error-tracking-and-history)
7. [CLI Interface](#cli-interface)
8. [Packaging and Installation](#packaging-and-installation)
9. [Example: Aeon Pipeline Configuration](#example-aeon-pipeline-configuration)
10. [Open Questions for Discussion](#open-questions-for-discussion)

---

## Overview

### Purpose

DataJoint pipelines on HPC clusters currently require manual intervention to process data: a human must SSH into the cluster, write or edit scripts, and run `sbatch` or `populate()` by hand. This works for one-off jobs but does not scale to continuous, multi-table pipelines where new data arrives regularly and processing should happen automatically.

This spec describes a general-purpose package for automating DataJoint pipeline processing on SLURM-managed HPC clusters. The package handles the scheduling, job submission, resource allocation, and error tracking needed to turn a manual populate-and-check workflow into a hands-off automated system that can be monitored remotely.

The package is not specific to any one pipeline. It is designed so that any DataJoint project running on a SLURM cluster can adopt it by writing a configuration file that describes their tables and resource requirements. Project Aeon at the Sainsbury Wellcome Centre is the first intended user, but the design is general.

### What This Package Does

- Periodically submits SLURM jobs that call `populate()` on configured DataJoint tables
- Assigns different SLURM resource allocations (CPU, memory, GPU, wall time) to different groups of tables based on their computational requirements
- Tracks orchestrator runs and maintains a historical log of populate errors, complementing DataJoint's built-in per-table jobs system
- Provides a CLI for starting, stopping, and monitoring the automation

### What This Package Does Not Do

- No web UI or dashboard (that is the domain of DataJoint Works)
- No pipeline table definitions — those belong to the project's own codebase
- No data processing logic — the package calls `populate()` and lets DataJoint handle the rest
- No dependency orchestration between tables — DataJoint's `populate()` is idempotent and will simply return if upstream data is not ready, so the system relies on running frequently enough for data to cascade through the pipeline over successive cycles

### Design Principles

**Leverage DataJoint's existing infrastructure.** DataJoint 2.x's `populate()` method already handles auto-detection of pending work, job reservation via per-table `~~table_name` jobs tables, transactional execution, and SIGTERM-based graceful shutdown. This package does not replicate any of that. It is a thin scheduling and submission layer on top of what DataJoint already provides.

**Keep it simple.** The orchestrator's job is to submit SLURM jobs and log what happened. The workers' job is to call `populate()`. There is no complex state machine, no dependency graph resolution, and no inter-job communication. The system is easy to reason about: a scheduled job fires, it submits workers, the workers do their thing, errors get logged.

**Be general, not Aeon-specific.** Every project-specific detail (table names, resource requirements, schedule interval) lives in the configuration file, not in the package code. The package should work for any DataJoint pipeline on any SLURM cluster without modification.

---

## Architecture

The system has three layers:

### Configuration Layer

A YAML configuration file, maintained by the project team, that declares:
- What DataJoint tables exist in the pipeline
- What SLURM resources each group of tables needs
- How often the orchestrator should run
- What schema name to use for the package's own tracking tables

This is the only project-specific artifact. Everything else is provided by the package.

### Orchestrator Layer

A lightweight process that reads the configuration, checks what work is pending (for per-key tiers), and submits SLURM jobs for the workers. The orchestrator itself runs as a minimal SLURM job on a CPU partition. It typically completes in under a minute — its only job is to query DataJoint and call `sbatch`.

### Worker Layer

The SLURM jobs that do the actual `populate()` calls. Workers come in two modes:
- **Batch workers** handle a list of lightweight tables in a single SLURM job, calling `populate()` on each table in sequence.
- **Per-key workers** handle a single pending key for a single table, each in its own dedicated SLURM job with heavier resources.

Workers use DataJoint's `reserve_jobs=True` flag, which means multiple workers can safely run concurrently without processing the same data twice.

```
┌─────────────────────────────────────────────────────┐
│                    SLURM Cluster                     │
│                                                      │
│  ┌──────────────┐                                    │
│  │ Orchestrator  │  (CPU, 2 cores, 4GB, 30min)       │
│  │              │                                    │
│  │  Reads config │                                    │
│  │  Queries DJ   │                                    │
│  │  Submits jobs │                                    │
│  └──────┬───────┘                                    │
│         │                                            │
│         ├──── sbatch ──── Batch Worker (light tier)   │
│         │                 populates TableA, B, C, ... │
│         │                 (CPU, 2 cores, 8GB, 4hr)    │
│         │                                            │
│         ├──── sbatch ──── Per-Key Worker (medium tier) │
│         │                 populates 1 key per job     │
│         │                 (CPU, 8 cores, 64GB, 8hr)   │
│         │                                            │
│         └──── sbatch ──── Per-Key Worker (heavy tier)  │
│                           populates 1 key per job     │
│                           (GPU, 8 cores, 256GB, 7d)   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

The configuration file is a YAML file that lives in the project's repository (e.g., `slurm_worker_config.yaml`). It defines the schedule, the resource tiers, and the table assignments.

### Configuration Format

```yaml
# How often the orchestrator fires.
schedule:
  interval_hours: 3

# Schema name for the package's tracking tables.
# Combined with DataJoint's database_prefix setting.
# e.g., if database_prefix = "aeon_", this becomes "aeon_slurm_worker"
schema_name: slurm_worker

# Named resource tiers. Each tier defines SLURM resource
# requirements and a populate mode.
tiers:

  light:
    slurm:
      partition: cpu
      cpus: 2
      memory: 8G
      time: "4:00:00"
      # gpu is omitted — no GPU for this tier

    # "batch" mode: one SLURM job runs populate() on every
    # table in this tier, in the order listed. If a table has
    # nothing to populate, populate() returns immediately.
    mode: batch

    # Optional: limit how many keys each table processes per
    # cycle. Prevents the batch worker from exceeding its wall
    # time when there is a large backlog. Omit for no limit.
    max_calls: null

    # List tables in dependency order (upstream first).
    # This ordering is a recommendation — populate() is
    # idempotent, so out-of-order execution is safe, just
    # potentially less efficient.
    tables:
      - module.path.TableA
      - module.path.TableB
      - module.path.TableC

  medium:
    slurm:
      partition: cpu
      cpus: 8
      memory: 64G
      time: "8:00:00"

    # "per-key" mode: the orchestrator calls
    # TableClass.jobs.refresh() to find pending keys and
    # submits one SLURM job per key.
    mode: per-key

    # Orphan timeout (seconds): if a reserved job has been
    # running longer than this, consider it orphaned (worker
    # died) and re-add as pending. Should be slightly longer
    # than the wall time for this tier.
    orphan_timeout: 32400   # 9 hours (wall time is 8hr)

    tables:
      - module.path.TableD
      - module.path.TableE

  heavy:
    slurm:
      partition: gpu
      cpus: 8
      memory: 256G
      time: "7-08:00:00"
      gpu: "a100:1"

    mode: per-key
    orphan_timeout: 691200  # 8 days (wall time is 7d 8hr)

    tables:
      - module.path.TableF
```

### Configuration Details

**Tiers** are user-defined and named. The package imposes no limit on the number of tiers or their names. Each tier has a SLURM resource spec and a populate mode.

**Populate modes:**
- `batch` — One SLURM job is submitted for the entire tier. The worker calls `table.populate(reserve_jobs=True)` on each table in the listed order. This is appropriate for tables where individual rows take seconds to minutes to process. An optional `max_calls` parameter can limit how many keys a single batch worker processes per table, preventing the worker from running longer than its wall time when there is a large backlog.
- `per-key` — The orchestrator calls `TableClass.jobs.refresh()` and reads `TableClass.jobs.pending` to find keys that are ready for processing. It submits one SLURM job per pending key. This is appropriate for tables where individual rows take hours or days to process.

**Table references** use dotted Python module paths (e.g., `aeon.dj_pipeline.spike_sorting.SpikeSorting`). The package imports these at runtime to resolve the actual DataJoint table classes.

**SLURM resource specs** map directly to `#SBATCH` directives. The `gpu` field is optional — omit it for CPU-only tiers. For site-specific SLURM directives not covered by the standard fields (e.g., `--account`, `--qos`, `--constraint`), the `slurm` block also supports an `extra_args` list that is passed through to sbatch as-is.

**Worker logging:** Generated sbatch scripts configure `--output` and `--error` directives pointing to a `slurm_output/` directory with the job ID in the filename. This ensures that SLURM-level failures (import errors, environment issues, OOM kills) that never reach the `WorkerErrorHistory` table are still captured in log files on disk.

**Schedule interval** controls how often the orchestrator fires. Shorter intervals mean data cascades through the pipeline faster (new upstream rows become available to downstream tables sooner). A 3-hour interval is reasonable for most pipelines; adjust based on data arrival frequency and processing times.

---

## Orchestrator

The orchestrator is the central coordinator. It runs as a lightweight SLURM job and its only responsibility is to submit worker jobs.

### What the Orchestrator Does Each Cycle

1. Load the configuration file
2. For each tier, in the order they appear in the config:
   - If the tier's mode is `batch`: submit one SLURM job with the tier's resource spec. The job will call `populate(reserve_jobs=True)` on each table in the tier's list, in order.
   - If the tier's mode is `per-key`: for each table in the tier, call `TableClass.jobs.refresh(orphan_timeout=N)` (using the tier's configured `orphan_timeout`) to update the jobs queue and recover any orphaned keys, then read `TableClass.jobs.pending` to find keys that are ready for processing (this excludes keys that are already reserved, errored, or ignored). For each pending key, submit a SLURM job with the tier's resource spec. The job will call `populate(key, reserve_jobs=True)` for that specific key.
3. Log a summary to the `OrchestratorRunHistory` table: timestamp, number of jobs submitted per tier, any errors encountered during submission.
4. Schedule the next run (mechanism depends on chosen scheduling approach — see [Open Questions](#open-questions-for-discussion)).
5. Exit.

**Note on per-key job submission:** Using `TableClass.jobs.refresh(orphan_timeout=N)` followed by `TableClass.jobs.pending` is important for two reasons: (1) `refresh()` discovers new pending keys and recovers orphaned reserved jobs, and (2) `pending` returns only keys that are genuinely ready for processing -- it excludes keys that are already reserved by another worker, errored, or marked as ignored. Without this check, successive orchestrator cycles could submit duplicate SLURM jobs for keys that are already being processed, wasting queue priority and potentially billing.

### Robustness

The orchestrator wraps all operations in error handling so that a failure in one tier does not prevent other tiers from being processed. If submitting a job for one key fails, the orchestrator logs the error and continues with the remaining keys. The orchestrator itself should never crash — individual job submission failures are logged, not propagated.

### Resource Requirements

The orchestrator needs minimal resources. It only runs Python code to query DataJoint and call `sbatch` via subprocess. A reasonable default allocation: 2 CPU cores, 4GB memory, 30-minute wall time on a CPU partition.

---

## Workers

Workers are the SLURM jobs that do the actual data processing by calling DataJoint's `populate()`.

### Batch Workers

A batch worker receives a list of table module paths and calls `populate(reserve_jobs=True)` on each one in sequence. This is the worker type used for `batch` mode tiers.

```python
# Pseudocode for a batch worker
for table_path in config["tiers"]["light"]["tables"]:
    table_class = import_table(table_path)
    max_calls = config["tiers"]["light"].get("max_calls")
    result = table_class.populate(
        reserve_jobs=True,
        suppress_errors=True,
        max_calls=max_calls,
    )
    # result = {"success_count": int, "error_list": [...]}
    # The package's worker wrapper inspects error_list and
    # logs each error to WorkerErrorHistory
```

Two flags are important here. The `reserve_jobs=True` flag enables DataJoint's distributed job system: each key is atomically reserved in the `~~jobs` table before processing, and on failure the error message and stack trace are recorded there (preventing future retries until manually cleared). The `suppress_errors=True` flag tells DataJoint to catch exceptions during `make()` and continue to the next key instead of aborting the entire batch. Together, these ensure that (a) errors are tracked per key and (b) one failure does not block the rest of the batch. The `populate()` return value includes an `error_list` containing `(key, error_message)` tuples for any failures. The worker wrapper inspects this list and appends each error to `WorkerErrorHistory` with the full context (SLURM job ID, tier, timestamp).

The worker processes all tables and all available keys in a single run.

### Per-Key Workers

A per-key worker receives a single table module path and a single primary key dict. It calls `populate(key, reserve_jobs=True)` for that specific key. This is the worker type used for `per-key` mode tiers.

Per-key workers are appropriate for computationally expensive tables (e.g., spike sorting) where each row may take hours or days and requires dedicated resources (GPU, large memory).

### Environment Setup

All workers need access to:
- The project's Python environment (e.g., managed by `uv`, `conda`, or `venv`)
- DataJoint configuration (database credentials, connection settings)
- The project's codebase (so that table module paths can be imported)

The package generates sbatch scripts at runtime based on the tier configuration. These scripts handle module loading, environment activation, and invocation of the worker entry point. Site-specific setup (e.g., `module load` commands, environment variables like `PYTORCH_CUDA_ALLOC_CONF` for GPU tiers) is handled via the `extra_args` field or the generated script template (see [Open Question 5](#5-additional-slurm-configuration)).

---

## Error Tracking and History

### DataJoint 2.x's Built-in Jobs System

In DataJoint 2.x, each auto-populated table (Computed/Imported) has its own dedicated jobs table, named with the pattern `~~table_name` (e.g., `~~spike_sorting` for the SpikeSorting table). These jobs tables track the full lifecycle of each populate key:

- `pending` — key has been identified as needing processing (via `jobs.refresh()`)
- `reserved` — a worker has claimed this key and is processing it
- `success` — processing completed successfully (optionally kept or deleted, depending on config)
- `error` — processing failed; error message and stack trace are stored
- `ignore` — key has been manually marked to be skipped permanently

When `populate(reserve_jobs=True)` is called, it attempts to atomically reserve a pending key. If successful, it runs `make()`. On success, the job is either marked `success` or deleted. On failure, the job is marked `error` with the error details. Keys with `error` status are skipped by future `populate()` calls until manually cleared.

The `jobs.refresh()` method updates the jobs queue by computing `(key_source - target - existing_jobs)` and inserting new entries as `pending`. It also handles orphaned reserved jobs (workers that died without completing) via a configurable `orphan_timeout`.

**What the built-in system provides well:** Per-key reservation, concurrent worker safety, error storage, orphan recovery, and the `ignore` status for permanently skipping problematic keys.

**What the built-in system does not provide:** When an error entry is cleared (to allow retry), the error information is lost. There is no history of past failures across retries. There is also no record of when populate runs happened or what SLURM jobs were involved.

### WorkerErrorHistory Table

This package adds a `WorkerErrorHistory` table that maintains a permanent, append-only log of all populate errors. Every time a `populate()` call fails, the worker wrapper appends a row to this table containing:

| Column | Type | Description |
|--------|------|-------------|
| `error_id` | int (auto) | Unique identifier for this error occurrence |
| `table_name` | varchar | Full module path of the DataJoint table that failed |
| `key` | blob | The full primary key dict (serialized) |
| `error_message` | varchar | The error message |
| `error_stack` | blob | The full Python stack trace |
| `slurm_job_id` | varchar | The SLURM job ID of the worker that encountered the error |
| `tier` | varchar | Which tier this worker belongs to |
| `timestamp` | timestamp | When the error occurred |

This table is append-only — the system never deletes entries. The `key` column stores the full primary key dict as a serialized blob, which allows queries like "show me every error this key has ever hit" or "show me all errors for this table in the past week." Because different pipeline tables have different primary key schemas, the key is stored as a generic serialized dict rather than typed columns.

The table lives in its own DataJoint schema (name configured in the YAML, e.g., `aeon_slurm_worker`).

### OrchestratorRunHistory Table

A lightweight table that logs each orchestrator cycle:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | int (auto) | Unique identifier for this orchestrator run |
| `timestamp` | timestamp | When the orchestrator started |
| `config_hash` | char(32) | Hash of the config file used (to detect config changes) |
| `jobs_submitted` | blob | Summary dict: tier name → number of jobs submitted |
| `status` | enum | `"completed"` or `"error"` |
| `error_message` | varchar | If the orchestrator itself encountered an error |
| `duration_seconds` | float | How long the orchestrator took to run |

This provides visibility into whether the automation is actually running and what it is doing each cycle. At a 3-hour interval, this table accumulates roughly 2,900 rows per year.

### Error Recovery Workflow

1. Check for errors: `dj-slurm errors` (CLI command) or query `WorkerErrorHistory` directly
2. Investigate: review error messages and stack traces
3. Fix the underlying issue (code bug, missing data, permissions, etc.)
4. Clear the error: `dj-slurm clear-errors --table module.path.TableName --key '{"col": "value"}'` (removes the error entry from the table's `~~jobs` table and re-adds the key as `pending`)
5. Next orchestrator cycle: the key is picked up as pending and retried automatically

The error history in `WorkerErrorHistory` is preserved regardless of whether the error is cleared from the jobs table.

### SLURM Wall-Time Kills and Orphaned Jobs

When SLURM kills a worker that exceeds its wall-time limit, it sends SIGTERM followed (after a grace period) by SIGKILL. DataJoint 2.x installs a SIGTERM handler during `populate(reserve_jobs=True)` that raises `SystemExit`, allowing in-flight transactions to roll back. The error handler then attempts to mark the key as `error` in the `~~jobs` table, but if SIGKILL arrives before that database write completes, the key remains in `reserved` status.

DataJoint 2.x's `jobs.refresh(orphan_timeout=N)` handles this: reserved jobs older than N seconds are automatically deleted and re-added as `pending`. The orchestrator calls `refresh()` each cycle, so orphaned keys from killed workers are recovered automatically on the next cycle.

The `orphan_timeout` should be set per tier based on expected processing time. For example, a heavy tier with a 7-day wall time should use an orphan timeout slightly longer than 7 days, so that legitimately long-running jobs are not prematurely recovered.

---

## CLI Interface

The package provides a command-line tool (`dj-slurm` or similar name — to be decided) with the following commands:

### `dj-slurm start`

Submit the orchestrator to SLURM and begin the recurring automation cycle. Takes the path to the configuration file as an argument.

```bash
dj-slurm start --config slurm_worker_config.yaml
```

### `dj-slurm stop`

Stop the recurring automation. Cancels the pending next-run orchestrator job. Does not cancel currently running workers.

```bash
dj-slurm stop
```

### `dj-slurm run`

Run a single orchestrator cycle immediately (without scheduling the next one). Useful for testing and manual triggering.

```bash
dj-slurm run --config slurm_worker_config.yaml
dj-slurm run --config slurm_worker_config.yaml --dry-run  # show what would be submitted without actually calling sbatch
```

The `--dry-run` flag prints what SLURM jobs would be submitted (tier, table, key, resource spec) without actually submitting them. This is valuable for testing new configurations.

### `dj-slurm status`

Show the current state of the automation: whether the orchestrator is scheduled, what workers are currently running (via `squeue`), and the most recent orchestrator run summary.

```bash
dj-slurm status
```

### `dj-slurm errors`

Query the `WorkerErrorHistory` table. Supports filtering by table name, key, date range, and tier.

```bash
dj-slurm errors                              # show all recent errors
dj-slurm errors --table SpikeSorting          # errors for a specific table
dj-slurm errors --since 2026-05-01            # errors since a date
```

### `dj-slurm clear-errors`

Clear error entries from a table's `~~jobs` table to allow retrying failed keys. The key is re-added as `pending` so it will be picked up by the next orchestrator cycle. Does not delete from `WorkerErrorHistory` (that history is permanent).

The `--table` argument uses the full module path (as listed in the config file), or the short class name if it is unambiguous within the loaded config.

```bash
dj-slurm clear-errors --table aeon.dj_pipeline.spike_sorting.SpikeSorting --key '{"block_start": "2024-06-04 11:00:00"}'
dj-slurm clear-errors --table SpikeSorting --all   # short name OK if unambiguous in config
```

---

## Packaging and Installation

### Package Structure

The package is a standalone Python package with its own GitHub repository (repository location to be decided). It is installed as a dependency of the project that uses it.

```
datajoint-slurm/
├── pyproject.toml
├── README.md
├── src/
│   └── datajoint_slurm/
│       ├── __init__.py
│       ├── config.py          # YAML config loading and validation
│       ├── orchestrator.py    # Orchestrator logic
│       ├── worker.py          # Worker populate wrapper
│       ├── tables.py          # WorkerErrorHistory, OrchestratorRunHistory
│       ├── slurm.py           # sbatch generation and submission
│       └── cli.py             # CLI entry points
└── tests/
```

### Installation

```bash
pip install datajoint-slurm
```

Or in a project's `pyproject.toml`:

```toml
dependencies = [
    "datajoint>=2.0",
    "datajoint-slurm",
]
```

### Dependencies

- `datajoint>=2.0` — for table definitions, populate, and database access
- `pyyaml` — for configuration file parsing
- `click` or `argparse` — for CLI (to be decided)
- No other external dependencies. SLURM interaction is via `subprocess` calls to `sbatch`, `squeue`, and `scancel`, which are available on any SLURM cluster.

### Adoption by a New Project

1. Install the package (`pip install datajoint-slurm`)
2. Create a configuration YAML file listing tiers and tables
3. Ensure DataJoint is configured with database credentials and `database_prefix`
4. Run `dj-slurm start --config config.yaml` on the HPC

The package creates its database schema and tracking tables automatically on first run.

---

## Example: Aeon Pipeline Configuration

This section illustrates what the configuration would look like for the Aeon project at SWC. This is a first pass, not a final assignment — the actual tier assignments need to be validated through a discovery phase where we time how long each table's `populate()` takes for epochs of various lengths. Some tables currently listed in the light tier (e.g., `SortedSpikes`, `Waveform`, `SortingQuality`, `SyncedSpikes`, `UnitMatching`) may need to be promoted to the medium tier if they turn out to take more than a few minutes per key. This benchmarking should happen early in implementation.

```yaml
schedule:
  interval_hours: 3

schema_name: slurm_worker

tiers:
  light:
    slurm:
      partition: cpu
      cpus: 2
      memory: 8G
      time: "4:00:00"
    mode: batch
    tables:
      # Subject tables (upstream — populate first)
      - aeon.dj_pipeline.subject.CreatePyratIngestionTask
      - aeon.dj_pipeline.subject.PyratIngestion
      - aeon.dj_pipeline.subject.PyratCommentWeightProcedure
      - aeon.dj_pipeline.subject.SubjectDetail
      - aeon.dj_pipeline.subject.SubjectWeight
      - aeon.dj_pipeline.subject.SubjectProcedure
      - aeon.dj_pipeline.subject.SubjectComment
      # Acquisition tables
      - aeon.dj_pipeline.acquisition.EpochConfig
      # Ephys discovery
      - aeon.dj_pipeline.ephys.EphysEpoch
      - aeon.dj_pipeline.ephys.EphysBlockInfo
      # Tracking
      - aeon.dj_pipeline.tracking.SLEAPTracking
      # QC
      - aeon.dj_pipeline.qc.CameraQC
      # Stream tables (dynamically generated — exact paths TBD)
      # Spike sorting post-populate (lightweight reads)
      - aeon.dj_pipeline.spike_sorting.SortedSpikes
      - aeon.dj_pipeline.spike_sorting.Waveform
      - aeon.dj_pipeline.spike_sorting.SortingQuality
      - aeon.dj_pipeline.spike_sorting.SyncedSpikes
      - aeon.dj_pipeline.spike_sorting.UnitMatching

  medium:
    slurm:
      partition: cpu
      cpus: 8
      memory: 64G
      time: "8:00:00"
      extra_args:
        - "--export=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    mode: per-key
    orphan_timeout: 32400   # 9 hours
    tables:
      - aeon.dj_pipeline.spike_sorting.PreProcessing
      - aeon.dj_pipeline.spike_sorting.PostProcessing

  heavy:
    slurm:
      partition: gpu
      cpus: 8
      memory: 256G
      time: "7-08:00:00"
      gpu: "a100:1"
      extra_args:
        - "--export=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
    mode: per-key
    orphan_timeout: 691200  # 8 days
    tables:
      - aeon.dj_pipeline.spike_sorting.SpikeSorting
```

### How Data Flows Through This Configuration

Consider a new experimental epoch arriving in the system:

- **Cycle 1 (hour 0):** The light batch worker populates upstream tables — `EpochConfig` discovers the new epoch, `EphysEpoch` discovers the associated ephys data, `EphysBlockInfo` characterizes the blocks. The medium and heavy tiers have no pending keys yet.

- **Cycle 2 (hour 3):** The light batch worker has no new work for acquisition tables (already populated). The orchestrator finds pending keys for `PreProcessing` (the new blocks discovered in cycle 1) and submits medium-tier per-key jobs for them. If any PreProcessing jobs complete quickly, `SpikeSorting` keys may become pending — the orchestrator also checks the heavy tier and submits GPU jobs if needed.

- **Cycle 3 (hour 6):** PreProcessing jobs from cycle 2 have finished. The orchestrator submits heavy-tier GPU jobs for `SpikeSorting` (if not already submitted in cycle 2). Spike sorting begins.

- **Cycles 4-50+ (hours 9-150+):** Spike sorting runs for days. Each orchestrator cycle checks for completed sorting and submits `PostProcessing` medium-tier jobs as keys become available. The light batch worker picks up `SortedSpikes`, `Waveform`, `SortingQuality`, `SyncedSpikes`, and `UnitMatching` as their upstream data becomes available.

The pipeline processes data end-to-end without manual intervention. The 3-hour interval means it takes a few cycles for data to cascade from the top to the bottom of the pipeline, but this is acceptable for a pipeline where the bottleneck step (spike sorting) takes days.

---

## Open Questions for Discussion

### 1. Where Does the Scheduler Live?

The orchestrator needs to fire on a recurring schedule. There are two viable approaches, and the choice depends on what infrastructure is available and what the HPC operations team prefers.

#### Option A: Self-Resubmitting SLURM Job

The orchestrator is a SLURM job. At the end of each run, after submitting all workers, it submits itself again with a delayed start time:

```bash
sbatch --begin=now+3hours orchestrator.sh
```

This creates a perpetual chain: each orchestrator run schedules the next one.

**Starting:** `dj-slurm start` submits the first orchestrator job to SLURM.

**Stopping:** `dj-slurm stop` cancels the pending next-run job via `scancel`.

**If the chain breaks** (SLURM cluster outage, orchestrator crashes before resubmitting): the automation stops. Someone needs to notice and run `dj-slurm start` again. The `OrchestratorRunHistory` table makes this visible — if there is no recent run logged, the chain has broken.

**Advantages:**
- Fully self-contained within SLURM — no external dependencies or accounts needed
- Anyone with `sbatch` access can start and stop the automation
- Portable to any site with a SLURM cluster — no site-specific setup beyond SLURM access
- No IT/sysadmin involvement required

**Disadvantages:**
- If the chain breaks, automation stops until someone manually restarts it
- HPC administrators at some sites may have policies against self-resubmitting jobs
- Changing the schedule interval requires stopping and restarting the automation

#### Option B: Cron Job on a Shared Service Account

A system crontab entry on the HPC login node runs `dj-slurm run` on a schedule. The cron job runs under a shared service account (not a personal user account).

```
# Crontab entry: run every 3 hours
0 */3 * * * /path/to/dj-slurm run --config /path/to/config.yaml
```

**Starting:** IT team adds the crontab entry (one-time setup).

**Stopping:** IT team removes or comments out the crontab entry.

**If the login node reboots:** Cron resumes automatically — crontab entries persist across reboots.

**Advantages:**
- Standard Linux scheduling mechanism — well-understood and battle-tested
- Survives reboots automatically
- Easy to change the schedule (edit one line in the crontab)
- Clear separation of concerns: cron handles scheduling, SLURM handles compute

**Disadvantages:**
- Requires a shared service account to be created by the IT/HPC operations team
- Requires IT involvement for initial setup and any schedule changes
- Less portable — service account setup is site-specific, and some HPC environments restrict user access to cron
- Adds a dependency outside of SLURM that needs to be maintained

#### Recommendation

Both approaches work. Option A is simpler to deploy and more portable. Option B is more robust against unexpected failures. The choice may depend on what the SWC HPC operations team already has in place and what they are comfortable supporting.

A hybrid approach is also possible: start with Option A for quick deployment, and migrate to Option B if the site operations team offers to set up a service account.

### 2. Tracking Tables: Visibility and Schema Placement

This spec proposes two tables (`WorkerErrorHistory` and `OrchestratorRunHistory`) in a single dedicated DataJoint schema (e.g., `aeon_slurm_worker`) — one set of tables for the entire pipeline, not one per schema. These tables are user-facing: they are meant to be queried directly when investigating errors or monitoring the system.

An alternative would be to make these hidden tables (using DataJoint's `~` prefix convention). Hidden tables do not appear in `schema.list_tables()` or in DataJoint diagrams, but are fully queryable.

The question is: **should the tracking tables be visible or hidden?** Visible tables in their own schema are easier to discover and query directly. Hidden tables are less intrusive but require knowing they exist. Since the primary use case is active monitoring and debugging, visibility seems preferable — but this is worth confirming.

### 3. Package Repository Location

The package needs its own GitHub repository. Options include:
- Under the DataJoint company organization (e.g., `datajoint-company/datajoint-slurm`)
- Under the DataJoint open-source organization (e.g., `datajoint/datajoint-slurm`)
- Under the SWC organization alongside aeon_mecha (if it should be more tightly coupled to Aeon initially)

The repository location also affects the package name on PyPI and how it is referenced in project dependencies.

### 4. Package and CLI Naming

Working names used in this spec:
- Package: `datajoint-slurm` (import as `datajoint_slurm`)
- CLI: `dj-slurm`

These may need to change to align with DataJoint's naming conventions or to avoid confusion with other packages.

### 5. Additional SLURM Configuration

The current spec maps tier resource specs directly to `#SBATCH` directives and supports an `extra_args` list per tier for site-specific directives (e.g., `--account`, `--qos`, `--constraint`, `--export`). This covers most SLURM customization needs.

However, some sites may also need pre-`populate()` shell commands that are not `#SBATCH` directives, such as:
- Module loading commands (e.g., `module load uv`, `module load cuda`)
- Environment setup scripts (e.g., `source /path/to/env.sh`)

These are currently handled by the generated sbatch script templates. The question is whether this should be configurable per tier (e.g., a `setup_commands` list in the YAML) or handled by a site-specific sbatch template that the package provides a hook for.

### 6. Database Schema Creation Privileges

The package creates its own DataJoint schema (e.g., `aeon_slurm_worker`) on first run. This requires `CREATE DATABASE` privileges on the database server. On shared DB servers (like SWC's aeon-db), this may require a one-time setup by a database administrator to pre-create the schema and grant the appropriate user permissions. The implementation should handle this gracefully — detect whether the schema exists and provide clear error messages if creation fails due to insufficient privileges.

### 7. Concurrent Orchestrator Protection

If two orchestrator runs overlap (e.g., a delayed SLURM start causes a cycle to still be running when the next fires), they could submit duplicate worker jobs. For per-key tiers this is handled by `reserve_jobs=True` (only one worker processes each key), so the duplicates would exit harmlessly. For batch tiers, overlapping populate calls on the same table are also safe due to job reservation. However, the wasted SLURM submissions are undesirable. The implementation should consider a lightweight lock mechanism (e.g., checking if an orchestrator SLURM job is already running via `squeue` before submitting workers).

### 8. DataJoint Version Compatibility

This spec is written against DataJoint 2.x, which introduced per-table `~~table_name` jobs tables, the `jobs.refresh()` API, and status values like `pending`, `reserved`, `success`, `error`, and `ignore`. However, Aeon's production HPC currently runs DataJoint 0.14.x, which uses a shared `~jobs` table per schema with `key_hash`-based lookups and a different `populate()` return value format.

The package needs to work for its first user. Options include:
- **Require DJ 2.x** — simplest to implement, but means Aeon would need to upgrade their production HPC environment before adopting the package
- **Support both versions** — a compatibility layer that abstracts the differences in the jobs API between 0.14.x and 2.x, allowing the same package to work on either
- **Separate branches** — main branch targets DJ 2.x, with a maintenance branch for DJ 0.14.x

**Recommendation:** Separate branches. The main branch targets DJ 2.x (the future of all DataJoint projects). A `maint/0.14.x` branch supports DJ 0.14.x for Aeon's current production environment. Tagged releases from each branch allow projects to pin the version they need in their `pyproject.toml`. As Aeon migrates to DJ 2.x, they switch to the main branch releases. This keeps the main codebase clean (no compatibility shims) while supporting the first user immediately.

### 9. Notification System (Stretch Goal)

A future enhancement could add email or Slack notifications when errors occur or when the orchestrator chain breaks. This is out of scope for the initial implementation but should be kept in mind as a possible extension point during design.
