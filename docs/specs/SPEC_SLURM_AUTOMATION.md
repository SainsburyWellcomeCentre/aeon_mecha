# SLURM Pipeline Automation Specifications

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Orchestrator](#orchestrator)
5. [Worker Layer](#worker-layer)
6. [Error Handling and Recovery](#error-handling-and-recovery)
7. [Scheduling and Deployment](#scheduling-and-deployment)
8. [Entry Points](#entry-points)
9. [Code Layout](#code-layout)
10. [Example: Aeon Pipeline Configuration](#example-aeon-pipeline-configuration)
11. [Open Questions for Discussion](#open-questions-for-discussion)

---

## Overview

### Purpose

DataJoint pipelines on HPC clusters currently require manual intervention to process data: a human must SSH into the cluster, write or edit scripts, and run `sbatch` or `populate()` by hand. This works for one-off jobs but does not scale to continuous, multi-table pipelines where new data arrives regularly and processing should happen automatically.

This spec describes a configuration-driven system for automating DataJoint pipeline processing on SLURM-managed HPC clusters. It handles the scheduling, job submission, and resource allocation needed to turn a manual populate-and-check workflow into a hands-off automated system that can be monitored remotely.

The design is config-driven: the tables to process, the resources each group needs, and how often to run are all declared in a configuration file, with no project-specific logic in the orchestrator itself. Project Aeon at the Sainsbury Wellcome Centre is the first user.

### What This System Does

- Periodically submits SLURM jobs that call `populate()` on configured tables
- Assigns different SLURM resource allocations (CPU, memory, GPU, wall time) to different groups of tables based on their computational requirements
- Computes how many parallel workers each group needs from the count of pending work, capped per group
- Logs each orchestrator run for monitoring

### What This System Does Not Do

- No web UI or dashboard (that is the domain of DataJoint Works)
- No pipeline table definitions — those belong to the project's own codebase
- No data processing logic — the system calls `populate()` and lets DataJoint handle the rest
- No dependency orchestration between tables — DataJoint's `populate()` is idempotent and simply returns if upstream data is not ready, so the system relies on running frequently enough for data to cascade through the pipeline over successive cycles
- No key assignment logic — DataJoint's built-in `reserve_jobs=True` mechanism handles distributed key assignment. The orchestrator only determines what SLURM resources to allocate and how many worker instances to submit.

### Design Principles

**Leverage DataJoint's built-in features.** DataJoint 2.x provides per-key reservation (`reserve_jobs=True`), orphan recovery (`jobs.refresh(orphan_timeout=...)`), and error tracking via per-table `~~jobs` tables. The system uses all of these directly.

**Keep the orchestrator simple and uniform.** The orchestrator's only job is to count pending work and submit SLURM jobs. It treats every worker type the same way — there is no special-casing, no dependency-graph resolution, and no inter-job communication. All behavioral differences between worker types come from configuration, not from orchestrator logic.

**Push behavior into configuration.** Every project-specific detail — table names, resource requirements, schedule interval, how aggressively each group processes work — lives in the configuration file. Changing behavior means editing the config, not the code.

---

## Architecture

The system has three layers:

### Configuration Layer

A YAML configuration file, maintained by the project team, that declares:
- The named worker types in the pipeline
- The SLURM resources each worker type needs
- The tables each worker type processes
- How many parallel workers to run for each worker type
- How often the orchestrator should run

This is the only project-specific artifact.

### Orchestrator Layer

A lightweight process that reads the configuration, checks what work is pending, and submits SLURM jobs. It runs as a minimal CPU job and typically completes in well under a minute — its only work is to query DataJoint and call `sbatch`. It applies the same logic to every worker type.

### Worker Layer

The SLURM jobs that do the actual `populate()` calls. Each worker is launched for a single worker type and processes that worker type's tables, configured with that type's settings. Every worker instance maps to exactly one SLURM job.

```
┌─────────────────────────────────────────────────────┐
│                    SLURM Cluster                     │
│                                                      │
│  ┌──────────────┐                                    │
│  │ Orchestrator  │  (CPU, 2 cores, 4GB, 30min)       │
│  │              │                                    │
│  │  Reads config │                                    │
│  │  Counts work  │                                    │
│  │  Submits N    │                                    │
│  └──────┬───────┘                                    │
│         │                                            │
│         ├──── sbatch ──── Worker: light × N            │
│         │                 (CPU, 2 cores, 8GB, 4hr)    │
│         │                                            │
│         ├──── sbatch ──── Worker: medium × N           │
│         │                 (CPU, 8 cores, 64GB, 8hr)   │
│         │                                            │
│         └──── sbatch ──── Worker: heavy × N            │
│                           (GPU, 8 cores, 256GB, 7d)   │
│                                                      │
│  N is computed per worker type from pending work,    │
│  capped at that type's max_concurrent.               │
└─────────────────────────────────────────────────────┘
```

---

## Configuration

The configuration file is a YAML file that lives in the project's repository. It defines the schedule, the worker types, and their table assignments.

### Configuration Format

```yaml
# How often the orchestrator fires.
schedule:
  interval_hours: 3

# Schema name for the orchestrator's run-history table.
# Combined with DataJoint's database_prefix setting.
# e.g., if database_prefix = "aeon_", this becomes "aeon_slurm_worker"
schema_name: slurm_worker

# Named worker types. Each worker type defines SLURM resource
# requirements, worker loop settings, scaling, and table assignments.
worker_types:

  light:
    slurm:
      partition: cpu
      cpus: 2
      memory: 8G
      time: "4:00:00"
      # gpu is omitted — no GPU for this worker type

    # Loop settings for workers of this type.
    worker:
      sleep_duration: 60         # seconds between cycles
      run_duration: 13800        # 3h50m — exit before 4h wall time

    # Maximum number of parallel SLURM jobs for this worker type.
    max_concurrent: 1

    # Tables this worker type processes, in dependency order
    # (upstream first). Ordering is a recommendation — populate()
    # is idempotent, so out-of-order execution is safe, just less
    # efficient. max_calls is optional, per table.
    tables:
      - path: module.path.TableA
      - path: module.path.TableB
      - path: module.path.TableC

  medium:
    slurm:
      partition: cpu
      cpus: 8
      memory: 64G
      time: "8:00:00"

    worker:
      sleep_duration: 30
      run_duration: 28800        # 8hr
      stale_timeout_hours: 9     # recover orphans after 9hr

    # Orchestrator computes needed instances from pending keys
    # and per-table max_calls, capped at this limit.
    max_concurrent: 10

    tables:
      - path: module.path.TableD
        max_calls: 1
      - path: module.path.TableE
        max_calls: 10

  heavy:
    slurm:
      partition: gpu
      cpus: 8
      memory: 256G
      time: "7-08:00:00"
      gpu: "a100:1"

    worker:
      sleep_duration: 30
      run_duration: 604800       # 7 days
      stale_timeout_hours: 200   # >7 days

    # Capped to avoid monopolizing shared GPU resources.
    max_concurrent: 2

    tables:
      - path: module.path.TableF
        max_calls: 1
```

### Configuration Details

**Worker types** are user-defined and named. There is no limit on the number of worker types or their names. Each worker type has a SLURM resource spec, worker loop settings, a scaling cap, and a list of tables.

**Worker loop settings** describe how a worker of this type runs:
- `sleep_duration` — seconds between populate cycles
- `run_duration` — maximum seconds the worker runs before exiting gracefully. Set slightly below the SLURM wall time (e.g., 10 minutes less) so the worker finishes its current cycle and exits cleanly before SLURM sends SIGTERM/SIGKILL.
- `stale_timeout_hours` — hours before a reserved job is considered orphaned and recovered via `jobs.refresh(orphan_timeout=...)`. Should be slightly longer than the wall time for this worker type.

**Tables** are listed per worker type as a `path` (dotted Python module path, e.g. `aeon.dj_pipeline.spike_sorting.SpikeSorting`) plus an optional **`max_calls`**. `max_calls` limits how many keys that table processes per cycle and is passed to `populate()` as a keyword argument. Different tables in the same worker type can have different `max_calls` values; omit it for no limit. The module path is imported at runtime to resolve the actual DataJoint table class.

**Scaling** (`max_concurrent`) caps how many parallel SLURM jobs the orchestrator submits for a worker type. The orchestrator computes the number of instances needed from pending work (see [Orchestrator](#orchestrator)) and never exceeds this cap.

**SLURM resource specs** map directly to `#SBATCH` directives. The `gpu` field is optional — omit it for CPU-only worker types. For site-specific SLURM directives not covered by the standard fields (e.g., `--account`, `--qos`, `--constraint`, `--export`), the `slurm` block also supports an `extra_args` list that is passed through to sbatch as-is.

**Worker logging:** Generated sbatch scripts configure `--output` and `--error` directives pointing to a `slurm_output/` directory with the job ID in the filename. This ensures that SLURM-level failures (import errors, environment issues, OOM kills) are captured in log files on disk.

**Schedule interval** controls how often the orchestrator fires. Shorter intervals mean data cascades through the pipeline faster (new upstream rows become available to downstream tables sooner). Because the orchestrator exits immediately when there is nothing to submit, a short interval is cheap. A 3-hour interval is reasonable for most pipelines; adjust based on data arrival frequency and processing times.

---

## Orchestrator

The orchestrator is the central coordinator. It runs as a lightweight job and its only responsibility is to determine how many workers each worker type needs and submit them. It applies the same algorithm to every worker type.

### What the Orchestrator Does Each Cycle

1. Load the configuration file.
2. For each worker type, in the order it appears in the config:
   - For each table in the worker type, call `table.jobs.refresh(orphan_timeout=stale_timeout_hours)` and count pending keys.
   - Compute the instances needed (see [Scaling Logic](#scaling-logic)).
   - Count workers of this type already running or queued (matched by job name via `squeue`).
   - Submit `to_submit = max(0, min(needed, max_concurrent) − already_running)` SLURM jobs, each running a worker for this type.
3. Log a summary to the `OrchestratorRunHistory` table: timestamp, number of jobs submitted per worker type, any errors encountered during submission, and duration.
4. Exit.

### Scaling Logic

For a worker type, the number of instances needed is computed from pending work:

- For each table, count pending keys.
- If the table has `max_calls`, the instances needed for that table is `ceil(pending / max_calls)`. If `max_calls` is not set, one worker can drain the table, so the count is 1 (if any pending) or 0.
- Each worker instance processes **all** tables in its worker type sequentially, so the worker type's needed count is the **maximum** across its tables — not the sum.
- The orchestrator then caps the result at `max_concurrent` and subtracts workers already running or queued.

For example, a medium worker type with Table D at 12 pending keys (`max_calls: 1`) and Table E at 8 pending keys (`max_calls: 10`) needs `max(ceil(12/1), ceil(8/10)) = max(12, 1) = 12` instances, capped at `max_concurrent: 10` → submit 10. If 3 are already running, submit 7.

This single formula covers every worker type. A worker type with `max_concurrent: 1` and no `max_calls` resolves to "submit one worker if any work is pending and none is running" without any special handling — the long-running, sweep-everything behavior comes entirely from that worker type's loop settings, not from the orchestrator.

Each instance uses `populate(reserve_jobs=True)` — DataJoint's atomic reservation distributes keys dynamically at runtime, so the orchestrator never pre-assigns keys. The pending count only determines how many SLURM jobs to submit.

**Note on pending key counting:** `table.jobs.refresh()` is called before reading the pending count for two reasons: (1) it discovers new pending keys by computing `key_source − target − jobs`, and (2) it recovers orphaned reserved jobs via `orphan_timeout`.

**Multiple submissions:** Each instance is an independent `sbatch` call. If pending work changes between submission and execution (SLURM queue delays can be significant), the dynamic reservation system handles it gracefully: workers that find no work simply exit.

### Robustness

The orchestrator wraps all operations in error handling so that a failure in one worker type does not prevent the others from being processed. If counting pending keys for one table fails, the orchestrator logs the error and continues. The orchestrator itself should never crash — individual failures are logged, not propagated.

### Resource Requirements

The orchestrator needs minimal resources: it only queries DataJoint and calls `sbatch` via subprocess. A reasonable default: 2 CPU cores, 4GB memory, 30-minute wall time on a CPU partition.

### OrchestratorRunHistory Table

A lightweight table that logs each orchestrator cycle:

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | int (auto) | Unique identifier for this orchestrator run |
| `timestamp` | timestamp | When the orchestrator started |
| `config_hash` | char(32) | Hash of the config file used (to detect config changes) |
| `jobs_submitted` | blob | Summary dict: worker type → number of jobs submitted |
| `status` | enum | `"completed"` or `"error"` |
| `error_message` | varchar | If the orchestrator itself encountered an error |
| `duration_seconds` | float | How long the orchestrator took to run |

This provides visibility into whether the automation is actually running and what it is doing each cycle. If no recent run is logged, the schedule has stopped and someone should restart it. The table lives in its own DataJoint schema (name configured in the YAML, e.g., `aeon_slurm_worker`).

---

## Worker Layer

Each SLURM job submitted by the orchestrator runs a **worker** for one worker type. The worker repeatedly calls `populate()` on that worker type's tables until its work is done or its `run_duration` elapses, then exits. The worker reads the same per-worker-type configuration and presents the same interface to the orchestrator regardless of how it is implemented. There are two ways to provide the worker:

### Option 1: DataJoint Worker

Each SLURM job runs a DataJoint Worker configured for its worker type. It manages the populate loop, key reservation, scheduling, and graceful shutdown for the worker type's tables.

### Option 2: Lightweight populate loop

A minimal loop bundled with the orchestrator. For each table in the worker type, it calls `jobs.refresh(orphan_timeout=...)` and `populate(reserve_jobs=True, suppress_errors=True, max_calls=...)`, sleeping `sleep_duration` between cycles, and exits when `run_duration` is reached or SIGTERM is received:

```
install SIGTERM handler (set shutdown flag)
while not shutdown and elapsed < run_duration:
    for each table in the worker type:
        table.jobs.refresh(orphan_timeout=stale_timeout_hours)
        table.populate(reserve_jobs=True, suppress_errors=True, max_calls=max_calls)
    sleep(sleep_duration)
```

The orchestrator, configuration format, scaling logic, and SLURM integration are identical for both options; only the worker entry point differs.

### Worker Behavior Patterns

Because behavior is set by configuration, the same machinery produces different operating patterns:

- **Long-running sweep** (typical for lightweight tables): `max_concurrent: 1`, long `run_duration`, no `max_calls`. One SLURM job loops through all the worker type's tables, picking up new work each cycle.
- **Multi-key pass** (typical for medium-weight tables): `max_concurrent: N`, modest `max_calls`. Multiple independent SLURM jobs each process a bounded number of keys per table; DataJoint's reservation distributes keys among them.
- **Single-key** (typical for heavy/GPU tables): `max_concurrent: N` (capped low), `max_calls: 1`. Multiple independent SLURM jobs each grab one key, process it (possibly over days), then exit.

These are configuration outcomes, not modes enforced by the code.

### Environment Setup

All workers need access to:
- The project's Python environment (e.g., managed by `uv`, `conda`, or `venv`)
- DataJoint configuration (database credentials, connection settings)
- The project's codebase (so that table module paths can be imported)

The generated sbatch scripts handle environment activation and invocation of the worker entry point. Site-specific `#SBATCH` directives are passed via the `extra_args` field. Pre-`populate()` shell commands (e.g. `module load`) are covered in [Open Question 3](#3-additional-slurm-configuration).

---

## Error Handling and Recovery

### DataJoint 2.x's Built-in Jobs System

In DataJoint 2.x, each auto-populated table has its own dedicated jobs table, named `~~table_name` (e.g., `~~spike_sorting` for `SpikeSorting`). These track the lifecycle of each populate key:

- `pending` — key has been identified as needing processing (via `jobs.refresh()`)
- `reserved` — a worker has claimed this key and is processing it
- `success` — processing completed (optionally kept or deleted, depending on config)
- `error` — processing failed; error message and stack trace are stored
- `ignore` — key has been manually marked to be skipped permanently

When `populate(reserve_jobs=True)` is called, it atomically reserves a pending key, runs `make()`, and marks the result `success` (or deletes it) or `error`. Keys in `error` status are skipped by future `populate()` calls until cleared.

`jobs.refresh()` updates the queue by computing `(key_source − target − existing_jobs)` and inserting new entries as `pending`. It also recovers orphaned `reserved` jobs (workers that died without completing) via a configurable `orphan_timeout`.

This provides per-key reservation, concurrent worker safety, error storage, orphan recovery, and the `ignore` status for permanently skipping problematic keys.

### Error Recovery Workflow

1. Check for errors: query the table's `~~jobs` table for `error`-status rows.
2. Investigate the error messages and stack traces.
3. Fix the underlying issue (code bug, missing data, permissions, etc.).
4. Clear the error: remove the error entry from the table's `~~jobs` table, so it is re-inserted as `pending` on the next `refresh()`.
5. Next orchestrator cycle: the key is picked up as pending and retried.

### SLURM Wall-Time Kills and Orphaned Jobs

When SLURM kills a worker that exceeds its wall-time limit, it sends SIGTERM followed (after a grace period) by SIGKILL. DataJoint 2.x installs a SIGTERM handler during `populate(reserve_jobs=True)` that raises `SystemExit`, allowing in-flight transactions to roll back.

If SIGKILL arrives before a key can be marked `error`, it remains in `reserved` status. The `stale_timeout_hours` setting handles this: `jobs.refresh(orphan_timeout=...)` is called each cycle (by both workers and the orchestrator), automatically recovering reserved jobs older than the timeout. Set `stale_timeout_hours` per worker type based on expected processing time — e.g., a heavy worker type with a 7-day wall time should use roughly 200 hours so legitimately long jobs are not recovered prematurely.

---

## Scheduling and Deployment

The orchestrator fires on a recurring schedule via cron. Two deployment models are supported, and both can coexist:

### Per-user (initial approach)

The owning experimenter sets up a cron entry on the HPC gateway (or runs the orchestrator entry point manually):

```
# Run every 3 hours
0 */3 * * * /path/to/orchestrator.sh --config /path/to/config.yaml
```

All SLURM jobs are then submitted under that user's account, so cluster usage is attributed to the person whose data is being processed. This is the recommended starting point: each experimenter controls their own schedule and can trigger runs on demand.

### Centralized (later option)

The orchestrator runs as a cron on a dedicated machine, managing one or more projects from a single place. This requires a dedicated service account on the cluster so that submissions are attributed to that account rather than an individual. Suitable once the per-user approach is proven and a service account is available.

The orchestrator is idempotent and exits immediately when there is nothing to submit, so a short cron interval (down to a few minutes) is inexpensive regardless of deployment model.

---

## Entry Points

The orchestrator is invoked through a single clean entry point that reads a config file and runs one cycle:

```bash
orchestrator.sh --config config.yaml             # run one orchestrator cycle
orchestrator.sh --config config.yaml --dry-run   # print what would be submitted, submit nothing
```

The `--dry-run` flag prints the planned submissions per worker type (instance count, resource spec) without calling `sbatch` — useful for validating a new configuration.

A small status command surfaces the current state: what workers are running (via `squeue`), the most recent orchestrator run, and pending/error counts across configured tables (via `table.jobs.progress()`).

---

## Code Layout

The automation lives inside the project's own codebase (for Aeon, `aeon_mecha`), alongside the pipeline definitions rather than as a separately distributed package. This follows the existing DataJoint project convention of a populate/worker module sitting next to the pipeline module:

```
aeon/
├── dj_pipeline/          # pipeline table definitions (existing)
├── populate/             # worker-type definitions and worker entry point
└── slurm_orchestration/  # orchestrator, sbatch generation, config loading, entry point
```

A per-project YAML config file (carrying `schema_name`, the active `database_prefix`, the worker types, and their table assignments) drives the orchestrator. The orchestrator imports the project's pipeline to resolve table module paths and count pending keys.

### Dependencies

- `datajoint>=2.1.0` — for table definitions, populate, and the per-table `~~jobs` system
- `pyyaml` — for configuration file parsing
- SLURM interaction is via `subprocess` calls to `sbatch`, `squeue`, and `scancel`, available on any SLURM cluster

---

## Example: Aeon Pipeline Configuration

This illustrates the configuration for the Aeon project at SWC. This is a first pass, not a final assignment — the actual worker-type assignments need to be validated through a discovery phase that times how long each table's `populate()` takes for epochs of various lengths. Some tables currently in the light worker type (e.g., `SortedSpikes`, `Waveform`, `SortingQuality`, `SyncedSpikes`, `UnitMatching`) may need to move to medium if they turn out to take more than a few minutes per key. This benchmarking should happen early in implementation.

```yaml
schedule:
  interval_hours: 3

schema_name: slurm_worker

worker_types:
  light:
    slurm:
      partition: cpu
      cpus: 2
      memory: 8G
      time: "4:00:00"
    worker:
      sleep_duration: 60
      run_duration: 13800       # 3h50m — exit before 4h wall time
    max_concurrent: 1
    tables:
      # Subject tables (upstream — populate first)
      - path: aeon.dj_pipeline.subject.CreatePyratIngestionTask
      - path: aeon.dj_pipeline.subject.PyratIngestion
      - path: aeon.dj_pipeline.subject.PyratCommentWeightProcedure
      - path: aeon.dj_pipeline.subject.SubjectDetail
      - path: aeon.dj_pipeline.subject.SubjectWeight
      - path: aeon.dj_pipeline.subject.SubjectProcedure
      - path: aeon.dj_pipeline.subject.SubjectComment
      # Acquisition tables
      - path: aeon.dj_pipeline.acquisition.EpochConfig
      # Ephys discovery
      - path: aeon.dj_pipeline.ephys.EphysEpoch
      - path: aeon.dj_pipeline.ephys.EphysBlockInfo
      # Tracking
      - path: aeon.dj_pipeline.tracking.SLEAPTracking
      # QC
      - path: aeon.dj_pipeline.qc.CameraQC
      # Spike sorting post-processing (lightweight reads)
      - path: aeon.dj_pipeline.spike_sorting.SortedSpikes
      - path: aeon.dj_pipeline.spike_sorting.Waveform
      - path: aeon.dj_pipeline.spike_sorting.SortingQuality
      - path: aeon.dj_pipeline.spike_sorting.SyncedSpikes
      - path: aeon.dj_pipeline.spike_sorting.UnitMatching

  medium:
    slurm:
      partition: cpu
      cpus: 8
      memory: 64G
      time: "8:00:00"
    worker:
      sleep_duration: 30
      run_duration: 28800       # 8hr
      stale_timeout_hours: 9
    max_concurrent: 10
    tables:
      - path: aeon.dj_pipeline.spike_sorting.PreProcessing
        max_calls: 1
      - path: aeon.dj_pipeline.spike_sorting.PostProcessing
        max_calls: 1

  heavy:
    slurm:
      partition: gpu
      cpus: 8
      memory: 256G
      time: "7-08:00:00"
      gpu: "a100:1"
      extra_args:
        - "--export=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"
    worker:
      sleep_duration: 30
      run_duration: 604800      # 7 days
      stale_timeout_hours: 200
    max_concurrent: 2
    tables:
      - path: aeon.dj_pipeline.spike_sorting.SpikeSorting
        max_calls: 1
```

### How Data Flows Through This Configuration

Consider a new experimental epoch arriving in the system:

- **Cycle 1 (hour 0):** The light worker populates upstream tables — `EpochConfig` discovers the new epoch, `EphysEpoch` discovers the associated ephys data, `EphysBlockInfo` characterizes the blocks. The medium and heavy worker types have no pending keys yet, so the orchestrator submits zero instances for them.

- **Cycle 2 (hour 3):** The orchestrator finds pending keys for `PreProcessing` (the new blocks from cycle 1) and submits medium workers for them (up to 10). If any complete quickly, `SpikeSorting` keys become pending and the orchestrator submits GPU workers (up to 2).

- **Cycle 3 (hour 6):** PreProcessing has finished; the orchestrator submits heavy GPU workers for `SpikeSorting` if not already running. Spike sorting begins.

- **Cycles 4–50+ (hours 9–150+):** Spike sorting runs for days. Each cycle checks for completed sorting and submits `PostProcessing` medium workers as keys become available. The light worker picks up `SortedSpikes`, `Waveform`, `SortingQuality`, `SyncedSpikes`, and `UnitMatching` as their upstream data appears.

The pipeline processes data end-to-end without manual intervention. The 3-hour interval means data takes a few cycles to cascade from top to bottom, which is acceptable for a pipeline whose bottleneck step (spike sorting) takes days.

### GPU Resource Considerations

The SWC HPC cluster has 16 A100 GPUs (across 4 nodes) shared among all users. The `max_concurrent: 2` setting for the heavy worker type ensures the Aeon pipeline uses at most 2 A100s at any given time (~12.5% of the pool). Adjust based on cluster usage policies and other groups' needs. The cluster also has L40S, A4500, and Quadro RTX 5000 GPUs, but spike sorting is configured to use A100s specifically.

---

## Open Questions for Discussion

### 1. Worker Layer Implementation

The [Worker Layer](#worker-layer) can be provided either by the DataJoint Worker (Option 1) or by the bundled lightweight loop (Option 2). The orchestrator and configuration are identical either way. The choice determines only the worker entry point and is not yet finalized.

### 2. Database Schema Creation Privileges

The orchestrator creates its own DataJoint schema (e.g., `aeon_slurm_worker`) on first run, which requires `CREATE DATABASE` privileges. On shared DB servers (like SWC's aeon-db), this may require a one-time setup by a database administrator to pre-create the schema and grant permissions. The implementation should detect whether the schema exists and give a clear error if creation fails due to insufficient privileges.

### 3. Additional SLURM Configuration

Tier resource specs map to `#SBATCH` directives, with an `extra_args` list per worker type for site-specific directives (`--account`, `--qos`, `--constraint`, `--export`). Some sites may also need pre-`populate()` shell commands that are not `#SBATCH` directives:
- Module loading (e.g., `module load uv`, `module load cuda`)
- Environment setup scripts (e.g., `source /path/to/env.sh`)

These could be configured per worker type via a `setup_commands` list:

```yaml
heavy:
  slurm:
    partition: gpu
    # ...
  setup_commands:
    - "module load uv"
    - "module load cuda/12.1"
```

The question is whether this configurability is needed for the initial release, or whether a single site-provided setup script the worker calls is sufficient.

### 4. Concurrent Orchestrator Protection

If two orchestrator runs overlap (e.g., a delayed cron start while the previous is still running), they could submit duplicate workers. This is mostly harmless — duplicates compete for keys via `reserve_jobs=True` and excess workers exit when they find no work — but the wasted submissions are undesirable. The implementation should consider a lightweight lock (e.g., checking via `squeue` whether an orchestrator job is already running before submitting).
