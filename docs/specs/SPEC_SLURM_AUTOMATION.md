# SLURM Pipeline Automation Specifications

## What's New (v2)

Based on Thinh's feedback and alignment with the `datajoint-worker` spec:

- **Unified worker model.** Every SLURM task runs a `DataJointWorker` internally. No separate "batch" vs "per-key" modes — tiers differ only in SLURM resources, DataJointWorker config, and how many instances to submit. DJ's `reserve_jobs=True` handles all key distribution.
- **`datajoint-worker` is a dependency.** New Section 2 explains the relationship. This package is a SLURM scheduling layer on top of DataJointWorker, not a replacement for it.
- **Adopted DataJointWorker features.** Transient error auto-clearing, orphan recovery via `stale_timeout_hours`, worker registration via `RegisteredWorker` tables, and progress reporting via `get_workflow_operation_overview()` — all used as-is from `datajoint-worker`.
- **Kept WorkerErrorHistory.** Confirmed that DJ 2.x `~~jobs` tables delete error details when errors are cleared. This table fills that gap. Capture mechanism uses a DataJointWorker subclass.
- **Dropped DJ 0.14.x support.** DJ 2.x only going forward.
- **Repository under `datajoint-company`** alongside `datajoint-worker`, designed for eventual integration into a unified worker system.

---

## Table of Contents
1. [Overview](#overview)
2. [Relationship to DataJointWorker](#relationship-to-datajointworker)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Orchestrator](#orchestrator)
6. [Workers](#workers)
7. [Error Tracking and History](#error-tracking-and-history)
8. [CLI Interface](#cli-interface)
9. [Packaging and Installation](#packaging-and-installation)
10. [Example: Aeon Pipeline Configuration](#example-aeon-pipeline-configuration)
11. [Open Questions for Discussion](#open-questions-for-discussion)

---

## Overview

### Purpose

DataJoint pipelines on HPC clusters currently require manual intervention to process data: a human must SSH into the cluster, write or edit scripts, and run `sbatch` or `populate()` by hand. This works for one-off jobs but does not scale to continuous, multi-table pipelines where new data arrives regularly and processing should happen automatically.

This spec describes a general-purpose package for automating DataJoint pipeline processing on SLURM-managed HPC clusters. The package handles the scheduling, job submission, resource allocation, and error tracking needed to turn a manual populate-and-check workflow into a hands-off automated system that can be monitored remotely.

The package is not specific to any one pipeline. It is designed so that any DataJoint project running on a SLURM cluster can adopt it by writing a configuration file that describes their tables and resource requirements. Project Aeon at the Sainsbury Wellcome Centre is the first intended user, but the design is general.

### What This Package Does

- Periodically submits SLURM jobs that run DataJointWorker instances to call `populate()` on configured tables
- Assigns different SLURM resource allocations (CPU, memory, GPU, wall time) to different groups of tables based on their computational requirements
- Tracks orchestrator runs and maintains a historical log of populate errors, complementing DataJoint's built-in per-table jobs system
- Provides a CLI for starting, stopping, and monitoring the automation

### What This Package Does Not Do

- No web UI or dashboard (that is the domain of DataJoint Works)
- No pipeline table definitions — those belong to the project's own codebase
- No data processing logic — the package calls `populate()` via DataJointWorker and lets DataJoint handle the rest
- No dependency orchestration between tables — DataJoint's `populate()` is idempotent and will simply return if upstream data is not ready, so the system relies on running frequently enough for data to cascade through the pipeline over successive cycles
- No job reservation or key assignment logic — DataJoint's built-in `reserve_jobs=True` mechanism handles distributed key assignment, and DataJointWorker handles the populate loop. This package only determines what SLURM resources to allocate and how many worker instances to submit.

### Design Principles

**Leverage DataJointWorker.** The `datajoint-worker` package already handles the populate loop: iterating over registered tables, calling `populate(reserve_jobs=True)`, clearing transient errors, recovering orphaned jobs, and registering workers for external observability. This package does not replicate any of that. It is a SLURM scheduling and submission layer on top of DataJointWorker.

**Keep it simple.** The orchestrator's job is to count pending work and submit SLURM jobs. Each SLURM job runs a DataJointWorker that does the actual populate work. There is no complex state machine, no dependency graph resolution, and no inter-job communication.

**Be general, not Aeon-specific.** Every project-specific detail (table names, resource requirements, schedule interval) lives in the configuration file, not in the package code.

**Align with the DataJoint worker ecosystem.** This package is designed so that it can eventually be integrated into a unified worker management system that supports multiple infrastructure backends (SLURM, AWS, Kubernetes). The SLURM-specific parts (sbatch generation, job arrays, partition management) are isolated from the worker logic (which is handled by `datajoint-worker`).

---

## Relationship to DataJointWorker

This package depends on and extends `datajoint-worker` (`datajoint-company/datajoint-worker`), which provides the core `DataJointWorker` class for orchestrating DataJoint `populate()` operations.

### What DataJointWorker Provides

`DataJointWorker` is a configurable loop executor. You register tables as steps, call `run()`, and it loops: refresh jobs → populate each step → clear transient errors → sleep → repeat. It handles:

- **Populate loop** — iterates over registered tables, calls `populate(reserve_jobs=True, suppress_errors=True)` on each
- **Distributed key assignment** — DataJoint's `reserve_jobs=True` uses atomic SQL operations to ensure only one worker processes each key, even when multiple workers run concurrently
- **Orphan recovery** — calls `jobs.refresh(orphan_timeout=...)` before each populate cycle to recover keys left in `reserved` status by workers that died
- **Transient error auto-clearing** — automatically clears known infrastructure errors (deadlocks, lost connections, timeouts) so they retry on the next cycle
- **Worker registration** — registers itself and its process steps in `RegisteredWorker` / `RegisteredWorker.Process` database tables for external observability
- **Progress reporting** — `get_workflow_operation_overview()` returns pending/reserved/success/error/ignore counts per process
- **Graceful shutdown** — SIGTERM handler finishes the current cycle then exits
- **Idle detection** — `max_idled_cycle` stops the worker after N consecutive cycles with no work
- **Runtime limiting** — `run_duration` stops the worker after N seconds

### What This Package Adds

This SLURM package adds the infrastructure layer that DataJointWorker does not handle:

- **SLURM resource allocation** — determining what partition, CPUs, memory, GPU, and wall time each worker needs
- **Tier-based configuration** — grouping tables by resource requirements and defining SLURM specs per group
- **Instance scaling** — determining how many parallel worker instances to submit for each tier based on pending work
- **Job submission** — generating sbatch scripts and submitting them to SLURM, using job arrays for parallel instances
- **Orchestrator scheduling** — firing the orchestrator on a recurring schedule via SLURM or cron
- **Persistent error history** — `WorkerErrorHistory` table that preserves error details across retries (DataJoint's `~~jobs` tables lose error information when errors are cleared)
- **Orchestrator run logging** — `OrchestratorRunHistory` table tracking each automation cycle

### How They Fit Together

```
This package (datajoint-slurm)          datajoint-worker              datajoint-python 2.x
─────────────────────────────           ────────────────              ────────────────────

Orchestrator                            DataJointWorker               Per-table ~~jobs tables
  reads YAML config                       add_step(table)               jobs.refresh()
  counts pending work                     run()                         jobs.reserve(key)
  submits SLURM jobs ──sbatch──►          populate loop                 populate(reserve_jobs=True)
                                          error auto-clearing           error/success recording
                                          orphan recovery
                                          worker registration

WorkerErrorHistory                      RegisteredWorker
OrchestratorRunHistory                  RegisteredWorker.Process
                                        get_workflow_operation_overview()
```

Each SLURM job submitted by the orchestrator runs a DataJointWorker instance. The SLURM package controls how many instances run and with what resources. DataJointWorker controls what each instance does once it starts.

---

## Architecture

The system has three layers:

### Configuration Layer

A YAML configuration file, maintained by the project team, that declares:
- What DataJoint tables exist in the pipeline
- What SLURM resources each group of tables needs
- How the DataJointWorker should be configured for each group
- How many parallel workers to run for each group
- How often the orchestrator should run

This is the only project-specific artifact. Everything else is provided by the package.

### Orchestrator Layer

A lightweight process that reads the configuration, checks what work is pending, and submits SLURM jobs for the workers. The orchestrator itself runs as a minimal SLURM job on a CPU partition. It typically completes in under a minute — its only job is to query DataJoint and call `sbatch`.

### Worker Layer

The SLURM jobs that do the actual `populate()` calls. Each worker runs a DataJointWorker instance configured with the tables and settings for its tier. Workers differ across tiers in their SLURM resource allocation and DataJointWorker configuration, but they all use the same underlying mechanism: DataJointWorker's populate loop with `reserve_jobs=True` for distributed key assignment.

- **Long-running workers** (for lightweight tables) loop through their tables multiple times, picking up new work each cycle. They exit after a few idle cycles or when their wall time approaches.
- **Single-pass workers** (for heavy tables) grab one key, process it, and exit. Multiple instances run in parallel via SLURM job arrays, with DataJoint's reservation system sorting out who processes which key.

```
┌─────────────────────────────────────────────────────┐
│                    SLURM Cluster                     │
│                                                      │
│  ┌──────────────┐                                    │
│  │ Orchestrator  │  (CPU, 2 cores, 4GB, 30min)       │
│  │              │                                    │
│  │  Reads config │                                    │
│  │  Counts work  │                                    │
│  │  Submits jobs │                                    │
│  └──────┬───────┘                                    │
│         │                                            │
│         ├──── sbatch ──── Worker (light tier)          │
│         │                 1 instance, long-running     │
│         │                 loops through all tables     │
│         │                 (CPU, 2 cores, 8GB, 4hr)    │
│         │                                            │
│         ├──── sbatch ──── Workers (medium tier)        │
│         │     --array     N instances via job array    │
│         │                 each processes multiple keys │
│         │                 (CPU, 8 cores, 64GB, 8hr)   │
│         │                                            │
│         └──── sbatch ──── Workers (heavy tier)         │
│               --array     N instances via job array    │
│                           each processes 1 key        │
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

# Schema name for the package's tracking tables
# (WorkerErrorHistory, OrchestratorRunHistory).
# Combined with DataJoint's database_prefix setting.
# e.g., if database_prefix = "aeon_", this becomes "aeon_slurm_worker"
schema_name: slurm_worker

# Named resource tiers. Each tier defines SLURM resource
# requirements, DataJointWorker configuration, and scaling.
tiers:

  light:
    slurm:
      partition: cpu
      cpus: 2
      memory: 8G
      time: "4:00:00"
      # gpu is omitted — no GPU for this tier

    # DataJointWorker configuration for this tier.
    worker:
      sleep_duration: 60        # seconds between cycles
      max_idled_cycle: 3        # exit after 3 idle cycles
      run_duration: 13800       # 3h50m — exit before 4h wall time
      # max_calls not set — process all available keys per table

    # How many parallel SLURM jobs to submit for this tier.
    # "1" means a single long-running worker.
    max_concurrent: 1

    # Tables to populate, in dependency order (upstream first).
    # Ordering is a recommendation — populate() is idempotent,
    # so out-of-order execution is safe, just less efficient.
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

    worker:
      sleep_duration: 30
      max_idled_cycle: 1        # single pass, then exit
      max_calls: 5              # process up to 5 keys per table
      stale_timeout_hours: 9    # recover orphans after 9hr

    # Orchestrator computes needed instances from pending keys
    # and max_calls, capped at this limit. See Scaling section.
    max_concurrent: 10

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

    worker:
      sleep_duration: 30
      max_idled_cycle: 1
      max_calls: 1              # one key per worker
      stale_timeout_hours: 200  # >7 days

    # Capped to avoid monopolizing shared GPU resources.
    max_concurrent: 2

    tables:
      - module.path.TableF
```

### Configuration Details

**Tiers** are user-defined and named. The package imposes no limit on the number of tiers or their names. Each tier has a SLURM resource spec, a DataJointWorker configuration, a scaling limit, and a list of tables.

**Worker configuration** maps directly to `DataJointWorker` constructor and `add_step()` parameters:
- `sleep_duration` — seconds between populate cycles
- `max_idled_cycle` — stop the worker after this many consecutive cycles with no successful jobs. Use `-1` for unlimited (worker runs until wall time).
- `max_calls` — limit how many keys each table processes per cycle. Omit for no limit. This is passed to `add_step()` as a keyword argument forwarded to `populate()`.
- `stale_timeout_hours` — hours before a reserved job is considered orphaned and recovered. Should be slightly longer than the wall time for this tier.
- `run_duration` — maximum seconds the worker should run before exiting gracefully. Should be set slightly below the SLURM wall time (e.g., 10 minutes less) so the worker finishes its current cycle and exits cleanly before SLURM sends SIGTERM/SIGKILL. Primarily relevant for long-running tiers where `max_idled_cycle` alone may not cause the worker to stop before the wall time.
- `autoclear_error_patterns` — additional SQL LIKE patterns for transient errors to auto-clear (added to DataJointWorker's built-in list of deadlocks, lost connections, etc.)

**Scaling** (`max_concurrent`) controls how many parallel SLURM jobs the orchestrator submits for a tier. For tiers where `max_concurrent` is 1, the orchestrator submits a single SLURM job (if any pending keys exist). For tiers where `max_concurrent` is greater than 1, the orchestrator computes how many worker instances are needed based on the pending keys and `max_calls` setting:

- For each table in the tier, count pending keys.
- If `max_calls` is set, the number of workers needed for that table is `ceil(pending / max_calls)`. If `max_calls` is not set, one worker can handle all pending keys for that table, so the needed count is 1 (if any are pending) or 0.
- Since each worker processes all tables in its tier, take the *maximum* needed count across all tables in the tier (not the sum — a single worker instance handles all tables in its tier sequentially).
- Cap at `max_concurrent`.

For example, if a medium tier has Table D with 12 pending keys and Table E with 8 pending keys, and `max_calls: 5`, the needed count is `max(ceil(12/5), ceil(8/5)) = max(3, 2) = 3`. This submits a job array of 3 instances, each processing up to 5 keys from each table.

Each instance runs a DataJointWorker that uses `reserve_jobs=True` — DataJoint's atomic reservation system distributes keys dynamically at runtime, so there is no need for the orchestrator to pre-assign specific keys to specific instances.

**Table references** use dotted Python module paths (e.g., `aeon.dj_pipeline.spike_sorting.SpikeSorting`). The package imports these at runtime to resolve the actual DataJoint table classes.

**SLURM resource specs** map directly to `#SBATCH` directives. The `gpu` field is optional — omit it for CPU-only tiers. For site-specific SLURM directives not covered by the standard fields (e.g., `--account`, `--qos`, `--constraint`), the `slurm` block also supports an `extra_args` list that is passed through to sbatch as-is.

**Worker logging:** Generated sbatch scripts configure `--output` and `--error` directives pointing to a `slurm_output/` directory with the job ID in the filename. This ensures that SLURM-level failures (import errors, environment issues, OOM kills) that never reach the `WorkerErrorHistory` table are still captured in log files on disk.

**Schedule interval** controls how often the orchestrator fires. Shorter intervals mean data cascades through the pipeline faster (new upstream rows become available to downstream tables sooner). A 3-hour interval is reasonable for most pipelines; adjust based on data arrival frequency and processing times.

---

## Orchestrator

The orchestrator is the central coordinator. It runs as a lightweight SLURM job and its only responsibility is to determine how many workers each tier needs and submit them.

### What the Orchestrator Does Each Cycle

1. Load the configuration file
2. For each tier, in the order they appear in the config:
   - Before submitting, check `squeue` for any already-running jobs for this tier (matched by job name). Subtract running jobs from the needed count to avoid duplicate submissions. For `max_concurrent: 1` tiers, skip submission entirely if a worker is already running.
   - If `max_concurrent` is 1: submit a single SLURM job with the tier's resource spec (if no worker is already running and any pending keys exist).
   - If `max_concurrent` is greater than 1: for each table, call `table.jobs.refresh()` and count pending keys. Compute the needed instance count as `max(ceil(pending_per_table / max_calls))` across tables in the tier (see [Scaling](#configuration-details) for the full formula), subtract already-running instances, cap at `max_concurrent`, and submit that many instances as a SLURM job array (`sbatch --array=0-N`).
   - Each submitted job runs a DataJointWorker configured with the tier's tables and worker settings.
3. Log a summary to the `OrchestratorRunHistory` table: timestamp, number of jobs submitted per tier, any errors encountered during submission.
4. Schedule the next run (mechanism depends on chosen scheduling approach — see [Open Questions](#open-questions-for-discussion)).
5. Exit.

**Note on pending key counting:** The orchestrator calls `table.jobs.refresh()` before reading `table.jobs.pending` for two reasons: (1) `refresh()` discovers new pending keys by computing `key_source - target - jobs`, and (2) it recovers orphaned reserved jobs via `orphan_timeout`. The pending count determines how many SLURM tasks to submit, not which keys each task processes — DataJoint's reservation system handles key assignment at runtime.

**Job arrays:** When the orchestrator submits multiple instances for a tier, it uses SLURM job arrays (`sbatch --array=0-N`). Each array task runs independently with its own resources. SLURM treats the array as a single unit — easier to monitor via `squeue` and cancel via `scancel`. If pending work changes between submission and execution (SLURM queue delays can be significant), the dynamic reservation system handles it gracefully: tasks that find no work simply exit after one idle cycle.

### Robustness

The orchestrator wraps all operations in error handling so that a failure in one tier does not prevent other tiers from being processed. If checking pending keys for one table fails, the orchestrator logs the error and continues with the remaining tables and tiers. The orchestrator itself should never crash — individual failures are logged, not propagated.

### Resource Requirements

The orchestrator needs minimal resources. It only runs Python code to query DataJoint and call `sbatch` via subprocess. A reasonable default allocation: 2 CPU cores, 4GB memory, 30-minute wall time on a CPU partition.

---

## Workers

Workers are the SLURM jobs that do the actual data processing. Each worker runs a DataJointWorker instance configured with the tables for its tier.

### How Workers Operate

When a SLURM job starts, the worker script:

1. Loads the YAML configuration for its tier
2. Constructs a unique `worker_name` from the tier name and SLURM job identity: `{tier}_{SLURM_JOB_ID}` for single-instance tiers, `{tier}_{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}` for job array instances. This ensures each DataJointWorker registers with a distinct name in the `RegisteredWorker` table.
3. Creates a `DataJointWorker` instance with the worker name, the `worker_schema_name` from the config (same schema as this package's tracking tables, e.g., `slurm_worker`), and the tier's settings (`sleep_duration`, `max_idled_cycle`, `run_duration`, `stale_timeout_hours`, `autoclear_error_patterns`). DataJointWorker's `RegisteredWorker` and `RegisteredWorker.Process` tables are created in this schema alongside our `WorkerErrorHistory` and `OrchestratorRunHistory` tables.
4. Registers the tier's tables as steps via `add_step()`, passing `max_calls` if configured
5. Calls `worker.run()`

From there, DataJointWorker takes over:
- Each cycle, it calls `jobs.refresh(orphan_timeout=...)` then `populate(reserve_jobs=True, suppress_errors=True)` on each registered table
- `reserve_jobs=True` ensures atomic key reservation — multiple concurrent workers safely distribute work without processing the same key twice
- `suppress_errors=True` catches exceptions during `make()` and continues to the next key instead of aborting
- After populate, transient errors (deadlocks, lost connections, etc.) are automatically cleared so they retry on the next cycle
- The worker stops when `max_idled_cycle` consecutive idle cycles are reached, or `run_duration` is exceeded, or SIGTERM is received

### Worker Configurations by Tier Pattern

**Long-running workers** (typical for lightweight tables): `max_concurrent: 1`, `max_idled_cycle: 3`, no `max_calls` limit. One SLURM job runs a DataJointWorker that loops through all the tier's tables, processing whatever is available each cycle. It exits after 3 consecutive idle cycles, freeing the SLURM allocation when there's nothing to do.

**Multi-key workers** (typical for medium-weight tables): `max_concurrent: N`, `max_idled_cycle: 1`, `max_calls: 5`. Multiple SLURM jobs run as a job array. Each DataJointWorker does a single pass, processing up to 5 keys per table, then exits. DataJoint's reservation system distributes keys among the concurrent workers.

**Single-key workers** (typical for heavy/GPU tables): `max_concurrent: N` (capped low), `max_idled_cycle: 1`, `max_calls: 1`. Multiple SLURM jobs run as a job array. Each DataJointWorker grabs one key, processes it (possibly over days), then exits. The low `max_concurrent` cap prevents monopolizing shared GPU resources.

These are not distinct modes enforced by the package — they are configuration patterns. A project can configure any combination of settings per tier.

### Environment Setup

All workers need access to:
- The project's Python environment (e.g., managed by `uv`, `conda`, or `venv`)
- DataJoint configuration (database credentials, connection settings)
- The project's codebase (so that table module paths can be imported)

The package generates sbatch scripts at runtime based on the tier configuration. These scripts handle environment activation and invocation of the worker entry point. Site-specific setup (e.g., `module load` commands, environment variables like `PYTORCH_CUDA_ALLOC_CONF` for GPU tiers) can be passed via the `extra_args` field for SBATCH directives. Pre-populate shell commands (like `module load`) are a separate concern — see [Open Question 4](#4-additional-slurm-configuration) for whether a `setup_commands` config field is needed.

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

**What the built-in system does not provide:** When an error entry is cleared (to allow retry), the error row is deleted from the `~~jobs` table and a new `pending` row is inserted. The error information — message, stack trace, timestamp — is lost. There is no history of past failures across retries. There is also no record of when populate runs happened or what SLURM jobs were involved.

### DataJointWorker's Error Auto-Clearing

DataJointWorker automatically clears known transient errors after each populate cycle. Permanent patterns (deadlocks, lost connections, lock wait timeouts, SIGTERM) are always cleared. Projects can add pipeline-specific patterns via `autoclear_error_patterns` in the tier config.

This means transient infrastructure errors are retried automatically on the next cycle without manual intervention. Only genuine data/code errors require human attention.

### WorkerErrorHistory Table

This package adds a `WorkerErrorHistory` table that maintains a permanent, append-only log of all populate errors.

**Capture mechanism:** The worker entry point subclasses `DataJointWorker` to insert an error-capture step into the populate cycle. DataJointWorker's `_run_once()` method follows the sequence: `refresh()` → `populate()` each step → auto-clear transient errors. The subclass overrides this to query each table's `~~jobs` for rows with `error` status *after* populate but *before* transient error auto-clearing. Any errors not already in `WorkerErrorHistory` (matched by table name, key, and error message) are appended. This ensures that even transient errors (deadlocks, lost connections) that DataJointWorker subsequently auto-clears are captured before they are deleted from `~~jobs`. If subclassing proves too brittle across `datajoint-worker` versions, an alternative is to request an upstream callback hook in DataJointWorker's cycle — but subclassing is sufficient for the initial implementation.

Each captured error row contains:

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

This table fills a gap that DataJoint's `~~jobs` tables do not address: when you clear an error from `~~jobs` to allow a retry, the error details are deleted. If the retry fails with a different error, there is no way to compare the new error with the old one. `WorkerErrorHistory` provides that history — you can see every error a key has ever encountered, in order, regardless of whether the errors have been cleared from `~~jobs`.

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
2. Investigate: review error messages and stack traces, comparing across retries if needed
3. Fix the underlying issue (code bug, missing data, permissions, etc.)
4. Clear the error: `dj-slurm clear-errors --table module.path.TableName --key '{"col": "value"}'` (removes the error entry from the table's `~~jobs` table, triggering re-insertion as `pending` on next `refresh()`)
5. Next orchestrator cycle: the key is picked up as pending and retried automatically

The error history in `WorkerErrorHistory` is preserved regardless of whether the error is cleared from the jobs table.

### SLURM Wall-Time Kills and Orphaned Jobs

When SLURM kills a worker that exceeds its wall-time limit, it sends SIGTERM followed (after a grace period) by SIGKILL. DataJoint 2.x installs a SIGTERM handler during `populate(reserve_jobs=True)` that raises `SystemExit`, allowing in-flight transactions to roll back. DataJointWorker also installs its own SIGTERM handler that finishes the current cycle and exits gracefully.

If SIGKILL arrives before the key can be marked as `error`, it remains in `reserved` status. DataJointWorker's `stale_timeout_hours` parameter handles this: `jobs.refresh(orphan_timeout=...)` is called each cycle, automatically recovering reserved jobs older than the configured timeout. The orchestrator also calls `refresh()` when counting pending keys, so orphaned keys from killed workers are recovered on the next orchestrator cycle.

The `stale_timeout_hours` should be set per tier based on expected processing time. For example, a heavy tier with a 7-day wall time should use a stale timeout of roughly 200 hours (~8.3 days), so that legitimately long-running jobs are not prematurely recovered.

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

The `--dry-run` flag prints what SLURM jobs would be submitted (tier, instance count, resource spec) without actually submitting them. This is valuable for testing new configurations.

### `dj-slurm status`

Show the current state of the automation: whether the orchestrator is scheduled, what workers are currently running (via `squeue`), the most recent orchestrator run summary, and progress across all registered workers (via `get_workflow_operation_overview()`).

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

Clear error entries from a table's `~~jobs` table to allow retrying failed keys. The key is removed from `~~jobs` and will be re-added as `pending` on the next `jobs.refresh()` call. Does not delete from `WorkerErrorHistory` (that history is permanent).

The `--table` argument uses the full module path (as listed in the config file), or the short class name if it is unambiguous within the loaded config.

```bash
dj-slurm clear-errors --table aeon.dj_pipeline.spike_sorting.SpikeSorting --key '{"block_start": "2024-06-04 11:00:00"}'
dj-slurm clear-errors --table SpikeSorting --all   # short name OK if unambiguous in config
```

---

## Packaging and Installation

### Package Structure

The package is a standalone Python package with its own GitHub repository under the DataJoint company organization (e.g., `datajoint-company/datajoint-slurm`).

```
datajoint-slurm/
├── pyproject.toml
├── README.md
├── src/
│   └── datajoint_slurm/
│       ├── __init__.py
│       ├── config.py          # YAML config loading and validation
│       ├── orchestrator.py    # Orchestrator logic
│       ├── worker.py          # Worker entry point (creates DataJointWorker)
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
    "datajoint>=2.1.0",
    "datajoint-slurm",
]
```

### Dependencies

- `datajoint-worker` — for the DataJointWorker populate loop
- `datajoint>=2.1.0` — for table definitions, populate, and the per-table `~~jobs` system
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
    worker:
      sleep_duration: 60
      max_idled_cycle: 3
      run_duration: 13800       # 3h50m — exit before 4h wall time
    max_concurrent: 1
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
      # Spike sorting post-processing (lightweight reads)
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
    worker:
      sleep_duration: 30
      max_idled_cycle: 1
      max_calls: 5
      stale_timeout_hours: 9
    max_concurrent: 10
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
    worker:
      sleep_duration: 30
      max_idled_cycle: 1
      max_calls: 1
      stale_timeout_hours: 200
    max_concurrent: 2
    tables:
      - aeon.dj_pipeline.spike_sorting.SpikeSorting
```

### How Data Flows Through This Configuration

Consider a new experimental epoch arriving in the system:

- **Cycle 1 (hour 0):** The light worker populates upstream tables — `EpochConfig` discovers the new epoch, `EphysEpoch` discovers the associated ephys data, `EphysBlockInfo` characterizes the blocks. The medium and heavy tiers have no pending keys yet, so the orchestrator submits zero instances for them.

- **Cycle 2 (hour 3):** The light worker has no new work for acquisition tables (already populated). The orchestrator finds pending keys for `PreProcessing` (the new blocks discovered in cycle 1) and submits medium-tier workers for them (up to 10 instances). If any PreProcessing jobs complete quickly, `SpikeSorting` keys may become pending — the orchestrator also checks the heavy tier and submits GPU workers if needed (up to 2 instances).

- **Cycle 3 (hour 6):** PreProcessing jobs from cycle 2 have finished. The orchestrator submits heavy-tier GPU workers for `SpikeSorting` (if not already submitted in cycle 2). Spike sorting begins.

- **Cycles 4-50+ (hours 9-150+):** Spike sorting runs for days. Each orchestrator cycle checks for completed sorting and submits `PostProcessing` medium-tier workers as keys become available. The light worker picks up `SortedSpikes`, `Waveform`, `SortingQuality`, `SyncedSpikes`, and `UnitMatching` as their upstream data becomes available.

The pipeline processes data end-to-end without manual intervention. The 3-hour interval means it takes a few cycles for data to cascade from the top to the bottom of the pipeline, but this is acceptable for a pipeline where the bottleneck step (spike sorting) takes days.

### GPU Resource Considerations

The SWC HPC cluster has 16 A100 GPUs (across 4 nodes) shared among all users. The `max_concurrent: 2` setting for the heavy tier ensures the Aeon pipeline uses at most 2 A100s at any given time (~12.5% of the pool). This should be adjusted based on cluster usage policies and other groups' needs. The cluster also has L40S, A4500, and Quadro RTX 5000 GPUs, but spike sorting is configured to use A100s specifically.

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

### 3. Package and CLI Naming

Working names used in this spec:
- Package: `datajoint-slurm` (import as `datajoint_slurm`)
- CLI: `dj-slurm`

These may need to change to align with DataJoint's naming conventions or to avoid confusion with other packages.

### 4. Additional SLURM Configuration

The current spec maps tier resource specs directly to `#SBATCH` directives and supports an `extra_args` list per tier for site-specific directives (e.g., `--account`, `--qos`, `--constraint`, `--export`). This covers most SLURM customization needs.

However, some sites may also need pre-`populate()` shell commands that are not `#SBATCH` directives, such as:
- Module loading commands (e.g., `module load uv`, `module load cuda`)
- Environment setup scripts (e.g., `source /path/to/env.sh`)

These could be configurable per tier via a `setup_commands` list in the YAML:

```yaml
heavy:
  slurm:
    partition: gpu
    # ...
  setup_commands:
    - "module load uv"
    - "module load cuda/12.1"
    - "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
```

Alternatively, sites could provide a single setup script that the package calls before starting the worker. The question is whether this level of configurability is needed for the initial release.

### 5. Database Schema Creation Privileges

The package creates its own DataJoint schema (e.g., `aeon_slurm_worker`) on first run. This requires `CREATE DATABASE` privileges on the database server. On shared DB servers (like SWC's aeon-db), this may require a one-time setup by a database administrator to pre-create the schema and grant the appropriate user permissions. The implementation should handle this gracefully — detect whether the schema exists and provide clear error messages if creation fails due to insufficient privileges.

### 6. Concurrent Orchestrator Protection

If two orchestrator runs overlap (e.g., a delayed SLURM start causes a cycle to still be running when the next fires), they could submit duplicate workers. This is mostly harmless due to `reserve_jobs=True` — duplicate workers simply compete for keys and excess workers exit after one idle cycle. However, the wasted SLURM submissions are undesirable. The implementation should consider a lightweight lock mechanism (e.g., checking if an orchestrator SLURM job is already running via `squeue` before submitting workers).
