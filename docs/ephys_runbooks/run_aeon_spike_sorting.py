"""Example output of write_spike_sorting_scripts() — golden dataset values.

This file shows what the generated run_aeon_spike_sorting.py looks like
for the abcGolden01 test dataset. To generate your own, run
write_spike_sorting_scripts() from step03_spike_sorting.py.

Designed for SLURM job arrays: each array task gets its own GPU and
processes one key from the list. The SLURM script passes
$SLURM_ARRAY_TASK_ID as --task automatically.

To run a single task interactively (e.g. for debugging):
    uv run python run_aeon_spike_sorting.py --task 1

To select the pipeline step ("PreProcessing", "SpikeSorting", "PostProcessing"):
  - Pass --table at runtime to override TABLE_NAME (e.g. PreProcessing):
        uv run python run_aeon_spike_sorting.py --task 1 --table PreProcessing
  - Or edit TABLE_NAME below to change the default for all tasks.
"""

import argparse

from aeon.dj_pipeline import spike_sorting

# =============================================================================
# Configuration — edit these for your experiment
# =============================================================================

# Which pipeline step to run
TABLE_NAME = "SpikeSorting"

# Common key fields (shared across all blocks/shanks)
_base = {
    "experiment_name": "abcGolden01-aeonx1",
    "subject": "IAA-1147881",
    "insertion_number": 1,
    "probe_type": "neuropixels2.0",
    "electrode_config_name": "0-383",
    "paramset_id": "400",
}

# Golden dataset: 3 blocks × 4 shanks = 12 sorting jobs
# Task numbers are 1-indexed (task 1 = keys[0], task 12 = keys[11])
keys = [
    # --- shank0 (tasks 1-3) ---
    {
        **_base,
        "electrode_group": "shank0",
        "block_start": "2026-05-11 07:49:46.571574",
        "block_end": "2026-05-11 08:19:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank0",
        "block_start": "2026-05-11 08:09:46.571574",
        "block_end": "2026-05-11 08:39:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank0",
        "block_start": "2026-05-11 08:29:46.571574",
        "block_end": "2026-05-11 08:59:46.571574",
    },
    # --- shank1 (tasks 4-6) ---
    {
        **_base,
        "electrode_group": "shank1",
        "block_start": "2026-05-11 07:49:46.571574",
        "block_end": "2026-05-11 08:19:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank1",
        "block_start": "2026-05-11 08:09:46.571574",
        "block_end": "2026-05-11 08:39:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank1",
        "block_start": "2026-05-11 08:29:46.571574",
        "block_end": "2026-05-11 08:59:46.571574",
    },
    # --- shank2 (tasks 7-9) ---
    {
        **_base,
        "electrode_group": "shank2",
        "block_start": "2026-05-11 07:49:46.571574",
        "block_end": "2026-05-11 08:19:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank2",
        "block_start": "2026-05-11 08:09:46.571574",
        "block_end": "2026-05-11 08:39:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank2",
        "block_start": "2026-05-11 08:29:46.571574",
        "block_end": "2026-05-11 08:59:46.571574",
    },
    # --- shank3 (tasks 10-12) ---
    {
        **_base,
        "electrode_group": "shank3",
        "block_start": "2026-05-11 07:49:46.571574",
        "block_end": "2026-05-11 08:19:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank3",
        "block_start": "2026-05-11 08:09:46.571574",
        "block_end": "2026-05-11 08:39:46.571574",
    },
    {
        **_base,
        "electrode_group": "shank3",
        "block_start": "2026-05-11 08:29:46.571574",
        "block_end": "2026-05-11 08:59:46.571574",
    },
]

# =============================================================================

CLEAR_JOB = False  # Set True to clear 'error' job state before re-running


def clear_job(dj_table, key):
    """Clear errored jobs for this table to allow re-running.

    Note: DJ 2.x changed schema.jobs API. If this fails, set CLEAR_JOB=False
    and clear error jobs manually before resubmitting.
    """
    try:
        (spike_sorting.schema.jobs & {"table_name": dj_table.table_name, "status": "error"} & key).delete()
    except Exception as e:
        print(f"[WARNING] Could not clear error jobs: {dj_table.table_name} — {e}")


def main():
    """Run the selected pipeline step on the key corresponding to --task index."""
    parser = argparse.ArgumentParser(description="Spike sorting worker")
    parser.add_argument(
        "--task", type=int, required=True, help=f"Task number (1-{len(keys)}), from SLURM_ARRAY_TASK_ID"
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        choices=["PreProcessing", "SpikeSorting", "PostProcessing"],
        help=f"Pipeline step to run (default: {TABLE_NAME})",
    )
    args = parser.parse_args()

    if not 1 <= args.task <= len(keys):
        raise ValueError(f"Task must be 1-{len(keys)}, got {args.task}")

    populate_table = {
        "PreProcessing": spike_sorting.PreProcessing,
        "SpikeSorting": spike_sorting.SpikeSorting,
        "PostProcessing": spike_sorting.PostProcessing,
    }[args.table]

    key = keys[args.task - 1]
    print(
        f"\n=== Task {args.task}/{len(keys)}: {key['electrode_group']} "
        f"{key['block_start']} - {key['block_end']} ==="
    )

    if CLEAR_JOB:
        clear_job(populate_table, key)
    populate_table.populate(key, reserve_jobs=True, display_progress=True)
    print(f"=== Task {args.task} done ===")


if __name__ == "__main__":
    main()
