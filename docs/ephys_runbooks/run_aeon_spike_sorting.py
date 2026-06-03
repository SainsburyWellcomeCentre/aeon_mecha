"""Run spike sorting on a single key, selected by --task index.

Designed for SLURM job arrays: each array task gets its own GPU and
processes one key from the list. The SLURM script passes
$SLURM_ARRAY_TASK_ID as --task automatically.

Example output of write_spike_sorting_scripts() — golden dataset values.
Generated for experiment "abcGolden01-aeonx1", subject "IAA-1147881".

To run a single task interactively (e.g. for debugging):
    uv run python run_aeon_spike_sorting.py --task 1
"""

import argparse

from aeon.dj_pipeline import spike_sorting

# =============================================================================
# Configuration
# =============================================================================

_base = {
    "experiment_name": "abcGolden01-aeonx1",
    "subject": "IAA-1147881",
    "insertion_number": 1,
    "probe_type": "neuropixels2.0",
    "electrode_config_name": "0-383",
    "paramset_id": "400",
}

keys = [
    # --- shank0 ---
    {**_base, "electrode_group": "shank0", "block_start": "2026-05-11 07:49:46.571574", "block_end": "2026-05-11 08:19:46.571574"},
    {**_base, "electrode_group": "shank0", "block_start": "2026-05-11 08:09:46.571574", "block_end": "2026-05-11 08:39:46.571574"},
    {**_base, "electrode_group": "shank0", "block_start": "2026-05-11 08:29:46.571574", "block_end": "2026-05-11 08:59:46.571574"},
    # --- shank1 ---
    {**_base, "electrode_group": "shank1", "block_start": "2026-05-11 07:49:46.571574", "block_end": "2026-05-11 08:19:46.571574"},
    {**_base, "electrode_group": "shank1", "block_start": "2026-05-11 08:09:46.571574", "block_end": "2026-05-11 08:39:46.571574"},
    {**_base, "electrode_group": "shank1", "block_start": "2026-05-11 08:29:46.571574", "block_end": "2026-05-11 08:59:46.571574"},
    # --- shank2 ---
    {**_base, "electrode_group": "shank2", "block_start": "2026-05-11 07:49:46.571574", "block_end": "2026-05-11 08:19:46.571574"},
    {**_base, "electrode_group": "shank2", "block_start": "2026-05-11 08:09:46.571574", "block_end": "2026-05-11 08:39:46.571574"},
    {**_base, "electrode_group": "shank2", "block_start": "2026-05-11 08:29:46.571574", "block_end": "2026-05-11 08:59:46.571574"},
    # --- shank3 ---
    {**_base, "electrode_group": "shank3", "block_start": "2026-05-11 07:49:46.571574", "block_end": "2026-05-11 08:19:46.571574"},
    {**_base, "electrode_group": "shank3", "block_start": "2026-05-11 08:09:46.571574", "block_end": "2026-05-11 08:39:46.571574"},
    {**_base, "electrode_group": "shank3", "block_start": "2026-05-11 08:29:46.571574", "block_end": "2026-05-11 08:59:46.571574"},
]

# =============================================================================

# If a sorting job failed with "error" status, DataJoint will skip it on
# resubmit. Set CLEAR_JOB = True to clear error entries before re-running.
#
# NOTE: If a job was killed externally (scancel, OOM killer, node crash),
# it may be stuck as "reserved" instead of "error". CLEAR_JOB only handles
# "error" entries. To clear "reserved" entries, run manually:
#   from aeon.dj_pipeline import spike_sorting
#   (spike_sorting.schema.jobs & {"status": "reserved"}).delete()
CLEAR_JOB = False


def clear_job(key):
    """Clear errored jobs for SpikeSorting to allow re-running."""
    try:
        (spike_sorting.schema.jobs & {"table_name": spike_sorting.SpikeSorting.table_name, "status": "error"} & key).delete()
    except Exception as e:
        print(f"[WARNING] Could not clear error jobs: {e}")


def main():
    """Run spike sorting on the key corresponding to --task index."""
    parser = argparse.ArgumentParser(description="Spike sorting worker")
    parser.add_argument(
        "--task", type=int, required=True, help=f"Task number (1-{len(keys)}), from SLURM_ARRAY_TASK_ID"
    )
    args = parser.parse_args()

    if not 1 <= args.task <= len(keys):
        raise ValueError(f"Task must be 1-{len(keys)}, got {args.task}")

    key = keys[args.task - 1]
    print(
        f"\n=== Task {args.task}/{len(keys)}: {key['electrode_group']} "
        f"{key['block_start']} - {key['block_end']} ==="
    )

    if CLEAR_JOB:
        clear_job(key)
    spike_sorting.SpikeSorting.populate(key, reserve_jobs=True, display_progress=True)
    print(f"=== Task {args.task} done ===")


if __name__ == "__main__":
    main()
