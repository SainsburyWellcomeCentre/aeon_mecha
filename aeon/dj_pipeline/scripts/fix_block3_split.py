"""Fix block 3 OOM: replace the single 30h block with two 18h blocks.

Block 3 (2024-06-06 11:00 -> 2024-06-07 17:00) failed with CUDA OOM
during Kilosort4 final clustering. Split into:
  Block 3a: 2024-06-06 11:00 -> 2024-06-07 05:00 (18h)
  Block 3b: 2024-06-06 23:00 -> 2024-06-07 17:00 (18h)
Maintains 6h overlap with blocks 2, 3a<->3b, and 4.

Usage:
    uv run python -m aeon.dj_pipeline.scripts.fix_block3_split --dry-run
    uv run python -m aeon.dj_pipeline.scripts.fix_block3_split
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import datajoint as dj

# --- Configuration ---
EXPERIMENT_NAME = "social-ephys0.1-aeon3"
SUBJECT = "BAA-1104292"
INSERTION_NUMBER = 1

# Old block 3
OLD_START = "2024-06-06 11:00:00"
OLD_END = "2024-06-07 17:00:00"

# New split blocks
NEW_BLOCKS = [
    ("2024-06-06 11:00:00", "2024-06-07 05:00:00"),  # 3a: 18h
    ("2024-06-06 23:00:00", "2024-06-07 17:00:00"),  # 3b: 18h
]

# Sorting config (must match Phase 3)
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
ELECTRODE_GROUP_NAME = "0-95"
PARAMSET_ID = 1

PRODUCTION_HOST = "aeon-db2"
PRODUCTION_PREFIX = "aeon_"


def verify_production():
    host = dj.config.get("database.host", "")
    prefix = dj.config["custom"].get("database.prefix", "")
    assert PRODUCTION_HOST in host, f"Not production host! Got: {host}"
    assert prefix == PRODUCTION_PREFIX, f"Not production prefix! Got: {prefix}"
    print(f"  + Connected to {host} / {prefix}")


def main():
    parser = argparse.ArgumentParser(description="Split failed block 3 into two 18h blocks")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changes")
    args = parser.parse_args()

    verify_production()

    from aeon.dj_pipeline import ephys, spike_sorting

    old_block_key = {
        "experiment_name": EXPERIMENT_NAME,
        "subject": SUBJECT,
        "insertion_number": INSERTION_NUMBER,
        "block_start": OLD_START,
        "block_end": OLD_END,
    }

    old_task_key = {
        **old_block_key,
        "probe_type": PROBE_TYPE,
        "electrode_config_name": ELECTRODE_CONFIG_NAME,
        "electrode_group": ELECTRODE_GROUP_NAME,
        "paramset_id": PARAMSET_ID,
    }

    # ---------------------------------------------------------------
    # Step 1: Show current state
    # ---------------------------------------------------------------
    print("\n=== Step 1: Current state of old block 3 ===")
    old_block = ephys.EphysBlock & old_block_key
    old_info = ephys.EphysBlockInfo & old_block_key
    old_task = spike_sorting.SortingTask & old_task_key
    old_preproc = spike_sorting.PreProcessing & old_task_key
    old_sorting = spike_sorting.SpikeSorting & old_task_key

    print(f"  EphysBlock:     {len(old_block)} entries")
    print(f"  EphysBlockInfo: {len(old_info)} entries")
    print(f"  SortingTask:    {len(old_task)} entries")
    print(f"  PreProcessing:  {len(old_preproc)} entries")
    print(f"  SpikeSorting:   {len(old_sorting)} entries")

    # Check for error in jobs table
    jobs_error = spike_sorting.schema.jobs & {
        "table_name": spike_sorting.SpikeSorting.table_name,
        "status": "error",
    }
    print(f"  SpikeSorting error jobs: {len(jobs_error)}")

    if not old_block:
        print("\n  X Old block 3 not found — nothing to fix")
        return

    # ---------------------------------------------------------------
    # Step 2: Clean up old block 3 (bottom-up delete)
    # ---------------------------------------------------------------
    print("\n=== Step 2: Delete old block 3 (bottom-up) ===")

    if args.dry_run:
        print("  > Would delete SpikeSorting error jobs")
        print("  > Would delete PreProcessing + PreProcessing.File")
        print("  > Would delete SortingTask")
        print("  > Would delete EphysBlockInfo + EphysBlockInfo.Chunk + EphysBlockInfo.Channel")
        print("  > Would delete EphysBlock")
    else:
        # Clear error jobs first
        if jobs_error:
            (jobs_error).delete()
            print("  + Cleared SpikeSorting error jobs")

        # Delete SpikeSorting (should be empty since it failed)
        if old_sorting:
            (old_sorting).delete()
            print("  + Deleted SpikeSorting")

        # Delete PreProcessing (+ File part table)
        if old_preproc:
            (old_preproc).delete()
            print("  + Deleted PreProcessing")

        # Delete SortingTask
        if old_task:
            (old_task).delete()
            print("  + Deleted SortingTask")

        # Delete EphysBlockInfo (+ Chunk + Channel part tables)
        if old_info:
            (old_info).delete()
            print("  + Deleted EphysBlockInfo")

        # Delete EphysBlock
        (old_block).delete()
        print("  + Deleted EphysBlock")

    # ---------------------------------------------------------------
    # Step 3: Clean up old preprocessed files on disk
    # ---------------------------------------------------------------
    print("\n=== Step 3: Clean up old files on disk ===")
    # infer_output_dir expects datetime objects, not strings
    old_task_key_dt = {
        **old_task_key,
        "block_start": datetime.strptime(OLD_START, "%Y-%m-%d %H:%M:%S"),
        "block_end": datetime.strptime(OLD_END, "%Y-%m-%d %H:%M:%S"),
    }
    old_output_dir = spike_sorting.PreProcessing.infer_output_dir(old_task_key_dt)

    # Go up one level to get the block directory (contains the electrode group dir)
    old_block_dir = old_output_dir.parent.parent  # .../2024-06-06T11-00-00_2024-06-07T17-00-00/

    if old_block_dir.exists():
        if args.dry_run:
            print(f"  > Would delete: {old_block_dir}")
        else:
            shutil.rmtree(old_block_dir)
            print(f"  + Deleted: {old_block_dir}")
    else:
        print(f"  > Directory not found (already cleaned?): {old_block_dir}")

    # ---------------------------------------------------------------
    # Step 4: Insert new split blocks
    # ---------------------------------------------------------------
    print("\n=== Step 4: Insert new blocks (3a, 3b) ===")
    for i, (start, end) in enumerate(NEW_BLOCKS):
        label = chr(ord('a') + i)
        new_block_key = {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
            "insertion_number": INSERTION_NUMBER,
            "block_start": start,
            "block_end": end,
        }
        if args.dry_run:
            print(f"  > Would insert block 3{label}: {start} -> {end}")
        else:
            ephys.EphysBlock.insert1(new_block_key)
            print(f"  + Inserted block 3{label}: {start} -> {end}")

    # ---------------------------------------------------------------
    # Step 5: Populate EphysBlockInfo
    # ---------------------------------------------------------------
    print("\n=== Step 5: Populate EphysBlockInfo ===")
    if args.dry_run:
        print("  > Would run EphysBlockInfo.populate()")
    else:
        ephys.EphysBlockInfo.populate(display_progress=True, suppress_errors=False)
        for start, end in NEW_BLOCKS:
            key = {**old_block_key, "block_start": start, "block_end": end}
            info = (ephys.EphysBlockInfo & key).fetch(as_dict=True)
            if info:
                chunk_count = len(ephys.EphysBlockInfo.Chunk & key)
                print(f"  + {start} -> {end}: {info[0]['block_duration']:.1f}h, {chunk_count} chunks")
            else:
                print(f"  X {start} -> {end}: EphysBlockInfo not populated!")

    # ---------------------------------------------------------------
    # Step 6: Create SortingTasks
    # ---------------------------------------------------------------
    print("\n=== Step 6: Create SortingTasks ===")
    for i, (start, end) in enumerate(NEW_BLOCKS):
        label = chr(ord('a') + i)
        task_key = {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
            "insertion_number": INSERTION_NUMBER,
            "block_start": start,
            "block_end": end,
            "probe_type": PROBE_TYPE,
            "electrode_config_name": ELECTRODE_CONFIG_NAME,
            "electrode_group": ELECTRODE_GROUP_NAME,
            "paramset_id": PARAMSET_ID,
        }
        if args.dry_run:
            print(f"  > Would create SortingTask for block 3{label}")
        else:
            spike_sorting.SortingTask.insert1(task_key)
            print(f"  + SortingTask created for block 3{label}")

    # ---------------------------------------------------------------
    # Step 7: Run PreProcessing
    # ---------------------------------------------------------------
    print("\n=== Step 7: Run PreProcessing ===")
    if args.dry_run:
        print("  > Would run PreProcessing.populate() for 2 new blocks")
    else:
        print("  > Running PreProcessing.populate()...")
        spike_sorting.PreProcessing.populate(
            display_progress=True, suppress_errors=False
        )
        count = len(spike_sorting.PreProcessing & {"experiment_name": EXPERIMENT_NAME})
        print(f"  + PreProcessing entries total: {count}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n=== Summary ===")
    if args.dry_run:
        print("  DRY RUN — no changes made")
    else:
        total_blocks = len(ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME})
        total_tasks = len(spike_sorting.SortingTask & {"experiment_name": EXPERIMENT_NAME})
        total_preproc = len(spike_sorting.PreProcessing & {"experiment_name": EXPERIMENT_NAME})
        print(f"  EphysBlock:    {total_blocks} (was 7, now 8)")
        print(f"  SortingTask:   {total_tasks} (was 7, now 8)")
        print(f"  PreProcessing: {total_preproc} (was 7, now 8)")
    print("\n  Next: submit SLURM jobs for blocks 3a and 3b")
    print("  prod_spike_sorting.py has been updated with the 8-block schedule.")
    print("  prod_spike_sorting.sh has --array=3-4 (the two new blocks).")
    print("  Run: sbatch prod_spike_sorting.sh")
    print()
    print("  NOTE: This is a workaround for block 3 OOM. The 30h block was too")
    print("  large for Kilosort4 final clustering on a 40GB A100. Splitting into")
    print("  two 18h blocks means an extra manual curation boundary, but all 6h")
    print("  overlaps are preserved for unit matching.")


if __name__ == "__main__":
    main()
