"""
Production spike sorting worker — run via SLURM.

Accepts --block N (1-indexed) to select which block to sort.
Designed for SLURM job arrays: sbatch --array=1-8 prod_spike_sorting.sh

Block schedule (after block 3 split):
  1: 2024-06-04 11:00 -> 2024-06-05 17:00 (30h)
  2: 2024-06-05 11:00 -> 2024-06-06 17:00 (30h)
  3: 2024-06-06 11:00 -> 2024-06-07 05:00 (18h)  <- 3a
  4: 2024-06-06 23:00 -> 2024-06-07 17:00 (18h)  <- 3b
  5: 2024-06-07 11:00 -> 2024-06-08 17:00 (30h)
  6: 2024-06-08 11:00 -> 2024-06-09 17:00 (30h)
  7: 2024-06-09 11:00 -> 2024-06-10 17:00 (30h)
  8: 2024-06-10 11:00 -> 2024-06-11 17:00 (30h)

Usage:
    # Single block:
    uv run python prod_spike_sorting.py --block 3
    # Via SLURM array:
    sbatch prod_spike_sorting.sh
"""

import argparse

import datajoint as dj
from aeon.dj_pipeline import acquisition, ephys, spike_sorting

# =============================================================================
# Configuration
# =============================================================================

# Which pipeline step to run
TABLE_NAME = "SpikeSorting"  # "PreProcessing", "SpikeSorting", or "PostProcessing"

# Explicit block schedule (after block 3 OOM split)
BLOCKS = [
    ("2024-06-04 11:00:00", "2024-06-05 17:00:00"),  # 1
    ("2024-06-05 11:00:00", "2024-06-06 17:00:00"),  # 2
    ("2024-06-06 11:00:00", "2024-06-07 05:00:00"),  # 3a
    ("2024-06-06 23:00:00", "2024-06-07 17:00:00"),  # 3b
    ("2024-06-07 11:00:00", "2024-06-08 17:00:00"),  # 4
    ("2024-06-08 11:00:00", "2024-06-09 17:00:00"),  # 5
    ("2024-06-09 11:00:00", "2024-06-10 17:00:00"),  # 6
    ("2024-06-10 11:00:00", "2024-06-11 17:00:00"),  # 7
]
N_BLOCKS = len(BLOCKS)

# Common key fields
BASE_KEY = {
    "experiment_name": "social-ephys0.1-aeon3",
    "subject": "BAA-1104292",
    "insertion_number": 1,              # 1=ProbeA, 2=ProbeB
    "probe_type": "neuropixels - NP2004",
    "electrode_config_name": "0-383",
    "electrode_group": "0-95",
    "paramset_id": 1,
}

CLEAR_JOB = True  # Clear any 'error' job state before re-running

# =============================================================================


def get_block_key(block_num):
    """Get block start/end from 1-indexed block number."""
    start, end = BLOCKS[block_num - 1]
    return {**BASE_KEY, "block_start": start, "block_end": end}


def clear_job(dj_table, key):
    """Clear this key from the jobs table (if errored) to allow re-running."""
    (spike_sorting.schema.jobs & {
        "table_name": dj_table.table_name,
        "key_hash": dj.key_hash(key),
        "status": "error"}).delete()


def main():
    parser = argparse.ArgumentParser(description="Production spike sorting worker")
    parser.add_argument("--block", type=int, required=True,
                        help=f"Block number (1-{N_BLOCKS})")
    parser.add_argument("--table", default=TABLE_NAME,
                        choices=["PreProcessing", "SpikeSorting", "PostProcessing"],
                        help=f"Pipeline step to run (default: {TABLE_NAME})")
    args = parser.parse_args()

    if not 1 <= args.block <= N_BLOCKS:
        raise ValueError(f"Block must be 1-{N_BLOCKS}, got {args.block}")

    # Safety: verify production
    host = dj.config.get("database.host", "")
    prefix = dj.config["custom"].get("database.prefix", "")
    print(f"Database: {host} / {prefix}")
    assert "aeon-db2" in host, f"Not production host! Got: {host}"
    assert prefix == "aeon_", f"Not production prefix! Got: {prefix}"

    populate_table = {
        "PreProcessing": spike_sorting.PreProcessing,
        "SpikeSorting": spike_sorting.SpikeSorting,
        "PostProcessing": spike_sorting.PostProcessing,
    }[args.table]

    key = get_block_key(args.block)
    print(f"Running {args.table} for block {args.block}/{N_BLOCKS}")
    print(f"  {key['block_start']} -> {key['block_end']}")
    print(f"  insertion_number={key['insertion_number']}")

    if CLEAR_JOB:
        clear_job(populate_table, key)
    populate_table.populate(key, reserve_jobs=True, display_progress=True)
    print(f"=== Block {args.block} done ===")


if __name__ == "__main__":
    main()
