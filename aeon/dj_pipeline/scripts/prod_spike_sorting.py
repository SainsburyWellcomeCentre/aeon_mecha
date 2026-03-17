"""
Production spike sorting worker — run via SLURM.

Edit the `keys` list below for each SLURM job submission.
Submit ONE block at a time for 30h blocks (they're large).

Usage:
    # Edit keys below, then:
    sbatch prod_spike_sorting.sh
"""

import datajoint as dj
from aeon.dj_pipeline import acquisition, ephys, spike_sorting

# =============================================================================
# Configuration — EDIT for each SLURM job
# =============================================================================

# Which pipeline step to run
table_name = "SpikeSorting"  # "PreProcessing", "SpikeSorting", or "PostProcessing"

# Common key fields (shared across blocks)
_base = {
    "experiment_name": "social-ephys0.1-aeon3",
    "subject": "BAA-1104292",
    "insertion_number": 1,              # 1=ProbeA, 2=ProbeB
    "probe_type": "neuropixels - NP2004",
    "electrode_config_name": "0-383",
    "electrode_group": "0-95",
    "paramset_id": 1,                   # Production paramset
}

# --- Single block mode (recommended for 30h blocks) ---
# UPDATE the block_start and block_end for the block you want to sort
keys = [
    {**_base, "block_start": "EDIT-ME", "block_end": "EDIT-ME"},
]

# --- Example: all 7 blocks for ProbeA (insertion 1) ---
# Uncomment and edit after running Phase 0 reconnaissance:
#
# _start = "2024-06-04 11:00:00"  # UPDATE from Phase 0
# from datetime import datetime, timedelta
# import pandas as pd
# _s = pd.Timestamp(_start)
# keys = [
#     {**_base, "block_start": str(_s + pd.Timedelta(hours=24*i)),
#              "block_end": str(_s + pd.Timedelta(hours=24*i + 30))}
#     for i in range(7)
# ]

# =============================================================================

_CLEAR_JOB = True  # Clear any 'error' job state before re-running


def clear_job(dj_table, key):
    """Clear this key from the jobs table (if errored) to allow re-running."""
    (spike_sorting.schema.jobs & {
        "table_name": dj_table.table_name,
        "key_hash": dj.key_hash(key),
        "status": "error"}).delete()


def main():
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
    }[table_name]

    print(f"Running {table_name} for {len(keys)} block(s)")

    for i, key in enumerate(keys):
        print(f"\n=== Block {i+1}/{len(keys)}: {key['block_start']} - {key['block_end']} ===")
        print(f"    insertion_number={key['insertion_number']}")
        if _CLEAR_JOB:
            clear_job(populate_table, key)
        populate_table.populate(key, reserve_jobs=True, display_progress=True)
        print(f"=== Block {i+1} done ===")


if __name__ == "__main__":
    main()
