"""
Run spike sorting pipeline on specified blocks.

Edit the `keys` list below for your experiment/blocks, then submit via SLURM:
    sbatch run_aeon_spike_sorting.sh

Change `table_name` to control which step to run:
    "PreProcessing", "SpikeSorting", or "PostProcessing"
"""

import datajoint as dj
from aeon.dj_pipeline import acquisition, ephys, spike_sorting

# =============================================================================
# Configuration — edit these for your experiment
# =============================================================================

# Which pipeline step to run
table_name = "SpikeSorting"

# Common key fields (shared across all blocks)
_base = {
    "experiment_name": "social-ephys0.1-aeon3",
    "subject": "test-subject-001",
    "insertion_number": 1,
    "probe_type": "neuropixels - NP2004",
    "electrode_config_name": "0-383",
    "electrode_group": "0-95",
    "paramset_id": 400,
}

# --- Single block mode: one block per SLURM job (use for large/long blocks) ---
keys = [
    {**_base, "block_start": "2024-06-04 11:00:00", "block_end": "2024-06-04 14:00:00"},
]

# --- Multi block mode: multiple blocks in one SLURM job (use for small/short blocks) ---
# keys = [
#     {**_base, "block_start": "2024-06-04 11:00:00", "block_end": "2024-06-04 14:00:00"},
#     {**_base, "block_start": "2024-06-04 13:00:00", "block_end": "2024-06-04 16:00:00"},
#     {**_base, "block_start": "2024-06-04 15:00:00", "block_end": "2024-06-04 18:00:00"},
#     {**_base, "block_start": "2024-06-04 17:00:00", "block_end": "2024-06-04 20:00:00"},
# ]

# =============================================================================

_CLEAR_JOB = True  # Clear any existing 'error' job for each key before populating


def clear_job(dj_table, key):
    """Clear this key from the jobs table (if errored) to allow re-running."""
    (spike_sorting.schema.jobs & {
        "table_name": dj_table.table_name,
        "key_hash": dj.key_hash(key),
        "status": "error"}).delete()


def main():
    populate_table = {
        "PreProcessing": spike_sorting.PreProcessing,
        "SpikeSorting": spike_sorting.SpikeSorting,
        "PostProcessing": spike_sorting.PostProcessing,
    }[table_name]

    for i, key in enumerate(keys):
        print(f"\n=== Block {i+1}/{len(keys)}: {key['block_start']} - {key['block_end']} ===")
        if _CLEAR_JOB:
            clear_job(populate_table, key)
        populate_table.populate(key, reserve_jobs=True, display_progress=True)
        print(f"=== Block {i+1} done ===")


if __name__ == "__main__":
    main()
