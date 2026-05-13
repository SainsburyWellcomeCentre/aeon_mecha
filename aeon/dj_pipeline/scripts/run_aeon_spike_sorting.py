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
