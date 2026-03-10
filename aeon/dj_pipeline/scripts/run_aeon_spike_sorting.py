"""
This is a one-off script to test run the spike sorting pipeline on a specific dataset (key).
Run script from the root directory of the aeon_mecha repository

Change the value of `keys` below if you want to run spike sorting on a different dataset.
"""

import datajoint as dj
from aeon.dj_pipeline import acquisition, ephys, spike_sorting

# Specify which table to populate: "PreProcessing", "SpikeSorting", or "PostProcessing"
table_name = "SpikeSorting"

# Common key fields
_base = {'experiment_name': 'social-ephys0.1-aeon3',
         'subject': 'test-subject-001',
         'insertion_number': 1,
         'probe_type': 'neuropixels - NP2004',
         'electrode_config_name': '0-383',
         'electrode_group': '0-95',
         'paramset_id': 400}

# Remaining blocks (block 1 already sorted)
keys = [
    {**_base, 'block_start': "2024-06-04 13:00:00", 'block_end': "2024-06-04 16:00:00"},
    {**_base, 'block_start': "2024-06-04 15:00:00", 'block_end': "2024-06-04 18:00:00"},
    {**_base, 'block_start': "2024-06-04 17:00:00", 'block_end': "2024-06-04 20:00:00"},
]

_CLEAR_JOB = True  # Whether to clear any existing 'error' job for each key before populating


def clear_job(dj_table, key):
    """
    Clear this `key` from the jobs table (if any) to allow re-running the job
    E.g. if this job previously failed with `error` status
    """
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


if __name__ == '__main__':
    main()
