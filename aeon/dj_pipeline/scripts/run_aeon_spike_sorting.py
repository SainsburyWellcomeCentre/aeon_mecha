"""
This is a one-off script to test run the spike sorting pipeline on a specific dataset (key).
Run script from the root directory of the aeon_mecha repository
Run in the conda environment with aeon_mecha (ephys-branch) installed

Change the value of `key` below if you want to run spike sorting on a different dataset.
"""

import datajoint as dj
from aeon.dj_pipeline import acquisition, ephys, spike_sorting

key = {'experiment_name': 'social-ephys0.1-aeon3',
       'probe': 'NP2004-001',
       'block_start': "2024-06-04 11:00:00",
       'block_end': "2024-06-10 12:00:00",
       'probe_type': 'neuropixels - NP2004',
       'electrode_config_name': '0-383',
       'electrode_group': '0-143',
       'paramset_id': '250'}
_CLEAR_JOB = True


def clear_job():
    """
    Clear this `key` from the jobs table (if any) to allow re-running the job 
    E.g. if this job previously failed with `error` status
    """
    (spike_sorting.schema.jobs & {"key_hash": dj.key_hash(key), "status": "error"}).delete()


def main():
    if _CLEAR_JOB:
        clear_job()
    spike_sorting.SpikeSorting.populate(key, reserve_jobs=True, display_progress=True)


if __name__ == '__main__':
    main()
