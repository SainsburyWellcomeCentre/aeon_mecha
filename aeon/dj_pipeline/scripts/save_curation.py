"""
Script to save manual curation results from SpikeInterface GUI to the database.

================================================================================
IMPORTANT: THIS SCRIPT IS DESIGNED TO BE MODIFIED BEFORE RUNNING
================================================================================

This script is intended for interactive use. Before running:

1. Open this file in your IDE/editor
2. Modify the 'key' dictionary in the __main__ section (lines ~90-96) with your
   specific session parameters:
   - experiment_name: Your experiment identifier
   - block_start: Start datetime of the block (string or datetime)
   - block_end: End datetime of the block (string or datetime)
   - electrode_group: Electrode group identifier (e.g., "0-143")
   - paramset_id: Parameter set ID (e.g., "250")
3. Uncomment and configure the desired operation (save, make official, restore, etc.)
4. Run the script from your IDE (not from command line)

This workflow is designed for interactive, exploratory use where you modify the
parameters for each session you want to work with. For automated/scripted use,
consider calling the helper functions directly from your own code.

Available operations (uncomment the one you need):
    - save_curation(): Save curation without making it official
    - make_official_curation(): Make an existing curation official
    - save_and_make_official(): Save and immediately make official
    - restore_raw_sorting(): Restore raw (uncurated) sorting
"""

from aeon.dj_pipeline import spike_sorting_curation

if __name__ == "__main__":
    # Example key - modify these values for your session
    # The key must contain:
    #   - experiment_name: Your experiment identifier
    #   - block_start: Start datetime of the block (string or datetime)
    #   - block_end: End datetime of the block (string or datetime)
    #   - electrode_group: Electrode group identifier (e.g., "0-143")
    #   - paramset_id: Parameter set ID (e.g., "250")
    key = {
        "experiment_name": "social-ephys0.1-aeon3",
        "block_start": "2024-06-04 11:00:00",
        "block_end": "2024-06-10 12:00:00",
        "electrode_group": "0-143",
        "paramset_id": "250",
    }

    # Optional description for this curation
    description = ""

    # Option 1: Save curation only (without making it official)
    # This saves the curation to the database and returns the curation_id.
    # Uncomment the line below to use this option:
    curation_id = spike_sorting_curation.save_manual_curation(key, description=description)

    # Option 2: Make an existing curation official and apply it
    # This designates an existing curation as official, which triggers the worker manager
    # to apply it to the database (replacing raw sorting data with curated data).
    # Uncomment the lines below to use this option:
    # curation_id = 1  # The curation_id of the ManualCuration entry to make official
    # spike_sorting_curation.make_curation_official(key, curation_id)

    # Option 3: Save curation and immediately make it official and apply it
    # This combines Option 1 and Option 2 - saves the curation and makes it official in one step.
    # Uncomment the lines below to use this option:
    # curation_id = spike_sorting_curation.save_manual_curation(key, description=description)
    # spike_sorting_curation.make_curation_official(key, curation_id)

    # Option 4: Restore raw (uncurated) sorting
    # This removes the official curation designation, allowing the system to use raw sorting data again.
    # Uncomment the line below to use this option:
    # spike_sorting_curation.restore_raw_sorting(key)
