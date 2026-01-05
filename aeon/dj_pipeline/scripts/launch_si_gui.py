"""
Script to launch SpikeInterface GUI for manual spike sorting curation.

This script demonstrates how to use the launch_spikeinterface_gui helper function
from the spike_sorting_curation module.
"""

from typing import Dict, Any, Optional
from aeon.dj_pipeline import spike_sorting_curation


def launch_si_gui(
    key: Dict[str, Any], parent_curation_id: Optional[int] = None
) -> None:
    """
    Launch SpikeInterface GUI for manual spike sorting curation.

    This is a wrapper around spike_sorting_curation.launch_spikeinterface_gui().

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        parent_curation_id: Optional curation_id to base this curation on. If provided,
            the curation_data.json file will be initialized from the specified curation.
            If None, starts from the raw sorting results.
    """
    spike_sorting_curation.launch_spikeinterface_gui(key, parent_curation_id)


if __name__ == "__main__":
    # Example key - modify these values for your session
    key = {
        "experiment_name": "social-ephys0.1-aeon3",
        "block_start": "2024-06-04 11:00:00",
        "block_end": "2024-06-10 12:00:00",
        "electrode_group": "0-143",
        "paramset_id": "250",
    }

    launch_si_gui(key)

    # IMPORTANT: While using the SpikeInterface GUI for manual curation,
    # remember to periodically click the "Save in analyzer" button to save your work.
    # This ensures your curation changes are preserved in the sorting_analyzer directory.
    #
    # NOTE: If multiple users are curating the same session simultaneously, be aware that
    # saving in the GUI will overwrite the curation_data.json file. To preserve your curation
    # permanently, run save_curation.py immediately after saving in the GUI.
