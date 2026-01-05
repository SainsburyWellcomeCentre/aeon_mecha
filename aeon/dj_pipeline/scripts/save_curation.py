"""
Script to save manual curation results from SpikeInterface GUI to the database.

This script demonstrates how to use the helper functions from the spike_sorting_curation module
to save curations, make them official, and restore raw sorting.
"""

from typing import Dict, Any
from aeon.dj_pipeline import spike_sorting_curation


def save_curation(key: Dict[str, Any], description: str = "") -> int:
    """
    Save manual curation results from SpikeInterface GUI to the database.

    This is a wrapper around spike_sorting_curation.save_manual_curation().

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        description: Optional description/note for this curation.

    Returns:
        The curation_id assigned to the saved curation.
    """
    return spike_sorting_curation.save_manual_curation(key, description)


def make_official_curation(key: Dict[str, Any], curation_id: int) -> None:
    """
    Designate an existing curation as official.

    This is a wrapper around spike_sorting_curation.make_curation_official().

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        curation_id: The curation_id of the ManualCuration entry to make official.
    """
    spike_sorting_curation.make_curation_official(key, curation_id)


def save_and_make_official(key: Dict[str, Any], description: str = "") -> None:
    """
    Save curation and immediately make it official.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        description: Optional description/note for this curation.
    """
    # Save the curation first
    curation_id = save_curation(key, description=description)

    # Make it official
    make_official_curation(key, curation_id)


def restore_raw_sorting(key: Dict[str, Any]) -> None:
    """
    Restore raw (uncurated) sorting by removing official curation.

    This is a wrapper around spike_sorting_curation.restore_raw_sorting().

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
    """
    spike_sorting_curation.restore_raw_sorting(key)


if __name__ == "__main__":
    # Example key - modify these values for your session
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
    # Uncomment the line below to use this option:
    # save_curation(key, description=description)

    # Option 2: Make an existing curation official and apply it
    # Uncomment the lines below to use this option:
    # curation_id = 1  # The curation_id of the ManualCuration entry to make official
    # make_official_curation(key, curation_id)

    # Option 3: Save curation and immediately make it official and apply it
    # Uncomment the line below to use this option:
    # save_and_make_official(key, description=description)

    # Option 4: Restore raw (uncurated) sorting
    # Uncomment the line below to use this option:
    # restore_raw_sorting(key)
