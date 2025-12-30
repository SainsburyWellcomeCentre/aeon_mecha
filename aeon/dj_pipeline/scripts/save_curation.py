"""
Script to save manual curation results from SpikeInterface GUI to the database.

This script reads the curation_data.json file created by the SI GUI and saves it
as a new curation entry in the ManualCuration table with a unique curation_id.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime, UTC
import shutil
import json

from aeon.dj_pipeline import spike_sorting, spike_sorting_curation


def save_curation(
    key: Dict[str, Any], local_root_dir: Path, description: str = ""
) -> int:
    """
    Save manual curation results from SpikeInterface GUI to the database.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        local_root_dir: Path to the local mount of the server volume where
            the sorting data is stored. This should point to the root directory
            that contains experiment folders. Can be a string or Path object.
        description: Optional description/note for this curation.
    """
    # Convert to Path if string provided
    local_root_dir = Path(local_root_dir)

    # Get sorting method from database
    sorting_method = (
        spike_sorting.SortingMethod * spike_sorting.SortingParamSet & key
    ).fetch1("sorting_method")

    # Format block start/end times
    block_start = key["block_start"]
    block_end = key["block_end"]
    if isinstance(block_start, str):
        block_start = datetime.fromisoformat(block_start.replace(" ", "T"))
    if isinstance(block_end, str):
        block_end = datetime.fromisoformat(block_end.replace(" ", "T"))

    start_str = block_start.strftime("%Y-%m-%dT%H-%M-%S")
    end_str = block_end.strftime("%Y-%m-%dT%H-%M-%S")

    # Construct path to sorting_analyzer directory
    analyzer_dir = (
        local_root_dir
        / key["experiment_name"]
        / "ephys_blocks"
        / f"{start_str}_{end_str}"
        / key["electrode_group"]
        / f"{sorting_method}_{key['paramset_id']}"
        / "sorting_analyzer"
    )

    # Path to curation_data.json file
    curation_data_file = analyzer_dir / "spikeinterface_gui" / "curation_data.json"

    if not curation_data_file.exists():
        raise FileNotFoundError(
            f"Curation data file not found: {curation_data_file}\n"
            f"Please ensure you have saved your curation in the SI GUI using the 'Save in analyzer' button."
        )

    # Get the SpikeSorting key (need to fetch it from the database)
    # The key should match what's in SpikeSorting table
    spike_sorting_key = (spike_sorting.SpikeSorting & key).fetch1("KEY")

    # Find the next available curation_id
    existing_ids = (spike_sorting_curation.ManualCuration & spike_sorting_key).fetch(
        "curation_id"
    )
    next_curation_id = max(existing_ids) + 1 if len(existing_ids) > 0 else 1

    # Copy curation_data.json with curation_id suffix
    curated_file_name = f"curation_data_id{next_curation_id}.json"
    curated_file_path = curation_data_file.parent / curated_file_name

    # Copy the file first (as a safety measure)
    shutil.copy2(curation_data_file, curated_file_path)

    # Verify the copy was successful before deleting the original
    if not curated_file_path.exists():
        raise RuntimeError(
            f"Failed to copy curation file. Original file preserved at: {curation_data_file}"
        )

    # Verify the copied file is valid JSON
    try:
        with open(curated_file_path, "r") as f:
            json.load(f)  # Verify it's valid JSON
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Copied curation file is not valid JSON: {e}\n"
            f"Original file preserved at: {curation_data_file}"
        )

    # Now safe to delete the original file
    curation_data_file.unlink()
    print(f"Deleted original curation_data.json (saved as {curated_file_name})")

    # Prepare ManualCuration entry
    curation_datetime = datetime.now(UTC)
    curation_entry = {
        **spike_sorting_key,
        "curation_id": next_curation_id,
        "curation_datetime": curation_datetime,
        "parent_curation_id": -1,  # Based on raw sorting results
        "curation_method": "SpikeInterface",
        "description": description,
    }

    # Insert into ManualCuration table
    spike_sorting_curation.ManualCuration.insert1(curation_entry)

    # Insert the curated file into ManualCuration.File
    file_entry = {
        **spike_sorting_key,
        "curation_id": next_curation_id,
        "file_name": curated_file_name,
        "file": curated_file_path,
    }
    spike_sorting_curation.ManualCuration.File.insert1(file_entry)

    print(f"Successfully saved curation with curation_id={next_curation_id}")
    print(f"Curation file saved as: {curated_file_path}")

    return next_curation_id


def make_official_curation(
    key: Dict[str, Any], curation_id: int, local_root_dir: Path
) -> None:
    """
    Designate an existing curation as official and apply it to update the database.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        curation_id: The curation_id of the ManualCuration entry to make official.
        local_root_dir: Path to the local mount of the server volume where
            the sorting data is stored. This should point to the root directory
            that contains experiment folders. Can be a string or Path object.
    """
    # Get the SpikeSorting key
    spike_sorting_key = (spike_sorting.SpikeSorting & key).fetch1("KEY")

    # Verify the curation exists
    curation_key = {**spike_sorting_key, "curation_id": curation_id}
    if not (spike_sorting_curation.ManualCuration & curation_key):
        raise ValueError(
            f"Curation with curation_id={curation_id} not found for this sorting task."
        )

    # Get the SortedSpikes key (need to find the one with curation_id=-1, the raw sorting)
    sorted_spikes_key = (spike_sorting.SortedSpikes & key & {"curation_id": -1}).fetch1(
        "KEY"
    )

    # Check if OfficialCuration already exists for this SortedSpikes
    if spike_sorting_curation.OfficialCuration & sorted_spikes_key:
        existing_curation = (
            spike_sorting_curation.OfficialCuration & sorted_spikes_key
        ).fetch1()
        if existing_curation["curation_id"] != curation_id:
            raise ValueError(
                f"An official curation already exists for this session "
                f"(curation_id={existing_curation['curation_id']}). "
                f"Please remove it first if you want to set a different one."
            )
        else:
            print(f"Official curation with curation_id={curation_id} already exists.")
            print("Triggering ApplyOfficialCuration.populate()...")
            spike_sorting_curation.ApplyOfficialCuration.populate(sorted_spikes_key)
            return

    # Create OfficialCuration entry
    official_curation_entry = {
        **sorted_spikes_key,
        **spike_sorting_key,
        "curation_id": curation_id,
    }
    spike_sorting_curation.OfficialCuration.insert1(
        official_curation_entry, skip_duplicates=True
    )

    print(f"Created OfficialCuration entry for curation_id={curation_id}")
    print("Triggering ApplyOfficialCuration.populate()...")

    # Trigger the application
    spike_sorting_curation.ApplyOfficialCuration.populate(sorted_spikes_key)


def save_and_make_official(
    key: Dict[str, Any], local_root_dir: Path, description: str = ""
) -> None:
    """
    Save curation and immediately make it official and apply it.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        local_root_dir: Path to the local mount of the server volume where
            the sorting data is stored. This should point to the root directory
            that contains experiment folders. Can be a string or Path object.
        description: Optional description/note for this curation.
    """
    # Save the curation first
    curation_id = save_curation(key, local_root_dir, description=description)

    # Make it official and apply it
    make_official_curation(key, curation_id, local_root_dir)


if __name__ == "__main__":
    # Example key - modify these values for your session
    key = {
        "experiment_name": "social-ephys0.1-aeon3",
        "block_start": "2024-06-04 11:00:00",
        "block_end": "2024-06-10 12:00:00",
        "electrode_group": "0-143",
        "paramset_id": "250",
    }

    # Example local root directory - modify to point to your local mount
    local_root_dir = Path("/path/to/local/mount/of/server/volume")

    # Optional description for this curation
    description = ""

    # Option 1: Save curation only (without making it official)
    # Uncomment the line below to use this option:
    # save_curation(key, local_root_dir, description=description)

    # Option 2: Make an existing curation official and apply it
    # Uncomment the lines below to use this option:
    # curation_id = 1  # The curation_id of the ManualCuration entry to make official
    # make_official_curation(key, curation_id, local_root_dir)

    # Option 3: Save curation and immediately make it official and apply it
    # Uncomment the line below to use this option:
    # save_and_make_official(key, local_root_dir, description=description)
