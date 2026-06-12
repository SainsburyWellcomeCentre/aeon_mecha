import os
import datajoint as dj
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
import json
import shutil
import numpy as np
import pandas as pd

from aeon.dj_pipeline import ephys, spike_sorting, get_schema_name
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir

schema = dj.Schema(get_schema_name("spike_sorting_curation"))
logger = dj.logger


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # method/package used to perform manual curation (e.g. SpikeInterface, Phy, FigURL, etc.)
    """
    contents = [
        ("Phy",),
        ("SpikeInterface",),
    ]


@schema
class ManualCuration(dj.Manual):
    definition = """
    # Manual curation from a SortedSpikes
    -> spike_sorting.SpikeSorting
    curation_id: int32
    ---
    curation_datetime: datetime    # UTC time when the curation was performed
    parent_curation_id=-1: int32   # if -1, this curation is based on the raw spike sorting results
    -> CurationMethod              # which method/package used for manual curation (inform how to ingest the results)
    description="": varchar(1000)  # user-defined description/note of the curation
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: <filepath@dj_store>
        """


@schema
class OfficialCuration(dj.Manual):
    definition = """  # One final/official curation for a SortedSpikes
    -> spike_sorting.SortedSpikes
    ---
    -> ManualCuration
    """


@schema
class ApplyOfficialCuration(dj.Imported):
    definition = """
    -> OfficialCuration
    ---
    execution_time: datetime        # datetime of the start of this step
    new_unit_count: int32           # number of new units added
    removed_unit_count: int32       # number of units removed
    """

    def make(self, key):
        """
        Update/overwrite the SortedSpikes (and downstream) with the official curation results.

        This function:
        1. Loads the curation JSON from ManualCuration
        2. Applies curation to the sorting analyzer using SpikeInterface
        3. Saves the curated analyzer in a separate folder
        4. Deletes old SortedSpikes and downstream tables
        5. Tracks unit count changes
        """
        import spikeinterface as si
        from spikeinterface.curation import apply_curation

        execution_time = datetime.now(UTC)

        # Get curation_id from OfficialCuration entry (it's in attributes, not primary key)
        # The key only contains SortedSpikes primary key fields, not attributes
        curation_id = (OfficialCuration & key).fetch1("curation_id")

        # Auto-approved curation: no manual curation file, based on raw sorting
        has_curation_file = bool(ManualCuration.File & key & {"curation_id": curation_id})
        if not has_curation_file:
            parent_curation_id = (ManualCuration & key & {"curation_id": curation_id}).fetch1(
                "parent_curation_id"
            )
            if parent_curation_id == -1:
                # Raw sorting approved as official — no curation to apply
                # Update SortedSpikes.curation_id from -1 to the official curation_id
                sorted_key = (spike_sorting.SortedSpikes & key).fetch1("KEY")
                spike_sorting.SortedSpikes.update1({**sorted_key, "curation_id": curation_id})

                self.insert1({
                    **key,
                    "execution_time": execution_time,
                    "new_unit_count": 0,
                    "removed_unit_count": 0,
                })
                logger.info(
                    f"Auto-approved curation (curation_id={curation_id}): "
                    "raw sorting results accepted as official. No changes applied."
                )
                return

        # Get the curation file path
        curation_file_path = Path(
            (ManualCuration.File & key & {"curation_id": curation_id}).fetch1("file").full_path
        )

        if not curation_file_path.exists():
            raise FileNotFoundError(
                f"Curation file not found: {curation_file_path}\n"
                f"Please verify the file exists and is accessible."
            )

        # Load curation dictionary
        with open(curation_file_path, "r") as f:
            curation_dict = json.load(f)

        # Get sorting output directory
        sorting_root_dir = get_sorting_root_dir()
        output_dir = sorting_root_dir / (spike_sorting.PreProcessing & key).fetch1(
            "sorting_output_dir"
        )

        # Load original sorting analyzer
        analyzer_output_dir = spike_sorting._resolve_analyzer_dir(output_dir)
        if not analyzer_output_dir.exists():
            raise FileNotFoundError(
                f"Sorting analyzer directory not found: {analyzer_output_dir}"
            )

        sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_output_dir)

        # Track original unit IDs for comparison
        original_unit_ids = set(sorting_analyzer.sorting.unit_ids)

        # Apply curation directly to the analyzer
        # This returns a new SortingAnalyzer in memory with curation applied
        # Extensions are preserved/lazily recomputed for unchanged units
        logger.info(f"Applying curation (curation_id={curation_id}) to analyzer...")
        curated_analyzer = apply_curation(
            sorting_analyzer,
            curation_dict,
            merging_mode="soft",  # Use soft merging to avoid full recomputation
        )

        # Track new and removed units
        curated_unit_ids = set(curated_analyzer.sorting.unit_ids)
        new_unit_count = len(curated_unit_ids - original_unit_ids)
        removed_unit_count = len(original_unit_ids - curated_unit_ids)

        # Save curated analyzer to dedicated folder
        # Keep raw analyzer in sorting_analyzer, curated in sorting_analyzer_curated_id{curation_id}
        curated_analyzer_dir = output_dir / f"sorting_analyzer_curated_id{curation_id}"
        logger.info(f"Saving curated analyzer to: {curated_analyzer_dir}")
        params = (spike_sorting.SortingParamSet & key).fetch1("params")
        save_format = params.get("save_format", "zarr")
        if save_format == "zarr":
            if curated_analyzer_dir.exists():
                shutil.rmtree(curated_analyzer_dir)
            curated_analyzer.save_as(format="zarr", folder=curated_analyzer_dir)
        else:
            curated_analyzer.save(folder=curated_analyzer_dir, overwrite=True)
        
        # Store the applied analyzer directory path in ManualCuration.File
        analyzer_file_entry = {
            **key,
            "curation_id": curation_id,
            "file_name": "curation_applied_analyzer",
            "file": curated_analyzer_dir,
        }
        # Insert or replace (can happen if user applies, reverts, then re-applies same curation)
        ManualCuration.File.insert1(analyzer_file_entry, replace=True)
        logger.debug(f"Stored applied analyzer path in database for curation_id={curation_id}")

        # Check current curation state in SortedSpikes
        # There can only be one SortedSpikes entry per primary key (curation_id is an attribute, not part of PK)
        current_sorted_spikes = spike_sorting.SortedSpikes & key
        if current_sorted_spikes:
            current_curation_id = current_sorted_spikes.fetch1("curation_id")
        else:
            current_curation_id = None

        # Handle different scenarios based on current curation state
        if current_curation_id == curation_id:
            # This curation is already applied
            logger.info(
                f"Curation (curation_id={curation_id}) is already applied to this block. "
                "No action needed. If you need to reprocess, restore the raw uncurated version first."
            )
            return

        if current_curation_id is not None and current_curation_id != -1:
            # A different curation is already applied
            raise ValueError(
                f"A different curation (curation_id={current_curation_id}) has already been applied to this block. "
                f"Cannot apply curation_id={curation_id}. "
                "If you want to apply a new curation, the data needs to be reverted to the uncurated version first."
            )

        # Current state: curation_id == -1 or no entry exists
        # Proceed with applying the curation

        # Delete existing SortedSpikes entry (curation_id=-1) if it exists
        # DataJoint will automatically delete downstream tables
        # NOTE: No unit matching cleanup needed here. UnitMatching requires
        # ApplyOfficialCuration to exist (via key_source), so when we're first
        # applying a curation to replace raw sorting (curation_id=-1), there's
        # no unit matching data referencing these units yet. If a user is
        # re-curating after undoing a previous curation, restore_raw_sorting()
        # handles the unit matching cleanup before we get here.
        if current_curation_id == -1:
            logger.info(
                "Deleting old SortedSpikes (curation_id=-1) and downstream tables..."
            )
            (spike_sorting.SortedSpikes & key).delete(safemode=False)
            logger.info(
                "Deleted SortedSpikes (downstream tables auto-deleted by DataJoint)"
            )

        # Insert ApplyOfficialCuration entry
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "new_unit_count": new_unit_count,
                "removed_unit_count": removed_unit_count,
            }
        )

        logger.info(
            f"Successfully applied official curation (curation_id={curation_id})\n"
            f"  New units: {new_unit_count}\n"
            f"  Removed units: {removed_unit_count}\n"
            f"  Curated analyzer saved to: {curated_analyzer_dir}\n"
            f"  Raw analyzer remains in: {output_dir / 'sorting_analyzer'}"
        )

        logger.info("Run .populate() on spike_sorting.SortedSpikes and downstream tables to load the curated sorting results into the pipeline.")


# Helper functions for curation workflows


def _get_analyzer_dir_from_key(key: dict) -> Path:
    """
    Construct the path to the sorting_analyzer directory from a key using database paths.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id

    Returns:
        Path to the sorting_analyzer directory.
    """
    sorting_root_dir = get_sorting_root_dir()
    output_dir = sorting_root_dir / (spike_sorting.PreProcessing & key).fetch1("sorting_output_dir")
    return spike_sorting._resolve_analyzer_dir(output_dir)


def launch_spikeinterface_gui(
    key: dict, parent_curation_id: Optional[int] = None
) -> None:
    """
    Launch SpikeInterface GUI for manual spike sorting curation.

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
    import spikeinterface as si
    import shutil

    analyzer_dir = _get_analyzer_dir_from_key(key)

    if not analyzer_dir.exists():
        raise FileNotFoundError(
            f"Sorting analyzer directory not found: {analyzer_dir}\n"
            f"Please verify the key is correct and that PreProcessing has been run for this block."
        )

    # Handle parent curation if specified
    gui_dir = analyzer_dir / "spikeinterface_gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    curation_data_file = gui_dir / "curation_data.json"
    metadata_file = gui_dir / "curation_metadata.json"

    if parent_curation_id is not None:
        # Get the parent curation file
        parent_curation_key = {**key, "curation_id": parent_curation_id}
        if not (ManualCuration & parent_curation_key):
            raise ValueError(
                f"Parent curation with curation_id={parent_curation_id} not found "
                f"for this sorting task."
            )

        # Get the parent curation file path
        parent_file = Path(
            (ManualCuration.File & parent_curation_key).fetch1("file").full_path
        )

        if not parent_file.exists():
            raise FileNotFoundError(
                f"Parent curation file not found: {parent_file}\n"
                f"Please verify the file exists and is accessible from your local mount."
            )

        # Check if curation_data.json already exists
        if curation_data_file.exists():
            logger.warning(
                f"WARNING: curation_data.json already exists at {curation_data_file}. "
                "This file will be OVERWRITTEN if you proceed with loading the parent curation. "
                "Please either finish and save the current curation using save_curation.py, "
                "or delete the curation_data.json file manually."
            )
            return

        # Copy parent curation file to curation_data.json
        logger.info(f"Loading parent curation (curation_id={parent_curation_id})...")
        shutil.copy2(parent_file, curation_data_file)
        logger.info(f"Copied parent curation to: {curation_data_file}")

        # Save parent_curation_id to metadata file for later use in save_manual_curation()
        with open(metadata_file, "w") as f:
            json.dump({"parent_curation_id": parent_curation_id}, f)
        logger.info(f"Saved parent curation metadata to: {metadata_file}")

    # Handle metadata file when parent_curation_id is None
    if parent_curation_id is None:
        # Delete metadata file if it exists (clearing any previous parent)
        if metadata_file.exists():
            metadata_file.unlink()
            logger.info("Cleared previous parent curation metadata (starting from raw)")

    # Check for existing curation_data.json file (if not loading from parent)
    if curation_data_file.exists() and parent_curation_id is None:
        file_mtime = datetime.fromtimestamp(curation_data_file.stat().st_mtime, tz=UTC)
        time_since_modification = (datetime.now(UTC) - file_mtime).total_seconds()
        days_ago = time_since_modification / 86400

        logger.warning(
            f"Existing curation data found at {curation_data_file}. "
            f"Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S UTC')} "
            f"({days_ago:.1f} days ago). "
            "This curation has NOT been saved to a curation_id in the ManualCuration table yet. "
            "This is fine if you are picking up where you left off, but be wary of saving over "
            "the curation if it is from another user. NOTE: Clicking 'Save in analyzer' in the "
            "GUI will OVERWRITE the existing curation_data.json file with your new additions."
        )

    # Load sorting analyzer
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_dir)

    # Launch GUI
    # Try spikeinterface_gui first, fall back to built-in viewer if available
    try:
        from spikeinterface_gui import run_mainwindow

        run_mainwindow(sorting_analyzer, mode="desktop", curation=True)
    except ImportError:
        # Fallback to built-in viewer if spikeinterface_gui is not available
        si.view_sorting_analyzer(sorting_analyzer)

    # Remind user to save curation after GUI closes
    logger.info(
        "GUI closed. Don't forget to run save_curation.py to save your curation "
        "to the ManualCuration table with a unique curation_id."
    )


def save_manual_curation(key: dict, description: str = "") -> int:
    """
    Save manual curation results from SpikeInterface GUI to the database.

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
    analyzer_dir = _get_analyzer_dir_from_key(key)

    # Path to curation_data.json file
    curation_data_file = analyzer_dir / "spikeinterface_gui" / "curation_data.json"

    if not curation_data_file.exists():
        raise FileNotFoundError(
            f"Curation data file not found: {curation_data_file}\n"
            f"Please ensure you have saved your curation in the SI GUI using the 'Save in analyzer' button."
        )

    # Find the next available curation_id
    existing_ids = (ManualCuration & key).to_arrays("curation_id")
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
    logger.info(f"Deleted original curation_data.json (saved as {curated_file_name})")

    # Read parent_curation_id from metadata file if it exists
    metadata_file = curation_data_file.parent / "curation_metadata.json"
    parent_curation_id = -1  # Default: based on raw sorting results
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                parent_curation_id = metadata.get("parent_curation_id", -1)
            logger.info(f"Using parent_curation_id={parent_curation_id} from metadata file")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to read parent_curation_id from metadata file: {e}. "
                "Using default parent_curation_id=-1"
            )
            parent_curation_id = -1

    # Prepare ManualCuration entry
    curation_datetime = datetime.now(UTC)
    curation_entry = {
        **key,
        "curation_id": next_curation_id,
        "curation_datetime": curation_datetime,
        "parent_curation_id": parent_curation_id,
        "curation_method": "SpikeInterface",
        "description": description,
    }

    # Insert into ManualCuration table
    ManualCuration.insert1(curation_entry)

    # Insert the curated file into ManualCuration.File
    file_entry = {
        **key,
        "curation_id": next_curation_id,
        "file_name": curated_file_name,
        "file": curated_file_path,
    }
    ManualCuration.File.insert1(file_entry)

    # Clean up metadata file after successful save (metadata is now in database)
    if metadata_file.exists():
        metadata_file.unlink()
        logger.info("Deleted curation metadata file (metadata now in database)")

    logger.info(f"Successfully saved curation with curation_id={next_curation_id}")
    logger.info(f"Curation file saved as: {curated_file_path}")

    return next_curation_id


def make_curation_official(key: dict, curation_id: int) -> None:
    """
    Designate an existing curation as official.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - block_start (datetime or string)
            - block_end (datetime or string)
            - electrode_group
            - paramset_id
        curation_id: The curation_id of the ManualCuration entry to make official.
    """
    # Verify the curation exists
    curation_key = {**key, "curation_id": curation_id}
    if not (ManualCuration & curation_key):
        raise ValueError(
            f"Curation with curation_id={curation_id} not found for this sorting task."
        )

    # Get the SortedSpikes key (need to find the one with curation_id=-1, the raw sorting)
    sorted_spikes_key = (spike_sorting.SortedSpikes & key & {"curation_id": -1}).fetch1(
        "KEY"
    )

    # Check if OfficialCuration already exists for this SortedSpikes
    if OfficialCuration & sorted_spikes_key:
        existing_curation = (OfficialCuration & sorted_spikes_key).fetch1()
        if existing_curation["curation_id"] != curation_id:
            raise ValueError(
                f"An official curation already exists for this block "
                f"(curation_id={existing_curation['curation_id']}). "
                f"Please remove it first if you want to set a different one."
            )
        else:
            logger.info(f"Official curation with curation_id={curation_id} already exists.")
            return

    # Create OfficialCuration entry
    # sorted_spikes_key already contains SpikeSorting fields (through inheritance)
    official_curation_entry = {
        **sorted_spikes_key,
        "curation_id": curation_id,
    }
    OfficialCuration.insert1(official_curation_entry, skip_duplicates=True)

    logger.info(f"Created OfficialCuration entry for curation_id={curation_id}")
    logger.info("Run ApplyOfficialCuration.populate() to apply the curation to the analyzer.")


def restore_raw_sorting(key: dict) -> None:
    """Restore raw (uncurated) sorting by removing official curation.

    This function:
    1. Deletes the OfficialCuration entry (which cascades to ApplyOfficialCuration)
    2. Deletes UnitMatching for this block (cascades to .Unit and .Spikes Part rows)
    3. Deletes orphaned GlobalUnit entries (those with no remaining UnitMatching.Unit references)
    4. Deletes the curated SortedSpikes entry (which cascades to downstream tables)

    After restoring, re-run SortedSpikes.populate() + SyncedSpikes.populate() to load
    the new curation, then re-run UnitMatching.populate() to re-match units.

    Args:
        key: Dictionary key identifying the sorting task. Must contain:
            - experiment_name
            - subject
            - insertion_number
            - block_start (datetime or string)
            - block_end (datetime or string)
            - probe_type
            - electrode_config_name
            - electrode_group
            - paramset_id
    """
    # Check if there's an official curation
    official_curation = OfficialCuration & key
    if not official_curation:
        logger.info("No official curation found. Data is already in uncurated form.")
        return

    curation_id = official_curation.fetch1("curation_id")
    logger.info(f"Restoring raw sorting (removing official curation_id={curation_id})...")

    # Step 1: Delete OfficialCuration entry (cascades to ApplyOfficialCuration)
    logger.info("Deleting OfficialCuration entry...")
    official_curation.delete(safemode=False)
    logger.info("OfficialCuration and ApplyOfficialCuration entries deleted.")

    # Step 2: Delete UnitMatching for this block
    # Cascades to UnitMatching.Unit and UnitMatching.Spikes Part rows
    unit_matching_entries = spike_sorting.UnitMatching & key
    if unit_matching_entries:
        n_um = len(unit_matching_entries)
        logger.info(f"Deleting {n_um} UnitMatching entries for this block...")
        unit_matching_entries.delete(safemode=False)
        logger.info("UnitMatching entries deleted (cascaded to Unit and Spikes parts).")

    # Step 3: Delete orphaned GlobalUnit entries
    # (those with no remaining UnitMatching.Unit references from any block)
    insertion_key = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}
    n_orphans = 0
    for gu_key in (spike_sorting.GlobalUnit & insertion_key).keys():
        if len(spike_sorting.UnitMatching.Unit & gu_key) == 0:
            logger.info(f"Deleting orphaned GlobalUnit {gu_key['global_unit']}...")
            (spike_sorting.GlobalUnit & gu_key).delete(safemode=False)
            n_orphans += 1
    if n_orphans:
        logger.info(f"Deleted {n_orphans} orphaned GlobalUnit entries.")

    # Step 4: Delete the SortedSpikes entry (cascades to downstream tables)
    sorted_spikes_entry = spike_sorting.SortedSpikes & key
    if sorted_spikes_entry:
        logger.info("Deleting SortedSpikes entry and downstream tables...")
        sorted_spikes_entry.delete(safemode=False)
        logger.info(
            "SortedSpikes and downstream tables deleted.\n"
            "Next steps:\n"
            "  1. Run SortedSpikes.populate() to load the raw sorting results\n"
            "  2. Run SyncedSpikes.populate() to sync spike times\n"
            "  3. Apply new curation if needed (make_curation_official + ApplyOfficialCuration.populate)\n"
            "  4. Run UnitMatching.populate() to re-match units and generate Spikes"
        )
    else:
        logger.info("No SortedSpikes entry found (run populate to load the raw sorting results back into the pipeline).")
