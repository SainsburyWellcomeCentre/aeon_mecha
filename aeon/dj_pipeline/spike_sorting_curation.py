import os
import datajoint as dj
from datetime import datetime, UTC
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

from aeon.dj_pipeline import ephys, spike_sorting, get_schema_name
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir

# schema = dj.schema(get_schema_name("spike_sorting_curation"))
schema = dj.Schema()
logger = dj.logger


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # method/package used to perform manual curation (e.g. SpikeInterface, Phy, FigURL, etc.)
    """
    contents = [
        ("Phy", "SpikeInterface"),
    ]


@schema
class ManualCuration(dj.Manual):
    definition = """
    # Manual curation from a SortedSpikes
    -> spike_sorting.SpikeSorting
    curation_id: int
    ---
    curation_datetime: datetime    # UTC time when the curation was performed
    parent_curation_id=-1: int     # if -1, this curation is based on the raw spike sorting results
    -> CurationMethod              # which method/package used for manual curation (inform how to ingest the results)
    description="": varchar(1000)  # user-defined description/note of the curation
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@dj_store
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
    new_unit_count: int             # number of new units added
    removed_unit_count: int         # number of units removed
    """

    def make(self, key):
        """
        Update/overwrite the SortedSpikes (and downstream) with the official curation results.

        This function:
        1. Loads the curation JSON from ManualCuration
        2. Applies curation to the sorting analyzer using SpikeInterface
        3. Saves the curated analyzer in a separate folder
        4. Deletes old SortedSpikes and downstream tables
        5. Re-inserts SortedSpikes with curated data
        6. Re-populates downstream tables if they exist
        7. Tracks unit count changes
        """
        import spikeinterface as si
        from spikeinterface.curation import apply_curation

        execution_time = datetime.now(UTC)

        # Get curation_id from OfficialCuration entry (it's in attributes, not primary key)
        # The key only contains SortedSpikes primary key fields, not attributes
        curation_id = (OfficialCuration & key).fetch1("curation_id")

        # Get the curation file path
        curation_file_path = Path(
            (ManualCuration.File & key & {"curation_id": curation_id}).fetch1("file")
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
        analyzer_output_dir = output_dir / "sorting_analyzer"
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
        curated_analyzer.save(folder=curated_analyzer_dir, overwrite=True)

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
                f"Curation (curation_id={curation_id}) is already applied to this session. "
                "No action needed. If you need to reprocess, restore the raw uncurated version first."
            )
            return

        if current_curation_id is not None and current_curation_id != -1:
            # A different curation is already applied
            raise ValueError(
                f"A different curation (curation_id={current_curation_id}) has already been applied to this session. "
                f"Cannot apply curation_id={curation_id}. "
                "If you want to apply a new curation, the data needs to be reverted to the uncurated version first."
            )

        # Current state: curation_id == -1 or no entry exists
        # Proceed with applying the curation

        # Delete existing SortedSpikes entry (curation_id=-1) if it exists
        # DataJoint will automatically delete downstream tables
        if current_curation_id == -1:
            logger.info(
                "Deleting old SortedSpikes (curation_id=-1) and downstream tables..."
            )
            (spike_sorting.SortedSpikes & key).delete()
            logger.info(
                "Deleted SortedSpikes (downstream tables auto-deleted by DataJoint)"
            )

        # Assert that downstream tables don't have entries for this session
        # (they shouldn't after deletion, but verify to catch any inconsistencies)
        assert not (spike_sorting.Waveform & key), (
            "Waveform entries still exist after SortedSpikes deletion. "
            "This should not happen - downstream tables should be auto-deleted."
        )
        assert not (spike_sorting.SortingQuality & key), (
            "SortingQuality entries still exist after SortedSpikes deletion. "
            "This should not happen - downstream tables should be auto-deleted."
        )
        assert not (spike_sorting.SyncedSpikes & key), (
            "SyncedSpikes entries still exist after SortedSpikes deletion. "
            "This should not happen - downstream tables should be auto-deleted."
        )

        # Re-insert SortedSpikes with curated data using SortedSpikes.make()
        # SortedSpikes.make() will check for official curation and use the curated analyzer
        logger.info("Re-inserting SortedSpikes with curated data...")
        spike_sorting.SortedSpikes.populate(key)
        logger.info("SortedSpikes re-insertion complete")

        # Re-populate downstream tables for the new curated SortedSpikes entry
        logger.info("Re-populating downstream tables...")
        logger.info("Populating Waveform...")
        spike_sorting.Waveform.populate(key)
        logger.info("Populating SortingQuality...")
        spike_sorting.SortingQuality.populate(key)
        logger.info("Populating SyncedSpikes...")
        spike_sorting.SyncedSpikes.populate(key)

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
