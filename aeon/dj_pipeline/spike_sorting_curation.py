"""DataJoint tables for spike sorting manual and official curation workflows."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd

from aeon.dj_pipeline import ephys, get_schema_name, spike_sorting
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir
from aeon.dj_pipeline.utils.spike_sorting_utils import resolve_analyzer_dir

schema = dj.Schema(get_schema_name("spike_sorting_curation"))
logger = dj.logger


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # manual curation method/package (e.g. SpikeInterface, Phy, FigURL)
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
    -> CurationMethod              # manual curation method/package (inform how to ingest the results)
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
    definition = """  # One final/official curation for a block
    # Keyed on PostProcessing (not SortedSpikes, which is 1:1 with it and has the same primary key)
    # so that applying a curation - which deletes and re-populates SortedSpikes - does not
    # cascade-delete this record or the ApplyOfficialCuration that depends on it.
    -> spike_sorting.PostProcessing
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
        """Update/overwrite the SortedSpikes (and downstream) with the official curation results.

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
        # The key only contains the block's primary key fields, not attributes
        curation_id = (OfficialCuration & key).fetch1("curation_id")

        # Auto-approved curation: no manual curation file, based on raw sorting
        has_curation_file = bool(
            ManualCuration.File
            & key
            & {"curation_id": curation_id, "file_name": f"curation_data_id{curation_id}.json"}
        )
        if not has_curation_file:
            parent_curation_id = (ManualCuration & key & {"curation_id": curation_id}).fetch1(
                "parent_curation_id"
            )
            if parent_curation_id == -1:
                # Raw sorting approved as official — no curation to apply
                # Update SortedSpikes.curation_id from -1 to the official curation_id
                sorted_key = (spike_sorting.SortedSpikes & key).fetch1("KEY")
                spike_sorting.SortedSpikes.update1({**sorted_key, "curation_id": curation_id})

                self.insert1(
                    {
                        **key,
                        "execution_time": execution_time,
                        "new_unit_count": 0,
                        "removed_unit_count": 0,
                    }
                )
                logger.info(
                    f"Auto-approved curation (curation_id={curation_id}): "
                    "raw sorting results accepted as official. No changes applied."
                )
                return

        # Get the curation file path. Must restrict by file_name - ManualCuration.File also
        # holds a "curation_applied_analyzer" row (written later in this same function) for
        # any block that has ever had ApplyOfficialCuration run on it before, so on a
        # revert-then-reapply both rows already exist here and fetch1 without this filter
        # raises "2 tuples found".
        curation_file_path = Path(
            (
                ManualCuration.File
                & key
                & {"curation_id": curation_id, "file_name": f"curation_data_id{curation_id}.json"}
            )
            .fetch1("file")
            .full_path
        )

        if not curation_file_path.exists():
            raise FileNotFoundError(
                f"Curation file not found: {curation_file_path}\n"
                f"Please verify the file exists and is accessible."
            )

        # Load curation dictionary
        with open(curation_file_path) as f:
            curation_dict = json.load(f)

        # Units manually labeled "noise" are kept, not deleted: apply_curation() runs on the
        # curation exactly as saved, so noise units survive into the curated analyzer carrying
        # their "quality"="noise" property, which SortedSpikes.make() writes into
        # SortedSpikes.Unit.unit_quality on repopulation. They are held out of unit matching instead
        # (see _load_block_unit_spike_trains in spike_sorting.py), not deleted here.

        # Load original sorting analyzer
        analyzer_output_dir = _get_analyzer_dir_from_key(key)
        output_dir = analyzer_output_dir.parent
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
        params = (spike_sorting.SortingParamSet & key).fetch1("params")
        save_format = params.get("save_format", "zarr")
        if save_format == "zarr":
            # save_as(format="zarr") appends .zarr internally (clean_zarr_folder_name) -
            # resolve it here too so the exists-check/insert below match what's really on disk.
            from spikeinterface.core.core_tools import clean_zarr_folder_name

            curated_analyzer_dir = clean_zarr_folder_name(curated_analyzer_dir)
            logger.info(f"Saving curated analyzer to: {curated_analyzer_dir}")
            if curated_analyzer_dir.exists():
                shutil.rmtree(curated_analyzer_dir)
            curated_analyzer.save_as(format="zarr", folder=curated_analyzer_dir)
        else:
            logger.info(f"Saving curated analyzer to: {curated_analyzer_dir}")
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
        # There can only be one SortedSpikes entry per primary key
        # (curation_id is an attribute, not part of PK)
        current_sorted_spikes = spike_sorting.SortedSpikes & key
        current_curation_id = current_sorted_spikes.fetch1("curation_id") if current_sorted_spikes else None

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
                f"A different curation (curation_id={current_curation_id}) has already been applied to this"
                f" block. Cannot apply curation_id={curation_id}. If you want to apply a new curation, the"
                f" data needs to be reverted to the uncurated version first."
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
            logger.info("Deleting old SortedSpikes (curation_id=-1) and downstream tables...")
            (spike_sorting.SortedSpikes & key).delete(prompt=False)
            # Cascade reaches only SortedSpikes' downstream (Waveform, SortingQuality, SyncedSpikes,
            # UnitMatching). OfficialCuration and this ApplyOfficialCuration are keyed on
            # PostProcessing, not SortedSpikes, so they survive - which is what lets us record the
            # apply below and then rebuild SortedSpikes separately via SortedSpikes.populate().
            logger.info("Deleted SortedSpikes and its downstream tables (curation records preserved).")

        # Insert ApplyOfficialCuration entry. SortedSpikes is intentionally left deleted here; the
        # curated analyzer is on disk and SortedSpikes.make() rebuilds from it (reading the manual
        # quality labels into unit_quality) on the next SortedSpikes.populate().
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

        logger.info(
            "Run SortedSpikes.populate() to rebuild it from the curated analyzer, then .populate() "
            "the downstream tables (SyncedSpikes, etc.) to continue the pipeline."
        )




# Helper functions for curation workflows


def _get_analyzer_dir_from_key(key: dict) -> Path:
    """Construct the path to the sorting_analyzer directory from a key using database paths.

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
    return resolve_analyzer_dir(output_dir)


def launch_spikeinterface_gui(
    key: dict,
    parent_curation_id: int | None = None,
    layout: dict | None = None,
    label_definitions: dict | None = None,
    with_traces: bool = True,
) -> None:
    """Launch SpikeInterface GUI for manual spike sorting curation.

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
        layout: Optional custom view layout dict to pass to spikeinterface_gui's
            run_mainwindow(). If None, the GUI's default layout is used.
        label_definitions: Optional custom label categories to pass to spikeinterface_gui's
            run_mainwindow(). This replaces the built-in defaults entirely (it does not
            merge with them), and only takes effect for blocks with no curation_data.json /
            zarr curation attrs already saved - existing saved curations keep whatever
            label_definitions they were saved with. If None, the GUI's defaults are used.
        with_traces: If False, drops the "trace"/"tracemap" views (regardless of whether
            they're in layout) and stops the waveform view from overlaying live traces, so
            the raw recording is never accessed. Useful for very long blocks, where the
            recording's binary chunk files can otherwise be a major contributor to hitting
            the OS's open-file limit. Default True (matches spikeinterface_gui's default).
        """
    import shutil

    import spikeinterface as si

    analyzer_dir = _get_analyzer_dir_from_key(key)

    # Handle parent curation if specified
    gui_dir = analyzer_dir / "spikeinterface_gui"

    try:
        gui_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"No permission to create directory: {gui_dir}"
        ) from e
    except OSError as e:
        raise OSError(
            f"Failed to create directory {gui_dir}: {e}"
        ) from e

    curation_data_file = gui_dir / "curation_data.json"
    metadata_file = gui_dir / "curation_metadata.json"

    if parent_curation_id is not None:
        # Get the parent curation file
        parent_curation_key = {**key, "curation_id": parent_curation_id}
        if not (ManualCuration & parent_curation_key):
            raise ValueError(
                f"Parent curation with curation_id={parent_curation_id} not found for this sorting task."
            )

        # Get the parent curation file path. Name the file explicitly: ManualCuration.File is a
        # list of every file for a curation - the curation JSON here, plus the
        # "curation_applied_analyzer" pointer ApplyOfficialCuration writes under the same
        # curation_id - so a fetch1 must say which one it wants, or it raises "2 tuples found"
        # for any parent that has already been applied.
        parent_file = Path(
            (
                ManualCuration.File
                & parent_curation_key
                & {"file_name": f"curation_data_id{parent_curation_id}.json"}
            )
            .fetch1("file")
            .full_path
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
    if parent_curation_id is None and metadata_file.exists():
        # Delete metadata file if it exists (clearing any previous parent)
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

    # Load sorting analyzer. Load extensions one at a time (rather than
    # load_sorting_analyzer()'s default all-or-nothing load_extensions=True) and track
    # any that fail or come back empty, since a single bad extension (e.g. a numpy
    # version mismatch baked into a pickled object, or data that silently failed to
    # save - both seen in practice) would otherwise abort the entire GUI launch, or
    # crash a specific view later. A failed/empty extension is passed to
    # run_mainwindow's skip_extensions (so Controller doesn't retry loading it - it
    # calls analyzer.get_extension() itself on startup) AND used to drop any view whose
    # _depend_on needs it from the layout, since skip_extensions alone only gates the
    # two extensions Controller special-cases (waveforms, principal_components) - every
    # other view is still constructed as long as the extension is merely *declared* on
    # disk, regardless of whether its data actually loaded.
    sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_dir, load_extensions=False)
    failed_extensions = []
    for extension_name in sorting_analyzer.get_saved_extension_names():
        try:
            sorting_analyzer.load_extension(extension_name)
            if len(sorting_analyzer.extensions[extension_name].data) == 0:
                raise ValueError("extension loaded with no data - should be re-computed")
        except Exception as e:
            failed_extensions.append(extension_name)
            logger.warning(
                f"Skipping extension '{extension_name}' - failed to load "
                f"(likely computed in an incompatible environment, or incompletely "
                f"saved): {type(e).__name__}: {e}"
            )

    if failed_extensions:
        from spikeinterface_gui.layout_presets import get_layout_description
        from spikeinterface_gui.viewlist import get_all_possible_views

        # resolve the effective layout dict now (layout may be None, a dict, or a
        # path - same resolution run_mainwindow does internally) so it can be filtered
        # regardless of which form was passed in.
        layout_dict = get_layout_description(None, layout=layout)
        possible_class_views = get_all_possible_views()
        dropped_views = []
        filtered_layout = {}
        for zone, view_names in layout_dict.items():
            kept = []
            for view_name in view_names:
                depend_on = possible_class_views[view_name]._depend_on
                if depend_on is not None and any(ext in failed_extensions for ext in depend_on):
                    dropped_views.append(view_name)
                else:
                    kept.append(view_name)
            filtered_layout[zone] = kept
        layout = filtered_layout
        if dropped_views:
            logger.warning(
                f"Dropping view(s) {dropped_views} from the layout - they depend on "
                f"extension(s) that failed to load: {failed_extensions}"
            )

    # Launch GUI
    # Try spikeinterface_gui first, fall back to built-in viewer if available
    try:
        from spikeinterface_gui import run_mainwindow

        run_mainwindow(
            sorting_analyzer,
            mode="desktop",
            curation=True,
            layout=layout,
            label_definitions=label_definitions,
            skip_extensions=failed_extensions or None,
            with_traces=with_traces,
        )
    except ImportError:
        # Fallback to built-in viewer if spikeinterface_gui is not available
        si.view_sorting_analyzer(sorting_analyzer)

    # Remind user to save curation after GUI closes
    logger.info(
        "GUI closed. Don't forget to run save_curation.py to save your curation "
        "to the ManualCuration table with a unique curation_id."
    )


def save_manual_curation(key: dict, description: str = "") -> int:
    """Save manual curation results from SpikeInterface GUI to the database.

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
    gui_dir = analyzer_dir / "spikeinterface_gui"
    is_zarr = analyzer_dir.name.endswith(".zarr")

    # ManualCuration's insert needs the *complete* primary key (adds subject, probe_type,
    # electrode_config_name, etc. via -> SpikeSorting), not just the sorting-task fields
    # documented as the `key` argument above - resolve it here rather than assuming the
    # caller already has it (a partial key works fine for restrictions/filters below, just
    # not for insert1).
    full_key = (spike_sorting.SpikeSorting & key).fetch1("KEY")

    # Load the curation_data dict saved by the SI GUI's "Save in analyzer" button. For
    # binary_folder analyzers this is a plain curation_data.json file; for zarr analyzers
    # (Controller.save_curation_in_analyzer's zarr branch) it's stored in the zarr group's
    # .zattrs under a "curation_data" key instead - there is no curation_data.json at all.
    if is_zarr:
        import zarr

        curation_data = None
        if gui_dir.exists():
            zarr_root = zarr.open(str(analyzer_dir), mode="r")
            if "spikeinterface_gui" in zarr_root.keys():
                curation_data = zarr_root["spikeinterface_gui"].attrs.get("curation_data")
        if curation_data is None:
            raise FileNotFoundError(
                f"No curation_data found in zarr attrs at {gui_dir}/.zattrs\n"
                f"Please ensure you have saved your curation in the SI GUI using the 'Save in analyzer' button."
            )
    else:
        curation_data_file = gui_dir / "curation_data.json"
        if not curation_data_file.exists():
            raise FileNotFoundError(
                f"Curation data file not found: {curation_data_file}\n"
                f"Please ensure you have saved your curation in the SI GUI using the 'Save in analyzer' button."
            )
        with open(curation_data_file) as f:
            curation_data = json.load(f)

    # Find the next available curation_id
    existing_ids = (ManualCuration & key).to_arrays("curation_id")
    next_curation_id = max(existing_ids) + 1 if len(existing_ids) > 0 else 1

    # Write a versioned snapshot as its own JSON file - needed regardless of source format,
    # since ManualCuration.File tracks an actual file on disk (filepath@dj_store).
    curated_file_name = f"curation_data_id{next_curation_id}.json"
    curated_file_path = gui_dir / curated_file_name
    with open(curated_file_path, "w") as f:
        json.dump(curation_data, f, indent=4)

    # Verify the write was successful and valid before clearing the original
    if not curated_file_path.exists():
        raise RuntimeError(f"Failed to write curation snapshot to {curated_file_path}")
    try:
        with open(curated_file_path) as f:
            json.load(f)  # Verify it's valid JSON
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Written curation snapshot is not valid JSON: {curated_file_path}") from e

    # Read parent_curation_id from metadata file if it exists (plain file, same location
    # for both formats - it lives in the spikeinterface_gui directory either way)
    metadata_file = gui_dir / "curation_metadata.json"
    parent_curation_id = -1  # Default: based on raw sorting results
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                parent_curation_id = metadata.get("parent_curation_id", -1)
            logger.info(f"Using parent_curation_id={parent_curation_id} from metadata file")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"Failed to read parent_curation_id from metadata file: {e}. "
                "Using default parent_curation_id=-1"
            )
            parent_curation_id = -1

    # Insert into ManualCuration/ManualCuration.File *before* clearing the live copy below:
    # if this fails (e.g. a schema/connection issue), the live curation_data is still intact
    # and save_manual_curation() can just be retried, rather than leaving an orphaned
    # snapshot file on disk with no way to reconstruct it from the (already-cleared) source.
    #
    # Wrap both inserts in one transaction so the master and its File part land together or
    # not at all. This is a standalone function, not a table make(), so it gets none of the
    # autopopulate framework's automatic per-make() transaction - without this, a failure
    # between the two inserts would leave a ManualCuration row with no File row behind.
    curation_datetime = datetime.now(UTC)
    curation_entry = {
        **full_key,
        "curation_id": next_curation_id,
        "curation_datetime": curation_datetime,
        "parent_curation_id": parent_curation_id,
        "curation_method": "SpikeInterface",
        "description": description,
    }
    file_entry = {
        **full_key,
        "curation_id": next_curation_id,
        "file_name": curated_file_name,
        "file": curated_file_path,
    }
    with ManualCuration.connection.transaction:
        ManualCuration.insert1(curation_entry)
        ManualCuration.File.insert1(file_entry)

    # Now safe to clear the original, live-editable copy
    if is_zarr:
        zarr_root_rw = zarr.open(str(analyzer_dir), mode="r+")
        del zarr_root_rw["spikeinterface_gui"].attrs["curation_data"]
        logger.info(f"Cleared curation_data from zarr attrs (saved as {curated_file_name})")
    else:
        curation_data_file.unlink()
        logger.info(f"Deleted original curation_data.json (saved as {curated_file_name})")

    # Clean up metadata file after successful save (metadata is now in database)
    if metadata_file.exists():
        metadata_file.unlink()
        logger.info("Deleted curation metadata file (metadata now in database)")

    logger.info(f"Successfully saved curation with curation_id={next_curation_id}")
    logger.info(f"Curation file saved as: {curated_file_path}")

    return next_curation_id


def make_curation_official(key: dict, curation_id: int) -> None:
    """Designate an existing curation as official.

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
        raise ValueError(f"Curation with curation_id={curation_id} not found for this sorting task.")

    # Get the SortedSpikes key (need to find the one with curation_id=-1, the raw sorting)
    sorted_spikes_key = (spike_sorting.SortedSpikes & key & {"curation_id": -1}).fetch1("KEY")

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

    # Gate: every unit the curator kept must carry a manual quality label, so noise-exclusion and
    # downstream analysis see a complete picture. Units the curator removed, merged, or split are
    # considered handled and don't need a standalone label. Checked here (when promoting to official)
    # so an incomplete curation is refused before any apply work happens.
    curation_file = Path(
        (
            ManualCuration.File
            & sorted_spikes_key
            & {"curation_id": curation_id, "file_name": f"curation_data_id{curation_id}.json"}
        )
        .fetch1("file")
        .full_path
    )
    with open(curation_file) as f:
        curation_dict = json.load(f)
    labeled = {
        lbl["unit_id"]
        for lbl in curation_dict.get("manual_labels", [])
        if lbl.get("labels", {}).get("quality")
    }
    handled = set(curation_dict.get("removed", []))
    for merge_key in ("merges", "merge_unit_groups"):
        for group in curation_dict.get(merge_key, []) or []:
            handled.update(group)
    for split_key in ("splits", "split_units"):
        handled.update(curation_dict.get(split_key, {}) or {})
    raw_unit_ids = {int(u) for u in (spike_sorting.SortedSpikes.Unit & sorted_spikes_key).fetch("unit")}
    unlabeled = raw_unit_ids - labeled - handled
    if unlabeled:
        raise ValueError(
            f"Curation {curation_id} can't be made official: {len(unlabeled)} unit(s) have no quality "
            f"label: {sorted(unlabeled)}. Label every unit (good/mua/noise) in the GUI before making "
            f"the curation official."
        )

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
    official_curation.delete(prompt=False)
    logger.info("OfficialCuration and ApplyOfficialCuration entries deleted.")

    # Step 2: Delete UnitMatching for this block
    # Cascades to UnitMatching.Unit and UnitMatching.Spikes Part rows
    unit_matching_entries = spike_sorting.UnitMatching & key
    if unit_matching_entries:
        n_um = len(unit_matching_entries)
        logger.info(f"Deleting {n_um} UnitMatching entries for this block...")
        unit_matching_entries.delete(prompt=False)
        logger.info("UnitMatching entries deleted (cascaded to Unit and Spikes parts).")

    # Step 3: Delete orphaned GlobalUnit entries
    # (those with no remaining UnitMatching.Unit references from any block)
    insertion_key = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}
    n_orphans = 0
    for gu_key in (spike_sorting.GlobalUnit & insertion_key).keys():  # noqa: SIM118
        if len(spike_sorting.UnitMatching.Unit & gu_key) == 0:
            logger.info(f"Deleting orphaned GlobalUnit {gu_key['global_unit']}...")
            (spike_sorting.GlobalUnit & gu_key).delete(prompt=False)
            n_orphans += 1
    if n_orphans:
        logger.info(f"Deleted {n_orphans} orphaned GlobalUnit entries.")

    # Step 4: Delete the SortedSpikes entry (cascades to downstream tables)
    sorted_spikes_entry = spike_sorting.SortedSpikes & key
    if sorted_spikes_entry:
        logger.info("Deleting SortedSpikes entry and downstream tables...")
        sorted_spikes_entry.delete(prompt=False)
        logger.info(
            "SortedSpikes and downstream tables deleted.\n"
            "Next steps:\n"
            "  1. Run SortedSpikes.populate() to load the raw sorting results\n"
            "  2. Run SyncedSpikes.populate() to sync spike times\n"
            "  3. Apply new curation if needed (make_curation_official + ApplyOfficialCuration.populate)\n"
            "  4. Run UnitMatching.populate() to re-match units and generate Spikes"
        )
    else:
        logger.info(
            "No SortedSpikes entry found. "
            "Run populate to load the raw sorting results back into the pipeline."
        )
