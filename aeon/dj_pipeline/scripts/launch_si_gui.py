"""
Script to launch SpikeInterface GUI for manual spike sorting curation.

This script loads a sorting analyzer from a local directory and launches
the SpikeInterface GUI for manual curation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import spikeinterface as si
import shutil
from aeon.dj_pipeline import spike_sorting, spike_sorting_curation


def launch_si_gui(
    key: Dict[str, Any], local_root_dir: Path, parent_curation_id: Optional[int] = None
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
        local_root_dir: Path to the local mount of the server volume where
            the sorting data is stored. This should point to the root directory
            that contains experiment folders. Can be a string or Path object.
        parent_curation_id: Optional curation_id to base this curation on. If provided,
            the curation_data.json file will be initialized from the specified curation.
            If None, starts from the raw sorting results.

    The function constructs the path to the sorting_analyzer directory based on
    the key and launches the SI GUI for manual curation.
    """
    from datetime import datetime, UTC

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
    # Structure: experiment_name / ephys_blocks / {start}_{end} / electrode_group / {method}_{paramset} / sorting_analyzer
    analyzer_dir = (
        local_root_dir
        / key["experiment_name"]
        / "ephys_blocks"
        / f"{start_str}_{end_str}"
        / key["electrode_group"]
        / f"{sorting_method}_{key['paramset_id']}"
        / "sorting_analyzer"
    )

    if not analyzer_dir.exists():
        raise FileNotFoundError(
            f"Sorting analyzer directory not found: {analyzer_dir}\n"
            f"Please verify the key and local_root_dir are correct."
        )

    # Handle parent curation if specified
    gui_dir = analyzer_dir / "spikeinterface_gui"
    gui_dir.mkdir(parents=True, exist_ok=True)
    curation_data_file = gui_dir / "curation_data.json"

    if parent_curation_id is not None:
        # Get the SpikeSorting key to query ManualCuration
        spike_sorting_key = (spike_sorting.SpikeSorting & key).fetch1("KEY")

        # Get the parent curation file
        parent_curation_key = {**spike_sorting_key, "curation_id": parent_curation_id}
        if not (spike_sorting_curation.ManualCuration & parent_curation_key):
            raise ValueError(
                f"Parent curation with curation_id={parent_curation_id} not found "
                f"for this sorting task."
            )

        # Get the parent curation file path
        parent_file = Path(
            (spike_sorting_curation.ManualCuration.File & parent_curation_key).fetch1(
                "file"
            )
        )

        if not parent_file.exists():
            raise FileNotFoundError(
                f"Parent curation file not found: {parent_file}\n"
                f"Please verify the file exists and is accessible from your local mount."
            )

        # Check if curation_data.json already exists
        if curation_data_file.exists():
            print("\n" + "=" * 70)
            print("WARNING: curation_data.json already exists!")
            print("=" * 70)
            print(f"File path: {curation_data_file}")
            print(
                "\nThis file will be OVERWRITTEN if you proceed with loading the parent curation."
            )
            print(
                "Please either:\n"
                "  1. Finish and save the current curation using save_curation.py, OR\n"
                "  2. Delete the curation_data.json file manually"
            )
            print(f"\nFile to delete: {curation_data_file}")
            print("=" * 70 + "\n")
            return

        # Copy parent curation file to curation_data.json
        print(f"Loading parent curation (curation_id={parent_curation_id})...")
        shutil.copy2(parent_file, curation_data_file)
        print(f"Copied parent curation to: {curation_data_file}\n")

    # Check for existing curation_data.json file (if not loading from parent)
    if curation_data_file.exists() and parent_curation_id is None:
        file_mtime = datetime.fromtimestamp(curation_data_file.stat().st_mtime, tz=UTC)
        time_since_modification = (datetime.now(UTC) - file_mtime).total_seconds()
        days_ago = time_since_modification / 86400

        print("\n" + "=" * 70)
        print("WARNING: Existing curation data found!")
        print("=" * 70)
        print(f"This session already has curation data in: {curation_data_file}")
        print(f"Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Modified {days_ago:.1f} days ago")
        print(
            "\nThis curation has NOT been saved to a curation_id in the ManualCuration table yet."
        )
        print("This is fine if you are picking up where you left off.")
        print(
            "However, be wary of saving over the curation if it is from another user."
        )
        print(
            "\nNOTE: Clicking 'Save in analyzer' in the GUI will OVERWRITE the existing"
        )
        print("curation_data.json file with your new additions.")
        print("=" * 70 + "\n")

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
    print("\n" + "=" * 70)
    print("GUI closed. Don't forget to run save_curation.py to save your curation")
    print("to the ManualCuration table with a unique curation_id.")
    print("=" * 70 + "\n")


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

    launch_si_gui(key, local_root_dir)

    # IMPORTANT: While using the SpikeInterface GUI for manual curation,
    # remember to periodically click the "Save in analyzer" button to save your work.
    # This ensures your curation changes are preserved in the sorting_analyzer directory.
    #
    # NOTE: If multiple users are curating the same session simultaneously, be aware that
    # saving in the GUI will overwrite the curation_data.json file. To preserve your curation
    # permanently, run save_curation.py immediately after saving in the GUI.
