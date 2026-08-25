"""
Script to launch SpikeInterface GUI for manual spike sorting curation.

================================================================================
IMPORTANT: THIS SCRIPT IS DESIGNED TO BE MODIFIED BEFORE RUNNING
================================================================================

This script is intended for interactive use. Before running:

1. Open this file in your IDE/editor
2. Modify the 'key' dictionary in the __main__ section with your
   specific session parameters:
   - experiment_name: Your experiment identifier
   - insertion_number: Probe insertion number
   - block_start: Start datetime of the block (string or datetime)
   - block_end: End datetime of the block (string or datetime)
   - electrode_group: Electrode group identifier (e.g., "0-143")
   - paramset_id: Parameter set ID (e.g., "250")
3. Optionally modify parent_curation_id if you want to base curation on an
   existing curation
4. Run the script from your IDE (not from command line)

This workflow is designed for interactive, exploratory use where you modify the
parameters for each session you want to curate. For automated/scripted use,
consider calling launch_spikeinterface_gui() directly from your own code.

Example usage:
    Modify the key dictionary below, then run: python launch_si_gui.py
    Or import and call: launch_si_gui(key, parent_curation_id=None)

    To use a custom view layout (e.g. if you don't have write permission to save your
    own default GUI settings), pass --layout pointing to a layout JSON file. See:
    https://spikeinterface-gui.readthedocs.io/en/latest/custom_layout_settings.html
        python launch_si_gui.py --layout spikeinterface_gui_layout.json
"""

import time

_launch_start_time = time.perf_counter()

import argparse
import json
import resource
import traceback

try:
    from aeon.dj_pipeline import spike_sorting_curation
except Exception:
    traceback.print_exc()
    raise


def _set_window_title(key: dict) -> None:
    """Set the GUI window title to the session key, so multiple open GUI windows
    (e.g. different shanks/blocks) can be told apart in the title bar, Cmd+Tab, and
    Mission Control - the default title is just the generic "SpikeInterface GUI".

    Patches setWindowTitle() itself (not __init__) because run_mainwindow() calls
    win.setWindowTitle('SpikeInterface GUI') unconditionally right after constructing
    the window - a title set during __init__ gets immediately overwritten by that.
    """
    from spikeinterface_gui.backend_qt import QtMainWindow
    from spikeinterface_gui.myqt import QT

    title = (
        f"{key['experiment_name']} | insertion{key['insertion_number']} | "
        f"{key['electrode_group']} | {key['block_start']}"
    )
    original_set_title = QT.QMainWindow.setWindowTitle

    def patched_set_title(self, _ignored_title):
        original_set_title(self, title)

    QtMainWindow.setWindowTitle = patched_set_title


def _print_launch_time() -> None:
    """Print how long the GUI took to launch (from script start to the window actually
    appearing), since there's no feedback otherwise while DataJoint connects and the
    analyzer + its extensions load - which can take anywhere from seconds to over a
    minute depending on the block.

    Patches show() rather than __init__: __init__ finishes before the window is
    actually visible, and show() is called exactly once, right before run_mainwindow()
    enters its blocking event loop - so this measures "ready to use", not just
    "constructed".
    """
    from spikeinterface_gui.backend_qt import QtMainWindow

    original_show = QtMainWindow.show

    def patched_show(self):
        original_show(self)
        elapsed = time.perf_counter() - _launch_start_time
        print(f"GUI launched in {elapsed:.1f}s")

    QtMainWindow.show = patched_show


def _analyzer_saved_confirmation() -> None:
    """Print a one-line confirmation when "Save in analyzer" succeeds, since the
    underlying method gives no feedback at all on success :). On failure, print a
    clear explanation for the common case (the /Volumes/aeon network mount dropping
    mid-session, which raises some OSError subclass) instead of a bare traceback.
    """
    from spikeinterface_gui.controller import Controller

    original_save = Controller.save_curation_in_analyzer

    def patched_save(self):
        try:
            original_save(self)
        except OSError as e:
            print(
                f"Save in analyzer FAILED - could not write to {self.analyzer.folder}\n"
                f"Check connection to /Volumes/aeon server, then retry 'Save in analyzer'.\n"
                f"Error: {type(e).__name__}: {e}"
            )
            return
        print(f"Curation saved in analyzer: {self.analyzer.folder}")

    Controller.save_curation_in_analyzer = patched_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Path to a custom spikeinterface-gui layout JSON file.",
    )
    parser.add_argument(
        "--no-traces",
        action="store_true",
        help=(
            "Drop the trace/tracemap views and stop the waveform view from accessing the "
            "raw recording. Useful for very long blocks, where the recording's binary "
            "chunk files can be a major contributor to hitting the OS's open-file limit."
        ),
    )
    args = parser.parse_args()

    # zarr's one-file-per-chunk storage (analyzer extensions, and the raw recording if traces
    # are on) can exhaust the OS's per-process open-file limit on a long curation session
    # (macOS defaults to 256) - print current limit + mitigations instead of raising it
    # automatically, so it stays under the user's control.
    if not args.no_traces:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(
            f"Open-file limit: {soft_limit}. If you hit 'Too many open files' during a long "
            f"curation session: raise your shell's limit before launching (e.g. `ulimit -n "
            f"4096`), and/or pass --no-traces to stop the GUI from accessing the raw recording."
        )

    layout = None
    if args.layout is not None:
        with open(args.layout) as f:
            layout = json.load(f)

    # Example key - modify these values for your session
    # The key must contain:
    #   - experiment_name: Your experiment identifier
    #   - insertion_number: Probe insertion number
    #   - block_start: Start datetime of the block (string or datetime)
    #   - block_end: End datetime of the block (string or datetime)
    #   - electrode_group: Electrode group identifier (e.g., "0-143")
    #   - paramset_id: Parameter set ID (e.g., "250")
    key = {
        "experiment_name": "abcGolden01-aeon3",
        "insertion_number": 1,
        "block_start": "2026-05-11 07:49:47.571574",
        "block_end": "2026-05-11 08:59:47.571574",
        "electrode_group": "shank0",
        "paramset_id": "400",
    }
    # Optional: parent_curation_id to base this curation on an existing curation.
    # If provided, the curation_data.json file will be initialized from the specified curation.
    # If None, starts from the raw sorting results.
    parent_curation_id = None

    # Custom label categories for the curation panel. Only applies to blocks with no
    # curation_data.json / zarr curation attrs already saved - see launch_spikeinterface_gui().
    # "tags" is edited via the multitag plugin view registered below, not the built-in
    # unit table editor. label_options is sourced from MANUAL_TAG_OPTIONS - the same list
    # spike_sorting_curation.py reads back off the analyzer into SortedSpikes.UnitTag - so
    # this, multitag_view.TAG_SHORTCUTS, and the DB-writing side can't drift apart.
    label_definitions = {
        "quality": {"label_options": ["good", "noise", "MUA"], "exclusive": True},
        "tags": {
            "label_options": list(spike_sorting_curation.MANUAL_TAG_OPTIONS),
            "exclusive": False,
        },
    }

    # Register the custom multitag view (checkbox + keyboard-shortcut multi-tagging,
    # see multitag_view.py) so it can be referenced in the layout below.
    import spikeinterface_gui.viewlist as _viewlist
    from multitag_view import MultiTagView

    _viewlist.builtin_views.append(MultiTagView)

    _set_window_title(key)
    _print_launch_time()
    _analyzer_saved_confirmation()

    # Launch SpikeInterface GUI for manual curation
    try:
        spike_sorting_curation.launch_spikeinterface_gui(
            key,
            parent_curation_id,
            layout=layout,
            label_definitions=label_definitions,
            with_traces=not args.no_traces,
        )
    except Exception:
        traceback.print_exc()
        raise

    # IMPORTANT: While using the SpikeInterface GUI for manual curation,
    # remember to periodically click the "Save in analyzer" button to save your work.
    # This ensures your curation changes are preserved in the sorting_analyzer directory.
    #
    # NOTE: If multiple users are curating the same session simultaneously, be aware that
    # saving in the GUI will overwrite the curation_data.json file. To preserve your curation
    # permanently, run save_curation.py immediately after saving in the GUI.
