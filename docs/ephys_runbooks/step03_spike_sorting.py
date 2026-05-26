"""
03 -- Spike Sorting
===================
Set up spike sorting prerequisites, preprocess ephys data, generate
SLURM submission scripts, and submit GPU sorting.

Before you can sort spikes, three manual/lookup tables must be populated:
    SortingParamSet  -- kilosort4 parameters (insert once globally)
    ElectrodeGroup   -- which electrodes to sort (all 384, or a subset)
    SortingTask      -- links each block + electrode group + param set

After those are in place:
    PreProcessing  -- bandpass-filters and writes a binary recording.dat
                      (CPU only, runs here in the script)
    SpikeSorting   -- runs kilosort4 on GPU (submitted via SLURM, NOT
                      called from this script)

This script handles everything up to SLURM submission:
    1. setup_sorting_prerequisites() -- populates the three manual tables
    2. run_preprocessing()           -- runs PreProcessing.populate()
    3. write_spike_sorting_scripts() -- generates run_aeon_spike_sorting.py
       and .sh in the working directory, pre-filled from DB state

After sorting completes, continue to Step 4 for post-processing and
curation. See docs/ephys_runbooks/run_aeon_spike_sorting.{py,sh} for
example output (golden dataset values).

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_runbooks/step03_spike_sorting.py
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
# (none at module level -- all imports are deferred inside functions
# to avoid DB side effects when the module is imported)

# --------------------------------------------------------------------------
# Configuration -- edit these for your experiment
# --------------------------------------------------------------------------

EXPERIMENT_NAME = "abcGolden01-aeonx1"
SUBJECT = "IAA-1147881"
PARAMSET_ID = 400
SORTING_METHOD = "kilosort4"

# How to group channels for spike sorting. Options:
#   "per_shank"  — one group per shank (recommended for multi-shank probes)
#   "all"        — all channels in one group
# For fully manual grouping, see "Advanced Configuration" at the bottom.
SORTING_GROUPS = "per_shank"

# Path to raw ephys data and channel map file (same values as step01).
# Needed when SORTING_GROUPS = "per_shank" to read shank assignments
# from the probeinterface JSON.
RAW_EPHYS_DIR = "/ceph/aeon/aeon/data/raw/AEONX1/abcGolden01/"
CHANNEL_MAP_FILE = "M81_ProbeB_4Shanks_1000_to_1700_um.json"


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def setup_sorting_prerequisites(
    experiment_name, subject, paramset_id, sorting_method,
    sorting_groups="per_shank", raw_ephys_dir=None, channel_map_file=None,
):
    """Populate the three manual/lookup tables that must exist before sorting.

    Three tables are populated in order:

    a) SortingParamSet (Lookup) -- kilosort4 parameters with SpikeInterface
       preprocessing and postprocessing settings. Only needs to be inserted
       once globally; subsequent calls skip if the paramset_id already exists.

    b) ElectrodeGroup + ElectrodeGroup.Electrode (Manual) -- defines which
       electrodes to include in sorting. The grouping strategy is set by
       ``sorting_groups``:
         - "per_shank": reads the probeinterface JSON to determine shank
           membership, creates one group per shank (e.g. shank0, shank1, ...)
         - "all": all active channels in one group

    c) SortingTask (Manual) -- one entry per (block, electrode_group),
       linking each to the parameter set. This is what
       PreProcessing.populate() uses to discover work.

    Args:
        experiment_name: The experiment to set up sorting for.
        subject: If given, only create SortingTask entries for this subject.
        paramset_id: Integer ID for the parameter set (converted to str for
            the varchar(16) column).
        sorting_method: Sorting algorithm name, e.g. "kilosort4".
        sorting_groups: "per_shank" or "all".
        raw_ephys_dir: Path to raw ephys data (needed for "per_shank").
        channel_map_file: Probeinterface JSON filename (needed for "per_shank").
    """
    # Deferred imports -- no DB side effects at module level.
    import json as _json
    from pathlib import Path

    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline import spike_sorting

    # ------------------------------------------------------------------
    # a) SortingParamSet -- insert once globally
    # ------------------------------------------------------------------
    # The paramset_id column is varchar(16), so we must convert to string.
    paramset_id_str = str(paramset_id)

    if not (spike_sorting.SortingParamSet & {"paramset_id": paramset_id_str}):
        params = {
            # NOTE: Preprocessing is fixed in the pipeline (bandpass 300-6000 Hz
            # + median common average referencing, via ephys_preproc() in
            # spike_sorting.py). It is not configurable through this params dict.
            #
            # Parameters passed to SpikeInterface's run_sorter().
            "SI_SORTING_PARAMS": {
                "n_pcs": 3,
                "do_CAR": False,
                "keep_good_only": True,
                "use_binary_file": True,
            },
            # Post-sorting analysis: which SpikeInterface extensions to compute
            # and their parameters. These generate quality metrics, waveform
            # templates, spike amplitudes, etc.
            "SI_POSTPROCESSING_PARAMS": {
                "extensions": {
                    "random_spikes": {},
                    "waveforms": {},
                    "templates": {},
                    "noise_levels": {},
                    "correlograms": {},
                    "isi_histograms": {},
                    "principal_components": {
                        "n_components": 5,
                        "mode": "by_channel_local",
                    },
                    "spike_amplitudes": {},
                    "spike_locations": {},
                    "template_metrics": {
                        "include_multi_channel_metrics": True,
                    },
                    "template_similarity": {},
                    "unit_locations": {},
                    "quality_metrics": {},
                },
                "job_kwargs": {"n_jobs": 1, "chunk_duration": "1s"},
                "export_to_phy": False,
                "export_report": True,
            },
        }
        spike_sorting.SortingParamSet.insert1(
            {
                "paramset_id": paramset_id_str,
                "sorting_method": sorting_method,
                "paramset_description": (
                    "Default parameter set for Kilosort4 with SpikeInterface"
                ),
                "params": params,
            }
        )
        print(f"Inserted SortingParamSet: paramset_id={paramset_id_str}")
    else:
        print(f"SortingParamSet already exists: paramset_id={paramset_id_str}")

    # ------------------------------------------------------------------
    # b) ElectrodeGroup + ElectrodeGroup.Electrode
    # ------------------------------------------------------------------
    # We need the (probe_type, electrode_config_name) key. Rather than
    # hard-coding it, query from EphysBlockInfo which was populated in
    # step 2 -- it already knows the electrode configuration.
    block_info = (
        ephys.EphysBlockInfo & {"experiment_name": experiment_name}
    ).fetch("probe_type", "electrode_config_name", as_dict=True, limit=1)

    if not block_info:
        print(
            "ERROR: No EphysBlockInfo entries found. "
            "Run step 2 (define_blocks) first."
        )
        return

    probe_type = block_info[0]["probe_type"]
    electrode_config_name = block_info[0]["electrode_config_name"]
    electrode_config_key = {
        "probe_type": probe_type,
        "electrode_config_name": electrode_config_name,
    }
    print(
        f"Using electrode config: probe_type={probe_type}, "
        f"electrode_config_name={electrode_config_name}"
    )

    # Get all electrodes in this config.
    all_electrodes = (
        ephys.ElectrodeConfig.Electrode & electrode_config_key
    ).fetch("electrode")

    # Build groups based on the sorting strategy.
    if sorting_groups == "per_shank":
        if not raw_ephys_dir or not channel_map_file:
            raise ValueError(
                "raw_ephys_dir and channel_map_file are required "
                "when sorting_groups='per_shank'."
            )

        # Read the probeinterface JSON for shank assignments.
        raw_path = Path(raw_ephys_dir)
        epoch_dirs = sorted(
            d for d in raw_path.iterdir()
            if d.is_dir() and "T" in d.name and not d.name.startswith(".")
        )
        if not epoch_dirs:
            raise FileNotFoundError(
                f"No epoch directories found in {raw_ephys_dir}"
            )
        json_path = epoch_dirs[0] / channel_map_file
        if not json_path.exists():
            raise FileNotFoundError(f"Channel map not found: {json_path}")

        with open(json_path) as f:
            pi_data = _json.load(f)

        # Build {electrode_site: shank_id} for active contacts.
        shank_map = {}
        for probe in pi_data.get("probes", []):
            contact_ids = probe.get("contact_ids", [])
            dci = probe.get("device_channel_indices", [])
            shank_ids = probe.get("shank_ids", [])
            for cid, ch_idx, shank in zip(contact_ids, dci, shank_ids):
                if int(ch_idx) >= 0:
                    shank_map[int(cid)] = str(shank)

        # Group electrodes by shank (only those in the ElectrodeConfig).
        active_set = set(int(e) for e in all_electrodes)
        shanks = {}
        for site, shank in sorted(shank_map.items()):
            if site in active_set:
                shanks.setdefault(shank, []).append(site)

        groups = []
        for shank_id in sorted(shanks):
            group_name = f"shank{shank_id}"
            sites = shanks[shank_id]
            groups.append((group_name, sites))

    elif sorting_groups == "all":
        groups = [("all", list(all_electrodes))]

    else:
        raise ValueError(
            f"Unknown sorting_groups: {sorting_groups!r}. "
            f"Use 'per_shank' or 'all'."
        )

    # Insert groups.
    for group_name, sites in groups:
        spike_sorting.ElectrodeGroup.insert1(
            {
                **electrode_config_key,
                "electrode_group": group_name,
                "electrode_group_description": (
                    f"Shank {group_name.replace('shank', '')}: "
                    f"{len(sites)} channels"
                    if sorting_groups == "per_shank"
                    else f"All {len(sites)} active channels"
                ),
                "electrode_count": len(sites),
            },
            skip_duplicates=True,
        )
        spike_sorting.ElectrodeGroup.Electrode.insert(
            (
                {
                    **electrode_config_key,
                    "electrode_group": group_name,
                    "electrode": e,
                }
                for e in sites
            ),
            skip_duplicates=True,
        )
        print(
            f"  ElectrodeGroup '{group_name}': {len(sites)} electrodes"
        )

    # ------------------------------------------------------------------
    # c) SortingTask -- one per (block, group)
    # ------------------------------------------------------------------
    blocks = (
        ephys.EphysBlock & {"experiment_name": experiment_name}
    ).to_dicts()
    if subject:
        blocks = [b for b in blocks if b["subject"] == subject]

    if not blocks:
        print(
            f"No EphysBlock entries found for experiment={experiment_name}, "
            f"subject={subject}. Run step 2 first."
        )
        return

    insert_count = 0
    for block in blocks:
        for group_name, _ in groups:
            task_key = {
                "experiment_name": block["experiment_name"],
                "subject": block["subject"],
                "insertion_number": block["insertion_number"],
                "block_start": block["block_start"],
                "block_end": block["block_end"],
                "probe_type": probe_type,
                "electrode_config_name": electrode_config_name,
                "electrode_group": group_name,
                "paramset_id": paramset_id_str,
            }
            spike_sorting.SortingTask.insert1(task_key, skip_duplicates=True)
            insert_count += 1

    total = len(
        spike_sorting.SortingTask & {"experiment_name": experiment_name}
    )
    print(f"SortingTask: inserted {insert_count} entries ({total} total)")


def run_preprocessing(experiment_name):
    """Run PreProcessing.populate() for all pending SortingTask entries.

    PreProcessing is a Computed table that:
        1. Loads raw ephys binary files for the block's time range
        2. Concatenates them into a single SpikeInterface recording
        3. Selects channels in the ElectrodeGroup
        4. Applies bandpass filtering (300-6000 Hz) and common average
           referencing (median)
        5. Writes a preprocessed binary file (recording.dat) to the
           sorting output directory on Ceph

    RESOURCE WARNINGS:
        - PreProcessing writes a full preprocessed binary (recording.dat).
        - For a 20-hour block with 384 channels at 30 kHz, expect ~1.5 TB
          of disk output per block.
        - For the test blocks in this guide (~30 minutes each), output is
          roughly 40 GB per block and 16 GB RAM should be sufficient.
        - For production blocks (20 hours), you will need more RAM and a
          longer srun allocation. Consider submitting preprocessing itself
          as a SLURM job for large blocks.

    TROUBLESHOOTING:
        If PreProcessing crashes with ``BrokenProcessPool`` during the
        ``write_binary_recording`` step, this is a known intermittent
        issue where forked worker processes are killed by the OS (likely
        a fork + BLAS threading incompatibility). We have seen this on
        the SWC HPC but have not been able to reliably reproduce it. If
        you hit this, please report it immediately — and as a temporary
        workaround, you can set ``n_jobs=1`` in PreProcessing.make_compute
        (spike_sorting.py line ~268). This disables multiprocessing and
        is slower but reliable.

    Args:
        experiment_name: The experiment to preprocess.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import spike_sorting

    pending = len(
        spike_sorting.SortingTask
        & {"experiment_name": experiment_name}
    ) - len(
        spike_sorting.PreProcessing
        & {"experiment_name": experiment_name}
    )
    print(f"Pending PreProcessing entries: {pending}")

    if pending == 0:
        print("Nothing to preprocess.")
        return

    print("Running PreProcessing.populate()...")
    print(
        "  (This writes preprocessed binary data to Ceph. "
        "It may take a while for large blocks.)"
    )
    spike_sorting.PreProcessing.populate(
        display_progress=True, suppress_errors=False
    )
    print("PreProcessing complete.")


def write_spike_sorting_scripts(experiment_name, subject, outdir=None):
    """Generate run_aeon_spike_sorting.py and .sh scripts from DB state.

    SpikeSorting requires a GPU and is submitted as a SLURM batch job.
    This function queries SortingTask for the experiment, generates a
    Python worker script and a SLURM submission script pre-filled with
    the correct keys, and writes both to the output directory.

    See docs/ephys_runbooks/run_aeon_spike_sorting.{py,sh} for example
    output (golden dataset values).

    Args:
        experiment_name: The experiment to generate scripts for.
        subject: Filter blocks to this subject.
        outdir: Directory to write scripts into. Defaults to cwd.
            Created if it doesn't exist.

    Returns:
        Tuple of (py_path, sh_path) for the written files,
        or None if no SortingTask entries were found.
    """
    # Deferred imports -- no DB side effects at module level.
    from pathlib import Path

    from aeon.dj_pipeline import spike_sorting

    # Query SortingTask entries that haven't been sorted yet.
    restriction = {"experiment_name": experiment_name}
    if subject:
        restriction["subject"] = subject

    all_tasks = spike_sorting.SortingTask & restriction
    pending = (all_tasks - spike_sorting.SpikeSorting).to_dicts()
    n_total = len(all_tasks)
    n_done = n_total - len(pending)

    if n_total == 0:
        print("No SortingTask entries found. Run setup_sorting_prerequisites first.")
        return None

    if not pending:
        print(
            f"All {n_total} SortingTask entries already have SpikeSorting "
            f"results. No scripts written."
        )
        return None

    if n_done > 0:
        print(
            f"Generating scripts for {len(pending)} of {n_total} "
            f"SortingTask entries ({n_done} already sorted)"
        )

    blocks = pending

    # Extract common fields from the first block.
    first = blocks[0]
    probe_type = first["probe_type"]
    electrode_config_name = first["electrode_config_name"]
    paramset_id = first["paramset_id"]
    insertion_number = first["insertion_number"]
    n_keys = len(blocks)

    # Group entries by electrode_group for organized output.
    by_group = {}
    for b in blocks:
        by_group.setdefault(b["electrode_group"], []).append(b)

    # Build the keys list string, organized by group.
    keys_lines = []
    for group_name in sorted(by_group):
        keys_lines.append(f"    # --- {group_name} ---")
        for b in sorted(by_group[group_name], key=lambda x: x["block_start"]):
            start_str = str(b["block_start"])
            end_str = str(b["block_end"])
            keys_lines.append(
                f'    {{**_base, "electrode_group": "{group_name}", '
                f'"block_start": "{start_str}", "block_end": "{end_str}"}},'
            )
    keys_block = "\n".join(keys_lines)

    # ---- Resolve output directory ----
    outdir = Path(outdir) if outdir else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    py_path = outdir / "run_aeon_spike_sorting.py"
    sh_path = outdir / "run_aeon_spike_sorting.sh"

    # ---- Generate Python script ----
    py_content = f'''\
"""Run spike sorting pipeline on a single key, selected by --task index.

Designed for SLURM job arrays: each array task gets its own GPU and
processes one key from the list. The SLURM script passes
$SLURM_ARRAY_TASK_ID as --task automatically.

Generated by write_spike_sorting_scripts() for experiment
"{experiment_name}", subject "{subject}".

To run a single task interactively (e.g. for debugging):
    uv run python run_aeon_spike_sorting.py --task 1

To select the pipeline step ("PreProcessing", "SpikeSorting", "PostProcessing"):
  - Pass --table at runtime to override TABLE_NAME (e.g. PreProcessing):
        uv run python run_aeon_spike_sorting.py --task 1 --table PreProcessing
  - Or edit TABLE_NAME below to change the default for all tasks.
"""

import argparse

from aeon.dj_pipeline import spike_sorting

# =============================================================================
# Configuration
# =============================================================================

TABLE_NAME = "SpikeSorting"

_base = {{
    "experiment_name": "{experiment_name}",
    "subject": "{subject}",
    "insertion_number": {insertion_number},
    "probe_type": "{probe_type}",
    "electrode_config_name": "{electrode_config_name}",
    "paramset_id": "{paramset_id}",
}}

keys = [
{keys_block}
]

# =============================================================================

CLEAR_JOB = False


def clear_job(dj_table, key):
    """Clear errored jobs for this table to allow re-running."""
    try:
        (spike_sorting.schema.jobs & {{"table_name": dj_table.table_name, "status": "error"}} & key).delete()
    except Exception as e:
        print(f"[WARNING] Could not clear error jobs: {{dj_table.table_name}} — {{e}}")


def main():
    """Run the selected pipeline step on the key corresponding to --task index."""
    parser = argparse.ArgumentParser(description="Spike sorting worker")
    parser.add_argument(
        "--task", type=int, required=True, help=f"Task number (1-{{len(keys)}}), from SLURM_ARRAY_TASK_ID"
    )
    parser.add_argument(
        "--table",
        default=TABLE_NAME,
        choices=["PreProcessing", "SpikeSorting", "PostProcessing"],
        help=f"Pipeline step to run (default: {{TABLE_NAME}})",
    )
    args = parser.parse_args()

    if not 1 <= args.task <= len(keys):
        raise ValueError(f"Task must be 1-{{len(keys)}}, got {{args.task}}")

    populate_table = {{
        "PreProcessing": spike_sorting.PreProcessing,
        "SpikeSorting": spike_sorting.SpikeSorting,
        "PostProcessing": spike_sorting.PostProcessing,
    }}[args.table]

    key = keys[args.task - 1]
    print(
        f"\\n=== Task {{args.task}}/{{len(keys)}}: {{key[\'electrode_group\']}} "
        f"{{key[\'block_start\']}} - {{key[\'block_end\']}} ==="
    )

    if CLEAR_JOB:
        clear_job(populate_table, key)
    populate_table.populate(key, reserve_jobs=True, display_progress=True)
    print(f"=== Task {{args.task}} done ===")


if __name__ == "__main__":
    main()
'''

    # ---- Generate shell script ----
    sh_content = f'''\
#!/bin/bash

# =============================================================================
# AEON Spike Sorting SLURM Script (Job Array)
# =============================================================================
# Submits as a job array — each task gets its own GPU and sorts one key.
# Task number ($SLURM_ARRAY_TASK_ID) selects which key from the Python
# script's keys list to process.
#
# Generated by write_spike_sorting_scripts() for experiment
# "{experiment_name}", subject "{subject}".
#
# Usage:  sbatch run_aeon_spike_sorting.sh
# Status: squeue --start -j <job_id>
# Cancel: scancel <job_id>          (cancels all array tasks)
#         scancel <job_id>_<task>    (cancels one task)
# =============================================================================

#SBATCH --job-name=aeon-spike-sorting         # job name
#SBATCH --partition=gpu                       # Change to 'cpu' for CPU-only mode
#SBATCH --gres=gpu:a100:1                    # Remove this line for CPU-only mode (options a100, p5000)
#SBATCH --nodes=1                             # node count
#SBATCH --ntasks=1                            # total number of tasks across all nodes
#SBATCH --mem=256G                            # total memory per node (typical: 64G for <2hr blocks)
#SBATCH --time=7-08:00:00                     # total run time limit (typical: 0-04:00:00 for <2hr blocks)
#SBATCH --array=1-{n_keys}                          # one task per key
#SBATCH --output=slurm_output/%N_%j_%a.out    # output file path (%a = array task ID)
#SBATCH --error=slurm_output/%N_%j_%a.err     # error file path

# Exit on any error
set -e

# Print job information
echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo "================================"

# Create output directory
mkdir -p slurm_output

# Load modules
echo "Loading modules..."
module load uv

# Change to the directory where the scripts were submitted from.
cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# Ensure venv exists and deps match lockfile
echo "Syncing dependencies..."
uv sync

# Set PyTorch CUDA memory allocator configuration to free reserved memory
# This helps prevent CUDA out of memory errors during long-running Kilosort4 jobs
# expandable_segments:True allows PyTorch to dynamically expand memory segments to reduce fragmentation
# garbage_collection_threshold:0.6 triggers GC when 60% of reserved memory is unused,
#   which should free the 7.69 GiB reserved but unallocated memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
echo "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6"

# Start resource profiler in the background
PROFILER_PATH="submodules/aeon_mecha/aeon/dj_pipeline/scripts/start_resource_profiler.py"
if [ -f "$PROFILER_PATH" ]; then
    echo "Starting resource profiler..."
    .venv/bin/python "$PROFILER_PATH" -o "./slurm_output/resource_use_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.csv" & PROFILER_PID=$!
    echo "Resource profiler started with PID: $PROFILER_PID"
else
    echo "Resource profiler not found at $PROFILER_PATH, skipping."
    PROFILER_PID=""
fi

# Verify Python script exists
SCRIPT_PATH="$SLURM_SUBMIT_DIR/run_aeon_spike_sorting.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# Run the spike sorting script for this array task
TASK=${{SLURM_ARRAY_TASK_ID:-1}}
echo "Starting spike sorting for task $TASK..."
.venv/bin/python "$SCRIPT_PATH" --task "$TASK"

# Stop the profiler
if [ -n "$PROFILER_PID" ]; then
    echo "Stopping resource profiler..."
    kill $PROFILER_PID
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "=== Job completed successfully ==="
    echo "End Time: $(date)"
    exit 0
else
    echo "=== Job failed ==="
    echo "End Time: $(date)"
    exit 1
fi
'''

    # ---- Write files ----
    if py_path.exists():
        print(f"  WARNING: Overwriting existing {py_path.name}")
    py_path.write_text(py_content)

    if sh_path.exists():
        print(f"  WARNING: Overwriting existing {sh_path.name}")
    sh_path.write_text(sh_content)

    # ---- Confirmation ----
    print(f"Wrote: {py_path} ({n_keys} keys)")
    print(f"Wrote: {sh_path}")
    print()
    print("  Submit with:")
    print("      sbatch run_aeon_spike_sorting.sh")

    return py_path, sh_path


# --------------------------------------------------------------------------
# Advanced Configuration
# --------------------------------------------------------------------------
#
# The SORTING_GROUPS setting above covers the two common cases (per-shank
# and all-channels). If you need full manual control over electrode groups
# or block parameters, this section explains how.
#
#
# --- Custom electrode groups ---
#
# The pipeline's ElectrodeGroup table accepts any name (varchar(16)) and
# any subset of electrodes. To create a custom group interactively:
#
#   from aeon.dj_pipeline import ephys, spike_sorting
#
#   # 1. Look up the electrode config from existing block info.
#   econfig_key = (
#       ephys.EphysBlockInfo & {"experiment_name": "your-experiment"}
#   ).fetch("probe_type", "electrode_config_name", as_dict=True, limit=1)[0]
#
#   # 2. See what electrodes are available.
#   all_sites = (ephys.ElectrodeConfig.Electrode & econfig_key).fetch("electrode")
#   print(f"Available electrode sites: {sorted(all_sites)}")
#
#   # 3. Create a group with your chosen subset.
#   my_sites = [114, 115, 116, 117, ...]  # your selection
#   spike_sorting.ElectrodeGroup.insert1({
#       **econfig_key,
#       "electrode_group": "my_group",   # any name up to 16 characters
#       "electrode_group_description": "Custom selection for region X",
#       "electrode_count": len(my_sites),
#   })
#   spike_sorting.ElectrodeGroup.Electrode.insert([
#       {**econfig_key, "electrode_group": "my_group", "electrode": e}
#       for e in my_sites
#   ])
#
#   # 4. Create SortingTask entries -- one per (block, group).
#   blocks = (ephys.EphysBlock & {"experiment_name": "your-experiment"}).to_dicts()
#   for block in blocks:
#       spike_sorting.SortingTask.insert1({
#           "experiment_name": block["experiment_name"],
#           "subject": block["subject"],
#           "insertion_number": block["insertion_number"],
#           "block_start": block["block_start"],
#           "block_end": block["block_end"],
#           **econfig_key,
#           "electrode_group": "my_group",
#           "paramset_id": "400",
#       }, skip_duplicates=True)
#
#
# --- Custom block parameters ---
#
# Block boundaries are configured in step02 (step02_define_blocks.py).
# The guide's test parameters are 30-minute blocks with 10-minute overlap.
#
# Production recommendations:
#   - 20-hour blocks with 5-hour overlap (standard for long sessions)
#   - Sessions <= 20 hours: single block, no overlap needed
#   - Sessions > 20 hours: automatic splitting with 5-hour overlap
#
# Edit BLOCK_DURATION_MIN, OVERLAP_MIN, and N_BLOCKS in step02, or call
# create_blocks() directly with your own parameters.
#


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 3: Spike Sorting")
    print("=" * 60)

    print("\n--- 1/3: Setup sorting prerequisites ---")
    setup_sorting_prerequisites(
        EXPERIMENT_NAME, SUBJECT, PARAMSET_ID, SORTING_METHOD,
        sorting_groups=SORTING_GROUPS,
        raw_ephys_dir=RAW_EPHYS_DIR,
        channel_map_file=CHANNEL_MAP_FILE,
    )

    print("\n--- 2/3: Run preprocessing ---")
    run_preprocessing(EXPERIMENT_NAME)

    print("\n--- 3/3: Generate SLURM scripts ---")
    write_spike_sorting_scripts(EXPERIMENT_NAME, SUBJECT)

    print("=" * 60)
    print("  Step 3 complete.")
    print("  Submit the SLURM job and wait for it to complete.")
    print("  Then continue to Step 4 (post-processing and curation).")
    print("=" * 60)
