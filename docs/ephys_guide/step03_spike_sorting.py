"""
03 -- Spike Sorting
===================
Set up spike sorting prerequisites, preprocess ephys data, submit GPU
sorting via SLURM, and ingest the results.

Before you can sort spikes, three manual/lookup tables must be populated:
    SortingParamSet  -- kilosort4 parameters (insert once globally)
    ElectrodeGroup   -- which electrodes to sort (all 384, or a subset)
    SortingTask      -- links each block + electrode group + param set

After those are in place:
    PreProcessing  -- bandpass-filters and writes a binary recording.dat
                      (CPU only, runs here in the script)
    SpikeSorting   -- runs kilosort4 on GPU (submitted via SLURM, NOT
                      called from this script)

This script handles everything up to SLURM submission. After sorting
completes, continue to Step 4 for post-processing and curation.

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_guide/step03_spike_sorting.py
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


def print_slurm_config(experiment_name, subject):
    """Print SLURM job configuration for submitting spike sorting.

    SpikeSorting requires a GPU and is submitted as a SLURM batch job,
    NOT run interactively from this script. This function prints:
        a) A Python config file to create/edit (run_aeon_spike_sorting.py)
        b) A SLURM submission script template (run_aeon_spike_sorting.sh)
        c) How to submit and monitor the job

    The templates are pre-filled with actual values from the database
    where possible (block boundaries, probe type, electrode config).

    Args:
        experiment_name: The experiment to generate config for.
        subject: Filter blocks to this subject.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline import spike_sorting

    # Query actual block info to fill in the template.
    blocks = (
        spike_sorting.SortingTask & {"experiment_name": experiment_name}
    ).to_dicts()
    if subject:
        blocks = [b for b in blocks if b["subject"] == subject]

    if not blocks:
        print("No SortingTask entries found. Run setup_sorting_prerequisites first.")
        return

    # Extract common fields from the first block.
    first = blocks[0]
    probe_type = first["probe_type"]
    electrode_config_name = first["electrode_config_name"]
    paramset_id = first["paramset_id"]

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

    # ---- a) Python config ----
    print()
    print("-" * 60)
    print("  (a) Copy-paste into: submodules/aeon_mecha/aeon/dj_pipeline/scripts/run_aeon_spike_sorting.py")
    print("-" * 60)
    print(
        f"""
# Replace the _base and keys sections in
# submodules/aeon_mecha/aeon/dj_pipeline/scripts/run_aeon_spike_sorting.py
# with these values:

table_name = "SpikeSorting"

_base = {{
    "experiment_name": "{experiment_name}",
    "subject": "{subject}",
    "insertion_number": {first['insertion_number']},
    "probe_type": "{probe_type}",
    "electrode_config_name": "{electrode_config_name}",
    "paramset_id": "{paramset_id}",
}}

keys = [
{keys_block}
]
"""
    )

    # ---- b) Submit and monitor ----
    print("-" * 60)
    print("  (b) Submit and monitor")
    print("-" * 60)
    print(
        """
  The SLURM script is already in the repo:
    submodules/aeon_mecha/run_aeon_spike_sorting.sh

  IMPORTANT: Update the array size in the .sh file to match your
  number of keys. Edit the line:
    #SBATCH --array=1-{len(blocks)}

  Submit (from your analysis repo root):
    sbatch submodules/aeon_mecha/run_aeon_spike_sorting.sh

  Monitor:
    squeue -u $USER                 # list your queued/running jobs
    sacct -j <jobid> --format=JobID,Elapsed,State,MaxRSS
    tail -f slurm_output/<node>_<jobid>.out   # live log

  Cancel:
    scancel <jobid>

  After sorting completes successfully, return to this script and run
  the post-sorting steps (run_post_sorting + verify_sorting).
"""
    )


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

    print("\n--- 3/3: SLURM config for spike sorting ---")
    print_slurm_config(EXPERIMENT_NAME, SUBJECT)

    print("=" * 60)
    print("  Step 3 complete.")
    print("  Submit the SLURM job and wait for it to complete.")
    print("  Then continue to Step 4 (post-processing and curation).")
    print("=" * 60)
