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
    PostProcessing -- computes quality metrics, waveforms, etc.
    SortedSpikes   -- loads spike times and unit info into the DB
    Waveform       -- loads waveform templates
    SortingQuality -- loads quality metrics per unit

The script walks through pre-SLURM setup (steps 1-3), prints the SLURM
template you need to submit, then provides post-sorting steps (4-5) that
you run after the SLURM job completes.

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
ELECTRODE_GROUP_NAME = "all"  # or "0-383" -- groups all active channels


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def setup_sorting_prerequisites(
    experiment_name, subject, paramset_id, sorting_method, electrode_group_name
):
    """Populate the three manual/lookup tables that must exist before sorting.

    Three tables are populated in order:

    a) SortingParamSet (Lookup) -- kilosort4 parameters with SpikeInterface
       preprocessing and postprocessing settings. Only needs to be inserted
       once globally; subsequent calls skip if the paramset_id already exists.

    b) ElectrodeGroup + ElectrodeGroup.Electrode (Manual) -- defines which
       electrodes to include in sorting. Queries the ElectrodeConfig from
       existing EphysBlockInfo entries so you do not need to know the probe
       type or config name in advance.

    c) SortingTask (Manual) -- one entry per EphysBlock, linking it to the
       electrode group and parameter set. This is what PreProcessing.populate()
       uses to discover work.

    Args:
        experiment_name: The experiment to set up sorting for.
        subject: If given, only create SortingTask entries for this subject.
        paramset_id: Integer ID for the parameter set (converted to str for
            the varchar(16) column).
        sorting_method: Sorting algorithm name, e.g. "kilosort4".
        electrode_group_name: Label for the electrode group, e.g. "all".
    """
    # Deferred imports -- no DB side effects at module level.
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
                "job_kwargs": {"n_jobs": 0.8, "chunk_duration": "1s"},
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
    electrodes = (
        ephys.ElectrodeConfig.Electrode & electrode_config_key
    ).fetch("electrode")

    # Insert the group header.
    spike_sorting.ElectrodeGroup.insert1(
        {
            **electrode_config_key,
            "electrode_group": electrode_group_name,
            "electrode_group_description": (
                f"All {len(electrodes)} active channels"
            ),
            "electrode_count": len(electrodes),
        },
        skip_duplicates=True,
    )

    # Insert one row per electrode in the group.
    spike_sorting.ElectrodeGroup.Electrode.insert(
        (
            {
                **electrode_config_key,
                "electrode_group": electrode_group_name,
                "electrode": e,
            }
            for e in electrodes
        ),
        skip_duplicates=True,
    )
    print(
        f"ElectrodeGroup '{electrode_group_name}': "
        f"{len(electrodes)} electrodes"
    )

    # ------------------------------------------------------------------
    # c) SortingTask -- one per block
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
        task_key = {
            "experiment_name": block["experiment_name"],
            "subject": block["subject"],
            "insertion_number": block["insertion_number"],
            "block_start": block["block_start"],
            "block_end": block["block_end"],
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
            "electrode_group": electrode_group_name,
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
    electrode_group = first["electrode_group"]
    paramset_id = first["paramset_id"]

    # Build the keys list string.
    keys_lines = []
    for b in blocks:
        start_str = str(b["block_start"])
        end_str = str(b["block_end"])
        keys_lines.append(
            f'    {{**_base, "block_start": "{start_str}", '
            f'"block_end": "{end_str}"}},'
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
    "electrode_group": "{electrode_group}",
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


def run_post_sorting(experiment_name):
    """Populate all post-sorting tables in dependency order.

    After SpikeSorting completes via SLURM, four more tables need to be
    populated. They must run in order because each depends on the previous:

        PostProcessing  -- runs SpikeInterface sorting analyzer to compute
                           quality metrics, waveform templates, PCA, etc.
                           Depends on: SpikeSorting
        SortedSpikes    -- loads spike times, unit assignments, and electrode
                           mappings into the database.
                           Depends on: PostProcessing
        Waveform        -- loads mean waveform templates per unit.
                           Depends on: SortedSpikes
        SortingQuality  -- loads per-unit quality metrics (SNR, ISI
                           violations, firing rate, etc.).
                           Depends on: SortedSpikes

    Waveform and SortingQuality both depend on SortedSpikes but are
    independent of each other, so their order does not matter.

    Args:
        experiment_name: The experiment to post-process.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # Check that SpikeSorting has completed entries.
    sorted_count = len(spike_sorting.SpikeSorting & restriction)
    if sorted_count == 0:
        print(
            "No SpikeSorting entries found. "
            "Has the SLURM job completed successfully?"
        )
        return

    print(f"SpikeSorting entries found: {sorted_count}")

    # 1. PostProcessing -- quality metrics and analyzer extensions.
    print("\nPopulating PostProcessing...")
    spike_sorting.PostProcessing.populate(
        display_progress=True, suppress_errors=False
    )

    # 2. SortedSpikes -- spike times and unit info into DB.
    print("\nPopulating SortedSpikes...")
    spike_sorting.SortedSpikes.populate(
        display_progress=True, suppress_errors=False
    )

    # 3. Waveform -- mean waveform templates per unit.
    #    Depends on SortedSpikes but independent of SortingQuality.
    print("\nPopulating Waveform...")
    spike_sorting.Waveform.populate(
        display_progress=True, suppress_errors=False
    )

    # 4. SortingQuality -- per-unit quality metrics.
    #    Depends on SortedSpikes but independent of Waveform.
    print("\nPopulating SortingQuality...")
    spike_sorting.SortingQuality.populate(
        display_progress=True, suppress_errors=False
    )

    print("\nPost-sorting population complete.")


def verify_sorting(experiment_name):
    """Query and display sorting pipeline status for this experiment.

    Prints counts for each table in the sorting pipeline and, for
    SortedSpikes, shows unit counts per block. This gives you a quick
    overview of how far the pipeline has progressed.

    Args:
        experiment_name: The experiment to verify.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # Count entries in each pipeline table.
    task_count = len(spike_sorting.SortingTask & restriction)
    preproc_count = len(spike_sorting.PreProcessing & restriction)
    sorting_count = len(spike_sorting.SpikeSorting & restriction)
    postproc_count = len(spike_sorting.PostProcessing & restriction)
    sorted_count = len(spike_sorting.SortedSpikes & restriction)
    waveform_count = len(spike_sorting.Waveform & restriction)
    quality_count = len(spike_sorting.SortingQuality & restriction)

    print(f"Pipeline status for '{experiment_name}':")
    print(f"  SortingTask:    {task_count}")
    print(f"  PreProcessing:  {preproc_count} / {task_count}")
    print(f"  SpikeSorting:   {sorting_count} / {preproc_count}")
    print(f"  PostProcessing: {postproc_count} / {sorting_count}")
    print(f"  SortedSpikes:   {sorted_count} / {postproc_count}")
    print(f"  Waveform:       {waveform_count} / {sorted_count}")
    print(f"  SortingQuality: {quality_count} / {sorted_count}")

    # Per-block unit counts (only if SortedSpikes has entries).
    if sorted_count > 0:
        print(f"\nUnit counts per block:")
        sorted_entries = (
            spike_sorting.SortedSpikes & restriction
        ).to_dicts()

        for entry in sorted(sorted_entries, key=lambda x: x["block_start"]):
            unit_count = len(spike_sorting.SortedSpikes.Unit & entry)
            has_waveform = bool(spike_sorting.Waveform & entry)
            has_quality = bool(spike_sorting.SortingQuality & entry)
            status_parts = []
            if has_waveform:
                status_parts.append("waveforms")
            if has_quality:
                status_parts.append("quality")
            status = ", ".join(status_parts) if status_parts else "pending"

            print(
                f"  {entry['block_start']} to {entry['block_end']}: "
                f"{unit_count} units [{status}]"
            )
    else:
        print("\nNo SortedSpikes entries yet -- run post-sorting steps first.")


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 3: Spike Sorting")
    print("=" * 60)

    print("\n--- 1/5: Setup sorting prerequisites ---")
    setup_sorting_prerequisites(
        EXPERIMENT_NAME, SUBJECT, PARAMSET_ID, SORTING_METHOD, ELECTRODE_GROUP_NAME
    )

    print("\n--- 2/5: Run preprocessing ---")
    run_preprocessing(EXPERIMENT_NAME)

    print("\n--- 3/5: SLURM config for spike sorting ---")
    print_slurm_config(EXPERIMENT_NAME, SUBJECT)

    print("\n--- STOP: Submit the SLURM job and wait for it to complete ---")
    print("After sorting finishes, run this script with --post-sorting")
    print("or call run_post_sorting() and verify_sorting() manually.\n")

    # Uncomment after SLURM sorting completes:
    # print("\n--- 4/5: Run post-sorting ---")
    # run_post_sorting(EXPERIMENT_NAME)
    # print("\n--- 5/5: Verify sorting results ---")
    # verify_sorting(EXPERIMENT_NAME)

    print("=" * 60)
    print("  Step 3 (pre-sorting) complete.")
    print("  Submit SLURM job, then run post-sorting steps.")
    print("=" * 60)
