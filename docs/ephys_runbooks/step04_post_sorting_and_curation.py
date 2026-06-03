"""04 -- Post-Sorting and Curation
================================
After the SLURM spike sorting job completes, this script:
  1. Populates post-sorting tables (PostProcessing, SortedSpikes,
     Waveform, SortingQuality)
  2. Verifies sorting results
  3. Runs auto-approval curation (or guides you through manual curation)

The post-sorting tables must be populated in order because each depends
on the previous:

    PostProcessing  -- runs SpikeInterface sorting analyzer to compute
                       quality metrics, waveform templates, PCA, etc.
    SortedSpikes    -- loads spike times and unit info into the DB
    Waveform        -- loads mean waveform templates per unit
    SortingQuality  -- loads per-unit quality metrics (SNR, ISI, etc.)

After post-sorting, curation labels units as good, MUA, or noise.
Two paths are supported:
    Path A -- Auto-approval (accept sorting as-is, good for testing)
    Path B -- Manual curation via SpikeInterface GUI

Run from the repo root on an HPC compute node:

    uv run python docs/ephys_runbooks/step04_post_sorting_and_curation.py
"""

# --------------------------------------------------------------------------
# Configuration -- edit these for your experiment
# --------------------------------------------------------------------------

from datetime import UTC

EXPERIMENT_NAME = "abcGolden01-aeonx1"


# --------------------------------------------------------------------------
# Post-sorting functions
# --------------------------------------------------------------------------


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
    """
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # Check that SpikeSorting has completed entries.
    sorted_count = len(spike_sorting.SpikeSorting & restriction)
    if sorted_count == 0:
        print("No SpikeSorting entries found. Has the SLURM job completed successfully?")
        return

    print(f"SpikeSorting entries found: {sorted_count}")

    # 1. PostProcessing -- quality metrics and analyzer extensions.
    print("\nPopulating PostProcessing...")
    spike_sorting.PostProcessing.populate(display_progress=True, suppress_errors=False)

    # 2. SortedSpikes -- spike times and unit info into DB.
    print("\nPopulating SortedSpikes...")
    spike_sorting.SortedSpikes.populate(display_progress=True, suppress_errors=False)

    # 3. Waveform -- mean waveform templates per unit.
    print("\nPopulating Waveform...")
    spike_sorting.Waveform.populate(display_progress=True, suppress_errors=False)

    # 4. SortingQuality -- per-unit quality metrics.
    print("\nPopulating SortingQuality...")
    spike_sorting.SortingQuality.populate(display_progress=True, suppress_errors=False)

    print("\nPost-sorting population complete.")


def verify_sorting(experiment_name):
    """Print sorting pipeline status: table counts and units per block."""
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

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

    if sorted_count > 0:
        print("\nUnit counts per block:")
        sorted_entries = (spike_sorting.SortedSpikes & restriction).to_dicts()

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

            print(f"  {entry['block_start']} to {entry['block_end']}: {unit_count} units [{status}]")
    else:
        print("\nNo SortedSpikes entries yet.")


# --------------------------------------------------------------------------
# Curation functions
# --------------------------------------------------------------------------


def auto_approve_curation(experiment_name):
    """Auto-approve all sorting results (Path A).

    Creates ManualCuration + OfficialCuration entries that accept the
    automated sorting results without modification. Good for testing or
    when you plan to curate later.

    Uses curation_id=0 with parent_curation_id=-1 and no curation file.
    ApplyOfficialCuration detects this and simply updates the curation_id
    on existing SortedSpikes entries -- no spike data is deleted or
    re-computed.
    """
    from datetime import datetime

    from aeon.dj_pipeline import spike_sorting
    from aeon.dj_pipeline import spike_sorting_curation as curation

    # Ensure CurationMethod exists.
    if not (curation.CurationMethod & {"curation_method": "SpikeInterface"}):
        curation.CurationMethod.insert1(
            {"curation_method": "SpikeInterface"},
            skip_duplicates=True,
        )

    sorting_entries = (spike_sorting.SpikeSorting & {"experiment_name": experiment_name}).keys()
    now = datetime.now(UTC)

    for sorting_key in sorting_entries:
        mc_key = {**sorting_key, "curation_id": 0}
        if not (curation.ManualCuration & mc_key):
            curation.ManualCuration.insert1(
                {
                    **mc_key,
                    "curation_datetime": now,
                    "parent_curation_id": -1,
                    "curation_method": "SpikeInterface",
                    "description": "Auto-approved: no manual curation applied",
                },
                skip_duplicates=True,
            )

        sorted_pk = {k: sorting_key[k] for k in spike_sorting.SortedSpikes.primary_key}
        if not (curation.OfficialCuration & sorted_pk):
            curation.OfficialCuration.insert1(
                {**sorted_pk, "curation_id": 0},
                skip_duplicates=True,
            )

    curation.ApplyOfficialCuration.populate(display_progress=True)
    print("Auto-approval curation complete.")


def print_manual_curation_guide():
    """Print instructions for manual curation (Path B)."""
    print("""
--- Manual Curation (Path B) ---

If you want to review units interactively instead of auto-approving:

1. LAUNCH THE GUI
   Open aeon/dj_pipeline/scripts/launch_si_gui.py, fill in the `key`
   dictionary with your sorting task's primary key, then run it.
   The SpikeInterface GUI opens with the sorting analyzer loaded.

2. REVIEW AND LABEL UNITS
   - Label units as "good", "mua" (multi-unit activity), or "noise"
   - Merge units that are the same neuron split across two IDs
   - Remove units that are clearly noise artifacts
   - Click "Save in analyzer" periodically to save progress

3. SAVE CURATION TO THE DATABASE
   Open aeon/dj_pipeline/scripts/save_curation.py, fill in the same
   `key` dictionary, then run it:

       from aeon.dj_pipeline import spike_sorting_curation
       curation_id = spike_sorting_curation.save_manual_curation(
           key, description="First pass curation"
       )

4. MAKE OFFICIAL AND APPLY
       spike_sorting_curation.make_curation_official(key, curation_id)
       from aeon.dj_pipeline import spike_sorting_curation as curation
       curation.ApplyOfficialCuration.populate(display_progress=True)

   NOTE: Applying manual curation DELETES existing SortedSpikes and all
   downstream entries (Waveform, SortingQuality, SyncedSpikes), then
   re-populates them with curated data. You must re-run:

       spike_sorting.SortedSpikes.populate(display_progress=True)
       spike_sorting.Waveform.populate(display_progress=True)
       spike_sorting.SortingQuality.populate(display_progress=True)
       spike_sorting.SyncedSpikes.populate(display_progress=True)

ITERATING: To refine an existing curation, pass parent_curation_id:
    spike_sorting_curation.launch_spikeinterface_gui(key, parent_curation_id=1)

RESTORING RAW: To undo an official curation:
    spike_sorting_curation.restore_raw_sorting(key)

See docs/specs/SPEC_SPIKE_SORTING_CURATION.md for full technical details.
""")


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 4: Post-Sorting and Curation")
    print("=" * 60)

    print("\n--- 1/4: Populate post-sorting tables ---")
    run_post_sorting(EXPERIMENT_NAME)

    print("\n--- 2/4: Verify sorting results ---")
    verify_sorting(EXPERIMENT_NAME)

    print("\n--- 3/4: Auto-approve curation ---")
    auto_approve_curation(EXPERIMENT_NAME)

    print("\n--- 4/4: Manual curation guide ---")
    print_manual_curation_guide()

    print("=" * 60)
    print("  Step 4 complete. Continue to Step 5 (unit matching).")
    print("=" * 60)
