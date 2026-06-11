"""05 -- Unit Matching
===================
Synchronize spike times to the behavioral clock, then match units across
overlapping ephys blocks to establish persistent neuron identities.

After spike sorting (step 3) and curation (step 4), spike times are still
in the ONIX hardware clock. This step:
    1. SyncedSpikes   -- translates spike times from the ONIX clock to the
                         HARP clock (the behavioral/acquisition system clock).
                         This alignment is required so spike times can be
                         compared to behavioral events (rewards, entries, etc.).
    2. UnitMatching   -- compares spike trains in the overlapping regions
                         between adjacent blocks to assign persistent neuron
                         identities ("global units") across blocks. Uses the
                         spike_time_overlap method with a configurable time
                         window (delta_time).

Prerequisites:
    - SortedSpikes must exist for every block (step 4, post-sorting)
    - ApplyOfficialCuration must exist for every block (step 4, either
      auto-approval or manual curation satisfies this gate)
    - EphysBlock entries with overlapping time windows (step 2)

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_runbooks/step05_unit_matching.py
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
MATCHING_PARAMSET_ID = 1

# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------


def sync_spikes(experiment_name):
    """Populate SyncedSpikes to align spike times from ONIX clock to HARP clock.

    Raw spike sorting produces spike indices relative to the concatenated
    binary recording. SyncedSpikes converts these indices back to actual
    timestamps synchronized to the HARP clock system. This is essential
    because:
        - The ONIX clock is the ephys hardware clock (runs independently)
        - The HARP clock is the behavioral/acquisition system clock
        - All behavioral events (rewards, beam breaks, video frames) use
          HARP timestamps
        - To correlate neural activity with behavior, spike times must be
          in the same clock domain

    The conversion uses sync models (linear regression fits) that were
    computed during EphysChunk ingestion. Each model maps a window of
    ONIX timestamps to HARP timestamps.

    SyncedSpikes.Unit stores the result: one entry per (unit, chunk) with
    spike_times as datetime64[ns] in the HARP clock.

    Args:
        experiment_name: The experiment to sync spikes for.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # Show current state before populating.
    sorted_count = len(spike_sorting.SortedSpikes & restriction)
    synced_before = len(spike_sorting.SyncedSpikes & restriction)
    print(f"SortedSpikes entries:       {sorted_count}")
    print(f"SyncedSpikes entries (pre): {synced_before}")

    if sorted_count == 0:
        print("No SortedSpikes entries found. Run step 4 (post-sorting) first.")
        return

    pending = sorted_count - synced_before
    if pending == 0:
        print("All blocks already synced. Nothing to do.")
        return

    print(f"Pending blocks to sync: {pending}")
    print("Running SyncedSpikes.populate()...")
    print(
        "  (This loads clock data and sync models for each block, then "
        "converts spike indices to HARP timestamps. May take a while "
        "for blocks with many units.)"
    )
    spike_sorting.SyncedSpikes.populate(display_progress=True, suppress_errors=False)

    synced_after = len(spike_sorting.SyncedSpikes & restriction)
    print(f"SyncedSpikes entries (post): {synced_after}")


def run_unit_matching(experiment_name, subject, matching_paramset_id):
    """Set up the matching parameter set and populate UnitMatching.

    Unit matching identifies the same neuron across overlapping ephys
    blocks. The algorithm (spike_time_overlap) works by:
        1. Starting from a "seed" block (the first block in time)
        2. For each subsequent block, comparing spike trains in the
           overlapping time window with previously matched blocks
        3. Using SpikeInterface's compare_two_sorters to find pairs of
           units with highly correlated spike times within delta_time
        4. Matched units inherit existing global unit IDs; unmatched
           units are assigned new global unit IDs

    This function:
        a) Inserts UnitMatchingParamSet (idempotent) with the seed block
           start time and matching parameters
        b) Populates UnitMatching, which processes blocks outward from
           the seed in both directions

    HARD PREREQUISITE: ApplyOfficialCuration must exist for every block.
    This is enforced in UnitMatching.key_source -- blocks without an
    ApplyOfficialCuration entry are invisible to populate(). Either
    auto-approval (Path A) or manual curation (Path B) from step 4
    satisfies this gate.

    Args:
        experiment_name: The experiment to run matching for.
        subject: Subject identifier (used to filter blocks).
        matching_paramset_id: Integer ID for the parameter set.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import ephys, spike_sorting

    restriction = {"experiment_name": experiment_name}

    # ------------------------------------------------------------------
    # a) Insert UnitMatchingParamSet if it does not exist
    # ------------------------------------------------------------------
    # The seed block is the first block in time -- unit matching
    # propagates outward from here in both directions.
    block_starts = (ephys.EphysBlock & restriction).to_arrays("block_start")

    if len(block_starts) == 0:
        print("No EphysBlock entries found. Run step 2 (define_blocks) first.")
        return

    first_block_start = min(block_starts)
    print(f"Seed block start (earliest block): {first_block_start}")

    paramset_key = {"matching_paramset_id": matching_paramset_id}
    if not (spike_sorting.UnitMatchingParamSet & paramset_key):
        spike_sorting.UnitMatchingParamSet.insert1(
            {
                "matching_paramset_id": matching_paramset_id,
                "matching_method": "spike_time_overlap",
                "seed_block_start": first_block_start,
                "matching_paramset_description": ("Spike time overlap matching, delta=0.4ms"),
                # delta_time is the maximum time difference (in ms) for
                # two spikes to be considered a match. 0.4ms is a
                # conservative default that works well for Neuropixels.
                "params": {"delta_time": 0.4},
            },
            skip_duplicates=True,
        )
        print(
            f"Inserted UnitMatchingParamSet: "
            f"matching_paramset_id={matching_paramset_id}, "
            f"method=spike_time_overlap, delta_time=0.4ms"
        )
    else:
        print(f"UnitMatchingParamSet already exists: matching_paramset_id={matching_paramset_id}")

    # ------------------------------------------------------------------
    # b) Check the ApplyOfficialCuration prerequisite
    # ------------------------------------------------------------------
    from aeon.dj_pipeline import spike_sorting_curation

    synced_count = len(spike_sorting.SyncedSpikes & restriction)
    curated_count = len(spike_sorting_curation.ApplyOfficialCuration & restriction)
    print(f"\nSyncedSpikes entries:           {synced_count}")
    print(f"ApplyOfficialCuration entries:  {curated_count}")

    if curated_count == 0:
        print(
            "\nWARNING: No ApplyOfficialCuration entries found. "
            "UnitMatching requires curation for every block. "
            "Run step 4 first (auto-approval or manual curation)."
        )
        return

    if curated_count < synced_count:
        print(
            f"\nWARNING: Only {curated_count}/{synced_count} blocks have "
            f"curation applied. Uncurated blocks will be skipped by "
            f"UnitMatching. Run step 4 to curate remaining blocks."
        )

    # ------------------------------------------------------------------
    # c) Populate UnitMatching
    # ------------------------------------------------------------------
    matched_before = len(spike_sorting.UnitMatching & restriction)
    print(f"\nUnitMatching entries (pre): {matched_before}")

    print("Running UnitMatching.populate()...")
    print(
        "  (Processes blocks outward from the seed block. For each block, "
        "compares spike trains in the overlap window with previously "
        "matched blocks to assign global unit IDs.)"
    )
    spike_sorting.UnitMatching.populate(display_progress=True, suppress_errors=False)

    matched_after = len(spike_sorting.UnitMatching & restriction)
    print(f"UnitMatching entries (post): {matched_after}")


def verify_matching(experiment_name):
    """Query and display unit matching results for this experiment.

    Prints counts for all matching-related tables and shows a summary
    of global unit assignments per probe insertion. This gives you a
    quick overview of how many persistent neurons were identified across
    blocks.

    Args:
        experiment_name: The experiment to verify.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # ------------------------------------------------------------------
    # Overall counts
    # ------------------------------------------------------------------
    synced_count = len(spike_sorting.SyncedSpikes & restriction)
    matching_count = len(spike_sorting.UnitMatching & restriction)
    global_units = len(spike_sorting.GlobalUnit & restriction)
    matched_units = len(spike_sorting.UnitMatching.Unit & restriction)
    matched_spikes = len(spike_sorting.UnitMatching.Spikes & restriction)

    print(f"Matching results for '{experiment_name}':")
    print(f"  SyncedSpikes entries:       {synced_count}")
    print(f"  UnitMatching entries:       {matching_count}")
    print(f"  GlobalUnit count:           {global_units}")
    print(f"  UnitMatching.Unit entries:  {matched_units} (unit-to-global mappings)")
    print(f"  UnitMatching.Spikes entries: {matched_spikes} (spike data per global unit per chunk)")

    if global_units == 0:
        print("\nNo global units found. Run unit matching first.")
        return

    # ------------------------------------------------------------------
    # Per-insertion summary of global units
    # ------------------------------------------------------------------
    print("\nGlobal unit summary:")
    global_entries = (spike_sorting.GlobalUnit & restriction).to_dicts()

    # Group by (subject, insertion_number)
    from collections import defaultdict

    by_insertion = defaultdict(list)
    for entry in global_entries:
        key = (entry["subject"], entry["insertion_number"])
        by_insertion[key].append(entry["global_unit"])

    for (subj, ins_num), units in sorted(by_insertion.items()):
        print(
            f"  Subject={subj}, insertion={ins_num}: "
            f"{len(units)} global units "
            f"(IDs {min(units)}-{max(units)})"
        )

    # ------------------------------------------------------------------
    # Per-block breakdown: how many units matched vs. new
    # ------------------------------------------------------------------
    if matching_count > 0:
        print("\nPer-block unit matching breakdown:")
        matching_entries = (spike_sorting.UnitMatching & restriction).to_dicts()

        for entry in sorted(matching_entries, key=lambda x: x["block_start"]):
            unit_count = len(spike_sorting.UnitMatching.Unit & entry)
            print(
                f"  {entry['block_start']} to {entry['block_end']}: "
                f"{unit_count} units matched "
                f"({entry['execution_duration']:.2f} hr)"
            )


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 5: Unit Matching")
    print("=" * 60)

    print("\n--- 1/3: Sync spikes ---")
    sync_spikes(EXPERIMENT_NAME)

    print("\n--- 2/3: Run unit matching ---")
    run_unit_matching(EXPERIMENT_NAME, SUBJECT, MATCHING_PARAMSET_ID)

    print("\n--- 3/3: Verify matching ---")
    verify_matching(EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("  Step 5 complete. Ready for analysis (Step 6).")
    print("=" * 60)
