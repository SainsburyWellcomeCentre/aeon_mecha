"""Validation script for the ephys v2 pipeline test.

.. warning::

    This script is **not currently compatible with the post-restructure ephys
    schema**. It validates ``EphysEpoch.Insertion.heading``, which no longer
    exists — the part table moved to ``EphysEpochConfig.Insertion``. Running
    it will ``AttributeError``. Modernization is tracked as a follow-up; for
    the current pipeline see SPEC_EPHYS_PIPELINE.md.

Checks that the pipeline ran correctly by verifying table contents,
PK structure, and data integrity.

Usage:
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_validate           # Quick checks
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_validate --full    # Include sorting + curation + unit matching checks
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Configuration (must match ephys_v2_setup.py)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "social-ephys0.1-aeon3"
PRODUCTION_PREFIX = "aeon_"
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
N_BLOCKS = 4
BLOCK_DURATION_HOURS = 3


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def verify_prefix_or_exit():
    """These setup/validation scripts are for testing only — never run against production."""
    import datajoint as dj

    prefix = dj.config.database.database_prefix or ""
    host = dj.config.database.host or ""

    if not prefix:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is empty.")
        print(f"    Make sure you run from the repo root directory.")
        sys.exit(1)

    if prefix == PRODUCTION_PREFIX:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is '{prefix}' (production).")
        print(f"    This script is for testing only — do not run against production.")
        sys.exit(1)

    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'.")
        print(f"    This script is for testing only — do not run against production.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Check tracking
# ---------------------------------------------------------------------------
class CheckResult:
    def __init__(self):
        self.checks = []

    def record(self, group, name, passed, detail=""):
        self.checks.append({
            "group": group,
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        status = "PASS" if passed else "FAIL"
        detail_str = f" ({detail})" if detail else ""
        print(f"  [{status}] {name}{detail_str}")

    def summary(self):
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"  Summary: {passed}/{total} checks passed", end="")
        if failed > 0:
            print(f", {failed} FAILED")
            print()
            for c in self.checks:
                if not c["passed"]:
                    detail_str = f" — {c['detail']}" if c['detail'] else ""
                    print(f"    ✗ [{c['group']}] {c['name']}{detail_str}")
        else:
            print()
        print(f"{'='*60}")

        return failed == 0


# ---------------------------------------------------------------------------
# Phase 1 checks
# ---------------------------------------------------------------------------
def check_experiment(results):
    """Verify experiment setup."""
    print(f"\n--- Experiment ---\n")

    from aeon.dj_pipeline import acquisition

    exp = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Experiment", "Experiment exists", len(exp) == 1)

    dirs = (acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    has_raw = any(d["directory_type"] == "raw" for d in dirs)
    results.record("Experiment", "Raw directory registered", has_raw)
    has_raw_ephys = any(d["directory_type"] == "raw-ephys" for d in dirs)
    results.record("Experiment", "Raw-ephys directory registered (optional)", has_raw_ephys)

    subjects = (acquisition.Experiment.Subject & {"experiment_name": EXPERIMENT_NAME}).to_arrays("subject")
    results.record("Experiment", "Subject registered", len(subjects) > 0,
                    f"subjects: {list(subjects)}")


def check_epochs(results):
    """Verify epoch ingestion."""
    print(f"\n--- Epochs ---\n")

    from aeon.dj_pipeline import acquisition

    epochs = (acquisition.Epoch & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Epochs", "Epochs ingested", len(epochs) > 0, f"{len(epochs)} epochs")


def check_ephys_epochs(results):
    """Verify EphysEpoch population and probe discovery."""
    print(f"\n--- EphysEpoch ---\n")

    from aeon.dj_pipeline import ephys

    # EphysEpoch entries
    ephys_epochs = (ephys.EphysEpoch & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EphysEpoch", "EphysEpoch populated", len(ephys_epochs) > 0,
                    f"{len(ephys_epochs)} entries")

    has_ephys = [e for e in ephys_epochs if e["has_ephys"]]
    results.record("EphysEpoch", "Some epochs have ephys data", len(has_ephys) > 0,
                    f"{len(has_ephys)} with ephys")

    # ProbeInsertion
    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EphysEpoch", "ProbeInsertion exists", len(probe_insertions) > 0,
                    f"{len(probe_insertions)} insertions")

    # Check PK structure: subject should be in PK
    pi_heading = ephys.ProbeInsertion.heading
    has_subject_pk = "subject" in pi_heading.primary_key
    results.record("EphysEpoch", "ProbeInsertion PK has subject", has_subject_pk,
                    f"PK: {pi_heading.primary_key}")

    # Check probe_label is on EphysEpoch.Insertion (not ProbeInsertion)
    has_probe_label_on_pi = "probe_label" in pi_heading.names
    results.record("EphysEpoch", "probe_label NOT on ProbeInsertion", not has_probe_label_on_pi)

    insertion_heading = ephys.EphysEpoch.Insertion.heading
    has_probe_label_on_insertion = "probe_label" in insertion_heading.names
    results.record("EphysEpoch", "probe_label on EphysEpoch.Insertion", has_probe_label_on_insertion)

    # Check EphysEpoch.Insertion entries exist
    epoch_insertions = (ephys.EphysEpoch.Insertion & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EphysEpoch", "EphysEpoch.Insertion entries created",
                    len(epoch_insertions) > 0,
                    f"{len(epoch_insertions)} entries")
    if epoch_insertions:
        labels = [ei["probe_label"] for ei in epoch_insertions]
        results.record("EphysEpoch", "probe_label values populated",
                        all(l for l in labels),
                        f"labels: {labels}")


def check_ephys_chunks(results):
    """Verify EphysChunk ingestion."""
    print(f"\n--- EphysChunk ---\n")

    from aeon.dj_pipeline import ephys

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EphysChunk", "Chunks ingested", len(chunks) > 0,
                    f"{len(chunks)} chunks")

    # PK check — subject should be in PK (via ProbeInsertion FK)
    chunk_heading = ephys.EphysChunk.heading
    has_subject_pk = "subject" in chunk_heading.primary_key
    results.record("EphysChunk", "EphysChunk PK has subject", has_subject_pk,
                    f"PK: {chunk_heading.primary_key}")

    # Check files part table
    files = (ephys.EphysChunk.File & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("EphysChunk", "EphysChunk.File entries exist", len(files) > 0,
                    f"{len(files)} files")

    # Per-insertion counts
    if chunks:
        from collections import Counter
        insertion_counts = Counter(c["insertion_number"] for c in chunks)
        for ins_num, count in sorted(insertion_counts.items()):
            results.record("EphysChunk", f"Insertion {ins_num} has chunks", count > 0,
                            f"{count} chunks")


def check_blocks(results):
    """Verify EphysBlock and EphysBlockInfo."""
    print(f"\n--- EphysBlock ---\n")

    from aeon.dj_pipeline import ephys

    blocks = (ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Blocks", "EphysBlock entries exist", len(blocks) > 0,
                    f"{len(blocks)} blocks")

    # PK check — subject should be in PK (via ProbeInsertion FK)
    block_heading = ephys.EphysBlock.heading
    has_subject_pk = "subject" in block_heading.primary_key
    results.record("Blocks", "EphysBlock PK has subject", has_subject_pk,
                    f"PK: {block_heading.primary_key}")

    # BlockInfo
    block_infos = (ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Blocks", "EphysBlockInfo populated", len(block_infos) > 0,
                    f"{len(block_infos)} entries")

    # Check durations
    if block_infos:
        durations = [bi["block_duration"] for bi in block_infos]
        all_correct = all(abs(d - BLOCK_DURATION_HOURS) < 0.1 for d in durations)
        results.record("Blocks", f"Block durations ~{BLOCK_DURATION_HOURS}h", all_correct,
                        f"durations: {[f'{d:.1f}h' for d in durations]}")

    # Check chunks per block
    if block_infos:
        for bi in block_infos:
            chunk_count = len(ephys.EphysBlockInfo.Chunk & bi)
            if chunk_count == 0:
                results.record("Blocks",
                    f"Block {bi['block_start']} ins={bi['insertion_number']} has chunks",
                    False, "0 chunks")


# ---------------------------------------------------------------------------
# Phase 2 checks (sorting)
# ---------------------------------------------------------------------------
def check_sorting_pipeline(results):
    """Verify spike sorting pipeline."""
    print(f"\n--- Spike Sorting Pipeline ---\n")

    from aeon.dj_pipeline import spike_sorting

    # SortingTask
    tasks = (spike_sorting.SortingTask & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    results.record("Sorting", "SortingTask entries exist", len(tasks) > 0,
                    f"{len(tasks)} tasks")

    # PK check on SortingTask — subject should be in PK (via EphysBlock → ProbeInsertion)
    task_heading = spike_sorting.SortingTask.heading
    has_subject = "subject" in task_heading.primary_key
    results.record("Sorting", "SortingTask PK has subject", has_subject,
                    f"PK: {task_heading.primary_key}")

    # PreProcessing
    preproc = len(spike_sorting.PreProcessing & {"experiment_name": EXPERIMENT_NAME})
    results.record("Sorting", "PreProcessing populated", preproc > 0,
                    f"{preproc} entries")

    # SpikeSorting
    sorting = len(spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME})
    results.record("Sorting", "SpikeSorting completed", sorting > 0,
                    f"{sorting} entries")

    # PostProcessing
    postproc = len(spike_sorting.PostProcessing & {"experiment_name": EXPERIMENT_NAME})
    results.record("Sorting", "PostProcessing completed", postproc > 0,
                    f"{postproc} entries")

    # SortedSpikes
    sorted_count = len(spike_sorting.SortedSpikes & {"experiment_name": EXPERIMENT_NAME})
    results.record("Sorting", "SortedSpikes populated", sorted_count > 0,
                    f"{sorted_count} entries")

    units = len(spike_sorting.SortedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    results.record("Sorting", "SortedSpikes.Unit has data", units > 0,
                    f"{units} units")


# ---------------------------------------------------------------------------
# Phase 3 checks (curation + unit matching)
# ---------------------------------------------------------------------------
def check_curation(results):
    """Verify curation pipeline."""
    print(f"\n--- Curation ---\n")

    from aeon.dj_pipeline import spike_sorting_curation as curation

    official = len(curation.OfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    results.record("Curation", "OfficialCuration entries exist", official > 0,
                    f"{official} entries")

    applied = len(curation.ApplyOfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    results.record("Curation", "ApplyOfficialCuration populated", applied > 0,
                    f"{applied} entries")


def check_synced_spikes(results):
    """Verify SyncedSpikes."""
    print(f"\n--- SyncedSpikes ---\n")

    from aeon.dj_pipeline import spike_sorting

    # SyncedSpikes
    synced = len(spike_sorting.SyncedSpikes & {"experiment_name": EXPERIMENT_NAME})
    results.record("SyncedSpikes", "SyncedSpikes populated", synced > 0,
                    f"{synced} entries")

    synced_units = len(spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    results.record("SyncedSpikes", "SyncedSpikes.Unit has data", synced_units > 0,
                    f"{synced_units} units")

    # Check spike times are datetime (not float)
    if synced_units > 0:
        try:
            import numpy as np
            sample = (spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME} & dj.Top(limit=1)).fetch1(
                "spike_times"
            )
            is_datetime = np.issubdtype(sample.dtype, np.datetime64)
            results.record("SyncedSpikes", "Spike times are datetime64",
                            is_datetime,
                            f"dtype={sample.dtype}")
        except Exception as e:
            results.record("SyncedSpikes", "Spike times are datetime64", False,
                            f"Error checking: {e}")


def check_unit_matching(results):
    """Verify UnitMatching and GlobalUnit."""
    print(f"\n--- Unit Matching ---\n")

    from aeon.dj_pipeline import spike_sorting

    # UnitMatching
    matching = len(spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME})
    results.record("UnitMatching", "UnitMatching populated", matching > 0,
                    f"{matching} entries")

    # GlobalUnit
    global_units = len(spike_sorting.GlobalUnit & {"experiment_name": EXPERIMENT_NAME})
    results.record("UnitMatching", "GlobalUnit entries exist", global_units > 0,
                    f"{global_units} units")

    # GlobalUnit PK check — subject should be in PK
    gu_heading = spike_sorting.GlobalUnit.heading
    has_subject = "subject" in gu_heading.primary_key
    results.record("UnitMatching", "GlobalUnit PK has subject", has_subject,
                    f"PK: {gu_heading.primary_key}")

    # UnitMatching.Unit — maps block units to global units
    matched_units = len(spike_sorting.UnitMatching.Unit & {"experiment_name": EXPERIMENT_NAME})
    results.record("UnitMatching", "UnitMatching.Unit has data", matched_units > 0,
                    f"{matched_units} entries")

    # UnitMatching.Spikes — ownership convention
    spikes = len(spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME})
    results.record("UnitMatching", "UnitMatching.Spikes has data", spikes > 0,
                    f"{spikes} entries")

    # Check spike times are datetime64[ns]
    if spikes > 0:
        try:
            import numpy as np
            sample = (spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME} & dj.Top(limit=1)).fetch1(
                "spike_times"
            )
            is_datetime = np.issubdtype(sample.dtype, np.datetime64)
            results.record("UnitMatching", "Spikes spike_times are datetime64",
                            is_datetime,
                            f"dtype={sample.dtype}")
        except Exception as e:
            results.record("UnitMatching", "Spikes spike_times are datetime64", False,
                            f"Error checking: {e}")

    # Check no duplicate (global_unit, chunk_start) pairs (ownership convention)
    if spikes > 0:
        try:
            spikes_data = (spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME}).proj(
                "global_unit", "chunk_start"
            ).to_dicts()
            pairs = [(s["global_unit"], s["chunk_start"]) for s in spikes_data]
            has_dupes = len(pairs) != len(set(pairs))
            results.record("UnitMatching", "No duplicate (global_unit, chunk_start) pairs",
                            not has_dupes,
                            f"{len(pairs)} pairs, {len(set(pairs))} unique")
        except Exception as e:
            results.record("UnitMatching", "No duplicate (global_unit, chunk_start) pairs", False,
                            f"Error checking: {e}")

    # Sanity check: GlobalUnit count should be reasonable
    # (close to max units in a single block, not sum of all blocks)
    if global_units > 0 and matching > 0:
        try:
            sorted_units_per_block = []
            blocks = (spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME}).keys()
            for bk in blocks:
                n = len(spike_sorting.SortedSpikes.Unit & bk)
                sorted_units_per_block.append(n)
            max_per_block = max(sorted_units_per_block) if sorted_units_per_block else 0
            sum_all = sum(sorted_units_per_block)
            # GlobalUnit count should be between max_per_block and sum_all
            # If matching works well, it should be closer to max_per_block
            reasonable = global_units <= sum_all
            results.record("UnitMatching", "GlobalUnit count is reasonable",
                            reasonable,
                            f"global={global_units}, max_per_block={max_per_block}, sum_all={sum_all}")
        except Exception as e:
            results.record("UnitMatching", "GlobalUnit count is reasonable", False,
                            f"Error checking: {e}")


def check_subject_in_pks(results):
    """Verify 'subject' IS in the right ephys table PKs."""
    print(f"\n--- PK Structure (subject in PK) ---\n")

    from aeon.dj_pipeline import ephys, spike_sorting

    tables_to_check = [
        ("ephys.ProbeInsertion", ephys.ProbeInsertion),
        ("ephys.EphysChunk", ephys.EphysChunk),
        ("ephys.EphysBlock", ephys.EphysBlock),
        ("ephys.EphysBlockInfo", ephys.EphysBlockInfo),
        ("spike_sorting.SortingTask", spike_sorting.SortingTask),
        ("spike_sorting.PreProcessing", spike_sorting.PreProcessing),
        ("spike_sorting.SpikeSorting", spike_sorting.SpikeSorting),
        ("spike_sorting.PostProcessing", spike_sorting.PostProcessing),
        ("spike_sorting.SortedSpikes", spike_sorting.SortedSpikes),
        ("spike_sorting.SyncedSpikes", spike_sorting.SyncedSpikes),
        ("spike_sorting.GlobalUnit", spike_sorting.GlobalUnit),
        ("spike_sorting.UnitMatching", spike_sorting.UnitMatching),
    ]

    all_have_subject = True
    for name, table in tables_to_check:
        has_subject = "subject" in table.heading.primary_key
        if not has_subject:
            results.record("PKStructure", f"{name} PK has subject", False,
                            f"PK: {table.heading.primary_key}")
            all_have_subject = False

    if all_have_subject:
        results.record("PKStructure", "'subject' in all ephys/sorting PKs", True,
                        f"Checked {len(tables_to_check)} tables")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate the ephys v2 pipeline test.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include sorting + curation + unit matching checks (requires all phases complete)",
    )
    args = parser.parse_args()

    verify_prefix_or_exit()

    print("=" * 60)
    print("  Ephys v2 Pipeline Validation")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Mode: {'Full' if args.full else 'Quick (Phase 1 only)'}")
    print("=" * 60)

    results = CheckResult()

    try:
        # Always check these (Phase 1)
        check_experiment(results)
        check_epochs(results)
        check_ephys_epochs(results)
        check_ephys_chunks(results)
        check_blocks(results)
        check_subject_in_pks(results)

        if args.full:
            check_sorting_pipeline(results)
            check_curation(results)
            check_synced_spikes(results)
            check_unit_matching(results)

    except Exception as e:
        print(f"\n  ✗ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
