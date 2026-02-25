"""Validation script for the ephys v2 PK revamp test.

Checks that the pipeline ran correctly by verifying table contents,
PK structure, and data integrity.

Usage:
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_validate           # Quick checks
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_validate --full    # Include sorting + curation checks
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Configuration (must match ephys_v2_setup.py)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "social-ephys0.1-aeon3"
EXPECTED_PREFIX = "elissas_aeon_ephys_test_"
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
N_BLOCKS = 4
BLOCK_DURATION_HOURS = 3


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def verify_prefix_or_exit():
    import datajoint as dj

    if "custom" not in dj.config:
        dj.config["custom"] = {}

    prefix = dj.config["custom"].get("database.prefix", "")
    host = dj.config.get("database.host", "")

    if prefix != EXPECTED_PREFIX:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is '{prefix}'")
        print(f"    Expected: '{EXPECTED_PREFIX}'")
        if not prefix:
            print(f"    Make sure you run from the aeon_mecha_ephys/ directory.")
        sys.exit(1)

    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'")
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

    exp = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("Experiment", "Experiment exists", len(exp) == 1)

    dirs = (acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    has_raw = any(d["directory_type"] == "raw" for d in dirs)
    results.record("Experiment", "Raw directory registered", has_raw)


def check_epochs(results):
    """Verify epoch ingestion."""
    print(f"\n--- Epochs ---\n")

    from aeon.dj_pipeline import acquisition

    epochs = (acquisition.Epoch & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("Epochs", "Epochs ingested", len(epochs) > 0, f"{len(epochs)} epochs")


def check_ephys_epochs(results):
    """Verify EphysEpoch population and probe discovery."""
    print(f"\n--- EphysEpoch ---\n")

    from aeon.dj_pipeline import ephys

    # EphysEpoch entries
    ephys_epochs = (ephys.EphysEpoch & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("EphysEpoch", "EphysEpoch populated", len(ephys_epochs) > 0,
                    f"{len(ephys_epochs)} entries")

    has_ephys = [e for e in ephys_epochs if e["has_ephys"]]
    results.record("EphysEpoch", "Some epochs have ephys data", len(has_ephys) > 0,
                    f"{len(has_ephys)} with ephys")

    # ProbeInsertion auto-created
    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("EphysEpoch", "ProbeInsertion auto-created", len(probe_insertions) > 0,
                    f"{len(probe_insertions)} insertions")

    # Check PK structure: no 'subject' column
    pi_heading = ephys.ProbeInsertion.heading
    has_subject_pk = "subject" in pi_heading.primary_key
    results.record("EphysEpoch", "ProbeInsertion PK has NO subject", not has_subject_pk,
                    f"PK: {pi_heading.primary_key}")

    # Check probe_label field exists
    has_probe_label = "probe_label" in pi_heading.names
    results.record("EphysEpoch", "ProbeInsertion has probe_label field", has_probe_label)

    # Check probe_label values
    if probe_insertions and has_probe_label:
        labels = [pi["probe_label"] for pi in probe_insertions]
        results.record("EphysEpoch", "probe_label values populated",
                        all(l for l in labels),
                        f"labels: {labels}")


def check_ephys_chunks(results):
    """Verify EphysChunk ingestion."""
    print(f"\n--- EphysChunk ---\n")

    from aeon.dj_pipeline import ephys

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("EphysChunk", "Chunks ingested", len(chunks) > 0,
                    f"{len(chunks)} chunks")

    # PK check
    chunk_heading = ephys.EphysChunk.heading
    has_subject_pk = "subject" in chunk_heading.primary_key
    results.record("EphysChunk", "EphysChunk PK has NO subject", not has_subject_pk,
                    f"PK: {chunk_heading.primary_key}")

    # Check files part table
    files = (ephys.EphysChunk.File & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
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

    blocks = (ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("Blocks", "EphysBlock entries exist", len(blocks) > 0,
                    f"{len(blocks)} blocks")

    # PK check
    block_heading = ephys.EphysBlock.heading
    has_subject_pk = "subject" in block_heading.primary_key
    results.record("Blocks", "EphysBlock PK has NO subject", not has_subject_pk,
                    f"PK: {block_heading.primary_key}")

    # BlockInfo
    block_infos = (ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
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
    tasks = (spike_sorting.SortingTask & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    results.record("Sorting", "SortingTask entries exist", len(tasks) > 0,
                    f"{len(tasks)} tasks")

    # PK check on SortingTask
    task_heading = spike_sorting.SortingTask.heading
    has_subject = "subject" in task_heading.primary_key
    results.record("Sorting", "SortingTask PK has NO subject", not has_subject,
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
# Phase 3 checks (curation + downstream)
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


def check_synced_and_matching(results):
    """Verify SyncedSpikes, UnitMatching, ChunkedSpikeTimes."""
    print(f"\n--- SyncedSpikes & UnitMatching ---\n")

    from aeon.dj_pipeline import spike_sorting

    # SyncedSpikes
    synced = len(spike_sorting.SyncedSpikes & {"experiment_name": EXPERIMENT_NAME})
    results.record("Downstream", "SyncedSpikes populated", synced > 0,
                    f"{synced} entries")

    synced_units = len(spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    results.record("Downstream", "SyncedSpikes.Unit has data", synced_units > 0,
                    f"{synced_units} units")

    # Check spike times are datetime (not float)
    if synced_units > 0:
        try:
            import numpy as np
            sample = (spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME}).fetch(
                "spike_times", limit=1
            )[0]
            is_datetime = np.issubdtype(sample.dtype, np.datetime64)
            results.record("Downstream", "Spike times are datetime64",
                            is_datetime,
                            f"dtype={sample.dtype}")
        except Exception as e:
            results.record("Downstream", "Spike times are datetime64", False,
                            f"Error checking: {e}")

    # UnitMatching
    matching = len(spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME})
    results.record("Downstream", "UnitMatching populated", matching > 0,
                    f"{matching} entries")

    universal = len(spike_sorting.UniversalUnit & {"experiment_name": EXPERIMENT_NAME})
    results.record("Downstream", "UniversalUnit entries", universal > 0,
                    f"{universal} units")

    # UniversalUnit PK check
    uu_heading = spike_sorting.UniversalUnit.heading
    has_subject = "subject" in uu_heading.primary_key
    results.record("Downstream", "UniversalUnit PK has NO subject", not has_subject,
                    f"PK: {uu_heading.primary_key}")

    # ChunkedSpikeTimes
    chunked = len(spike_sorting.ChunkedSpikeTimes & {"experiment_name": EXPERIMENT_NAME})
    results.record("Downstream", "ChunkedSpikeTimes populated", chunked > 0,
                    f"{chunked} entries")

    # Check ChunkedSpikeTimes dtype
    if chunked > 0:
        try:
            import numpy as np
            sample = (spike_sorting.ChunkedSpikeTimes & {"experiment_name": EXPERIMENT_NAME}).fetch(
                "spike_times", limit=1
            )[0]
            is_datetime = np.issubdtype(sample.dtype, np.datetime64)
            results.record("Downstream", "ChunkedSpikeTimes are datetime64",
                            is_datetime,
                            f"dtype={sample.dtype}")
        except Exception as e:
            results.record("Downstream", "ChunkedSpikeTimes are datetime64", False,
                            f"Error checking: {e}")


def check_no_subject_anywhere(results):
    """Verify 'subject' is NOT in any ephys table PK."""
    print(f"\n--- PK Structure (no subject) ---\n")

    from aeon.dj_pipeline import ephys, spike_sorting, spike_sorting_curation

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
        ("spike_sorting.UnitMatching", spike_sorting.UnitMatching),
        ("spike_sorting.UniversalUnit", spike_sorting.UniversalUnit),
        ("spike_sorting.ChunkedSpikeTimes", spike_sorting.ChunkedSpikeTimes),
    ]

    all_clean = True
    for name, table in tables_to_check:
        has_subject = "subject" in table.heading.primary_key
        if has_subject:
            results.record("PKStructure", f"{name} PK has NO subject", False,
                            f"PK: {table.heading.primary_key}")
            all_clean = False

    if all_clean:
        results.record("PKStructure", "No 'subject' in any ephys/sorting PK", True,
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
        help="Include sorting + curation checks (requires Phase 2+3 to be complete)",
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
        check_no_subject_anywhere(results)

        if args.full:
            check_sorting_pipeline(results)
            check_curation(results)
            check_synced_and_matching(results)

    except Exception as e:
        print(f"\n  ✗ Validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
