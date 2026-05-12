"""
02 -- Define Blocks
===================
Define time windows ("blocks") for spike sorting.

Continuous ephys recordings can span many hours or even days, but spike
sorting has practical limits on how much data can be processed in one go
(~30 hours max before GPU memory becomes an issue). Blocks let you carve
the recording into manageable pieces.

Blocks also enable *unit matching*: by making adjacent blocks overlap,
the same neuron appears in two separate sorting runs, and downstream
algorithms can link those detections into a single persistent unit ID.

Production parameters:
    20-hour blocks with 5-hour overlap

This guide uses small test parameters for fast iteration:
    30-minute blocks with 10-minute overlap, 3 blocks total (~70 min)

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_guide/step02_define_blocks.py
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
from datetime import datetime, timedelta

# --------------------------------------------------------------------------
# Configuration -- edit these for your experiment
# --------------------------------------------------------------------------

EXPERIMENT_NAME = "abcGolden01-aeonx1"
SUBJECT = "IAA-1147881"

# Test subset: small blocks for fast iteration.
# 30-minute blocks with 10-minute overlap produce 3 blocks spanning ~70 min.
BLOCK_DURATION_MIN = 30
OVERLAP_MIN = 10
N_BLOCKS = 3

# Production params (uncomment for real use):
# BLOCK_DURATION_HRS = 20
# OVERLAP_HRS = 5


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def query_available_data(experiment_name):
    """Fetch EphysChunk data and print a summary of the available recording window.

    This tells you what raw data the pipeline knows about, so you can plan
    block boundaries accordingly. Each chunk is one contiguous segment of
    recorded data (typically ~10 minutes for this golden dataset, or ~1 hour
    in production recordings).

    Args:
        experiment_name: The experiment to query.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline.ephys import EphysChunk, ProbeInsertion

    chunks = (EphysChunk & {"experiment_name": experiment_name}).to_dicts()
    if not chunks:
        print(f"No EphysChunk entries found for '{experiment_name}'.")
        print("Run step 1 (register_experiment) first to ingest chunks.")
        return

    starts = [c["chunk_start"] for c in chunks]
    ends = [c["chunk_end"] for c in chunks]
    earliest = min(starts)
    latest = max(ends)
    total_duration = latest - earliest

    print(f"Experiment: {experiment_name}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Earliest chunk start: {earliest}")
    print(f"Latest chunk end:     {latest}")
    print(f"Total span:           {total_duration}")

    # Per-ProbeInsertion breakdown (useful when multiple probes are implanted).
    insertions = (
        ProbeInsertion & {"experiment_name": experiment_name}
    ).to_dicts()
    if len(insertions) > 1:
        print(f"\nPer-ProbeInsertion breakdown ({len(insertions)} probes):")
    else:
        print(f"\nProbeInsertion breakdown:")

    for pi in insertions:
        pi_key = {
            "experiment_name": pi["experiment_name"],
            "subject": pi["subject"],
            "insertion_number": pi["insertion_number"],
        }
        pi_chunks = (EphysChunk & pi_key).to_dicts()
        if pi_chunks:
            pi_starts = [c["chunk_start"] for c in pi_chunks]
            pi_ends = [c["chunk_end"] for c in pi_chunks]
            print(
                f"  insertion {pi['insertion_number']} "
                f"(subject={pi['subject']}): "
                f"{len(pi_chunks)} chunks, "
                f"{min(pi_starts)} to {max(pi_ends)}"
            )
        else:
            print(
                f"  insertion {pi['insertion_number']} "
                f"(subject={pi['subject']}): 0 chunks"
            )


def calculate_block_boundaries(start_time, block_duration, overlap, n_blocks=None):
    """Compute block start/end times from a starting timestamp.

    This is a pure function -- no database access. It calculates evenly
    spaced, overlapping time windows.

    How the math works:
        advance = block_duration - overlap

        Each block starts `advance` later than the previous one:
            block 0: [start, start + duration)
            block 1: [start + advance, start + advance + duration)
            block 2: [start + 2*advance, start + 2*advance + duration)
            ...

        The overlap between consecutive blocks is:
            block[i].end - block[i+1].start = duration - advance = overlap

    Args:
        start_time: datetime -- beginning of the first block.
        block_duration: timedelta -- length of each block.
        overlap: timedelta -- how much adjacent blocks should overlap.
        n_blocks: int or None -- exact number of blocks to create. If None,
            blocks are created until start_time + block_duration would
            exceed start_time + total available duration (the caller is
            responsible for passing a sensible start_time in that case).

    Returns:
        List of (block_start, block_end) datetime tuples.
    """
    advance = block_duration - overlap
    if advance <= timedelta(0):
        raise ValueError(
            f"Overlap ({overlap}) must be less than block duration "
            f"({block_duration}). Otherwise blocks never advance."
        )

    blocks = []
    i = 0
    while True:
        block_start = start_time + i * advance
        block_end = block_start + block_duration

        if n_blocks is not None:
            # Fixed number of blocks: stop when we have enough.
            if i >= n_blocks:
                break
        else:
            # No fixed count: the caller should stop us externally, but
            # as a safety valve, break if we have generated 1000 blocks.
            if i >= 1000:
                print("WARNING: hit 1000-block safety limit. Stopping.")
                break

        blocks.append((block_start, block_end))
        i += 1

    return blocks


def create_blocks(
    experiment_name,
    subject=None,
    block_duration_min=30,
    overlap_min=10,
    n_blocks=3,
):
    """Create EphysBlock entries and populate EphysBlockInfo.

    Steps:
        1. Query ProbeInsertion for this experiment (optionally filtered by
           subject) to get all (experiment_name, subject, insertion_number)
           tuples.
        2. Find the earliest EphysChunk start time -- this anchors block 1.
        3. Calculate block boundaries using the specified duration and overlap.
        4. Insert one EphysBlock entry per (ProbeInsertion, block) combination.
        5. Populate EphysBlockInfo to compute metadata (duration, chunk
           associations, channel mappings).

    Args:
        experiment_name: The experiment to create blocks for.
        subject: If given, only create blocks for this subject's probes.
        block_duration_min: Block length in minutes.
        overlap_min: Overlap between adjacent blocks in minutes.
        n_blocks: Number of blocks to create.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline.ephys import (
        EphysBlock,
        EphysBlockInfo,
        EphysChunk,
        ProbeInsertion,
    )

    # --- 1. Get probe insertions ---
    restriction = {"experiment_name": experiment_name}
    if subject is not None:
        restriction["subject"] = subject
    probe_insertions = (ProbeInsertion & restriction).to_dicts()

    if not probe_insertions:
        print(f"No ProbeInsertion entries found for {restriction}.")
        print("Run step 1 (register_experiment) first.")
        return

    print(f"Found {len(probe_insertions)} probe insertion(s):")
    for pi in probe_insertions:
        print(
            f"  insertion {pi['insertion_number']}: "
            f"subject={pi['subject']}, probe={pi['probe']}"
        )

    # --- 2. Find the earliest chunk start ---
    chunks = (EphysChunk & {"experiment_name": experiment_name}).to_dicts()
    if not chunks:
        print("No EphysChunk entries found. Run step 1 first.")
        return

    start_time = min(c["chunk_start"] for c in chunks)
    print(f"\nEarliest chunk start: {start_time}")

    # --- 3. Calculate block boundaries ---
    block_duration = timedelta(minutes=block_duration_min)
    overlap = timedelta(minutes=overlap_min)

    blocks_to_create = calculate_block_boundaries(
        start_time, block_duration, overlap, n_blocks
    )

    print(f"\nBlock schedule ({len(blocks_to_create)} blocks):")
    advance = block_duration - overlap
    print(f"  Duration: {block_duration_min} min, Overlap: {overlap_min} min, "
          f"Advance: {advance.total_seconds() / 60:.0f} min")
    for i, (bstart, bend) in enumerate(blocks_to_create):
        print(f"  Block {i}: {bstart} to {bend}")

    # --- 4. Insert EphysBlock entries ---
    # NOTE: Block boundaries will NOT necessarily align to chunk edges. The
    # golden dataset uses 10-minute chunks (not the typical 1-hour). A block
    # may start or end in the middle of a chunk, and that is fine --
    # EphysBlockInfo.make() handles partial chunks at block boundaries.
    insert_count = 0
    for pi in probe_insertions:
        for bstart, bend in blocks_to_create:
            block_key = {
                "experiment_name": experiment_name,
                "subject": pi["subject"],
                "insertion_number": pi["insertion_number"],
                "block_start": bstart,
                "block_end": bend,
            }
            EphysBlock.insert1(block_key, skip_duplicates=True)
            insert_count += 1

    total_in_db = len(EphysBlock & {"experiment_name": experiment_name})
    print(f"\nInserted {insert_count} EphysBlock entries "
          f"({total_in_db} total in DB for this experiment).")

    # --- 5. Populate EphysBlockInfo ---
    # EphysBlockInfo is an Imported table that computes metadata for each
    # block: which chunks overlap, what electrode configuration was used,
    # block duration, and channel-to-electrode mappings.
    print("\nPopulating EphysBlockInfo...")
    EphysBlockInfo.populate(display_progress=True, suppress_errors=False)
    print("Done.")


def verify_blocks(experiment_name):
    """Query and display all blocks and their metadata for this experiment.

    For each block, shows:
        - Block boundaries (start/end)
        - Duration (from EphysBlockInfo, in hours)
        - Number of associated chunks

    Args:
        experiment_name: The experiment to verify.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline.ephys import EphysBlock, EphysBlockInfo

    blocks = (EphysBlock & {"experiment_name": experiment_name}).to_dicts()

    if not blocks:
        print(f"No EphysBlock entries found for '{experiment_name}'.")
        return

    print(f"EphysBlock entries for '{experiment_name}': {len(blocks)}")
    print()

    for b in sorted(blocks, key=lambda x: (x["insertion_number"], x["block_start"])):
        label = (
            f"  insertion {b['insertion_number']} | "
            f"{b['block_start']} to {b['block_end']}"
        )

        # Check whether EphysBlockInfo has been populated for this block.
        info_query = EphysBlockInfo & b
        if info_query:
            duration_hrs = info_query.fetch1("block_duration")
            chunk_count = len(EphysBlockInfo.Chunk & b)
            print(
                f"{label} | "
                f"duration={duration_hrs:.2f} hrs | "
                f"{chunk_count} chunk(s)"
            )
        else:
            print(f"{label} | EphysBlockInfo not yet populated")


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 2: Define Blocks")
    print("=" * 60)

    print("\n--- 1/3: Query available data ---")
    query_available_data(EXPERIMENT_NAME)

    print("\n--- 2/3: Create blocks ---")
    create_blocks(EXPERIMENT_NAME, SUBJECT, BLOCK_DURATION_MIN, OVERLAP_MIN, N_BLOCKS)

    print("\n--- 3/3: Verify blocks ---")
    verify_blocks(EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("  Step 2 complete. Ready for spike sorting (Step 3).")
    print("=" * 60)
