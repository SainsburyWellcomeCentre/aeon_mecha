"""Repair missing SyncModel entries by deleting chunks and re-ingesting.

The original ingest_chunks created EphysChunk + File entries but failed to
create SyncModel entries. This script:
1. Deletes downstream tables that depend on chunks (SyncedSpikes, UnitMatching, GlobalUnit)
2. Deletes EphysChunk entries (cascades to File, SyncModel, EphysBlockInfo.Chunk)
3. Re-runs ingest_chunks (creates chunks WITH sync models)
4. Re-runs EphysBlockInfo.populate (reconnects blocks to chunks)

After this, re-run steps 17-19 to regenerate SyncedSpikes and UnitMatching.
Spike sorting results (steps 13-15) are preserved.
"""

import datajoint as dj
from aeon.dj_pipeline import ephys, spike_sorting

EXPERIMENT_NAME = "social-ephys0.1-aeon3"
SUBJECT = "test-subject-001"
INSERTION_NUMBER = 1

key = {
    "experiment_name": EXPERIMENT_NAME,
    "subject": SUBJECT,
    "insertion_number": INSERTION_NUMBER,
}


def main():
    print("=" * 60)
    print("  Repair: Re-ingest chunks with SyncModel entries")
    print("=" * 60)

    # Show current state
    print("\n--- Current State ---")
    print(f"  EphysChunk:        {len(ephys.EphysChunk & key)}")
    print(f"  EphysChunk.File:   {len(ephys.EphysChunk.File & key)}")
    print(f"  EphysChunk.SyncModel: {len(ephys.EphysChunk.SyncModel & key)}")
    print(f"  EphysBlockInfo:    {len(ephys.EphysBlockInfo & key)}")
    print(f"  SyncedSpikes:      {len(spike_sorting.SyncedSpikes & key)}")
    print(f"  SyncedSpikes.Unit: {len(spike_sorting.SyncedSpikes.Unit & key)}")
    print(f"  UnitMatching:      {len(spike_sorting.UnitMatching & key)}")
    print(f"  GlobalUnit:        {len(spike_sorting.GlobalUnit & key)}")

    # Step 1: Delete downstream (UnitMatching, GlobalUnit, SyncedSpikes)
    print("\n--- Step 1: Delete downstream tables ---")

    um = spike_sorting.UnitMatching & key
    if um:
        print(f"  Deleting {len(um)} UnitMatching entries (cascades to Unit, Spikes)...")
        um.delete()
        print("  Done.")

    gu = spike_sorting.GlobalUnit & key
    if gu:
        print(f"  Deleting {len(gu)} GlobalUnit entries...")
        gu.delete()
        print("  Done.")

    ss = spike_sorting.SyncedSpikes & key
    if ss:
        print(f"  Deleting {len(ss)} SyncedSpikes entries (cascades to Unit)...")
        ss.delete()
        print("  Done.")

    # Step 2: Delete EphysChunk (cascades to File, SyncModel, EphysBlockInfo.Chunk)
    print("\n--- Step 2: Delete EphysChunk entries ---")
    chunks = ephys.EphysChunk & key
    if chunks:
        print(f"  Deleting {len(chunks)} EphysChunk entries (cascades to File, SyncModel)...")
        chunks.delete()
        print("  Done.")

    # Step 3: Re-ingest chunks
    print("\n--- Step 3: Re-ingest chunks (with SyncModel) ---")
    ephys.EphysChunk.ingest_chunks(EXPERIMENT_NAME)

    new_chunks = len(ephys.EphysChunk & key)
    new_sync = len(ephys.EphysChunk.SyncModel & key)
    print(f"  EphysChunk:        {new_chunks}")
    print(f"  EphysChunk.SyncModel: {new_sync}")

    if new_sync == 0:
        print("\n  *** ERROR: Still 0 SyncModel entries! Something is wrong. ***")
        return

    # Step 4: Re-populate EphysBlockInfo
    print("\n--- Step 4: Re-populate EphysBlockInfo ---")
    ephys.EphysBlockInfo.populate(key, display_progress=True)
    print(f"  EphysBlockInfo: {len(ephys.EphysBlockInfo & key)}")

    print("\n--- Repair Complete ---")
    print(f"  SyncModel entries: {new_sync}")
    print(f"\nNext: re-run steps 17, 18, 19:")
    print(f"  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --step 17")
    print(f"  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --step 18")
    print(f"  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --step 19")
    print(f"  (run step 19 multiple times until all blocks are matched)")


if __name__ == "__main__":
    dj.config["safemode"] = False  # skip delete confirmations
    main()
