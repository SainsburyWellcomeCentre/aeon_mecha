"""Diagnostic script to debug unit matching failures.

Checks spike time ranges, overlap windows, and whether spikes
actually exist in the overlap region for consecutive blocks.
"""

import numpy as np
import datajoint as dj

from aeon.dj_pipeline import ephys, spike_sorting

EXPERIMENT_NAME = "social-ephys0.1-aeon3"
SUBJECT = "test-subject-001"
INSERTION_NUMBER = 1

insertion_key = {
    "experiment_name": EXPERIMENT_NAME,
    "subject": SUBJECT,
    "insertion_number": INSERTION_NUMBER,
}


def main():
    print("=" * 60)
    print("  Unit Matching Diagnostic")
    print("=" * 60)

    # 1. Fetch all blocks
    blocks = (ephys.EphysBlock & insertion_key).fetch(
        "block_start", "block_end", as_dict=True, order_by="block_start"
    )
    print(f"\n--- {len(blocks)} Blocks ---")
    for i, b in enumerate(blocks):
        print(f"  Block {i}: {b['block_start']} → {b['block_end']}")

    # 2. For each block, check SyncedSpikes.Unit spike time ranges
    print(f"\n--- Spike Time Ranges per Block (from SyncedSpikes.Unit) ---")
    block_spike_data = {}
    for i, b in enumerate(blocks):
        block_key = {**insertion_key, "block_start": b["block_start"], "block_end": b["block_end"]}
        unit_data = (spike_sorting.SyncedSpikes.Unit & block_key).fetch(
            "unit", "chunk_start", "spike_times", "spike_count", as_dict=True
        )
        if not unit_data:
            print(f"  Block {i}: NO SyncedSpikes.Unit data!")
            continue

        # Collect all spike times for this block
        all_times = []
        units_seen = set()
        for entry in unit_data:
            units_seen.add(entry["unit"])
            if len(entry["spike_times"]) > 0:
                all_times.append(entry["spike_times"])

        if all_times:
            concatenated = np.concatenate(all_times)
            print(f"  Block {i}: {len(units_seen)} units, {len(concatenated)} total spikes")
            print(f"    dtype: {concatenated.dtype}")
            print(f"    min:   {concatenated.min()}")
            print(f"    max:   {concatenated.max()}")

            # Convert to epoch seconds for comparison
            if concatenated.dtype.kind == "M":
                epoch_s = concatenated.astype("datetime64[ns]").astype(np.int64) / 1e9
            else:
                epoch_s = concatenated
            print(f"    epoch_s range: {epoch_s.min():.3f} → {epoch_s.max():.3f}")

            block_spike_data[i] = {
                "units_seen": units_seen,
                "unit_data": unit_data,
                "epoch_s_min": epoch_s.min(),
                "epoch_s_max": epoch_s.max(),
                "dtype": concatenated.dtype,
            }
        else:
            print(f"  Block {i}: {len(units_seen)} units but ALL spike arrays empty!")

    # 3. Check overlap windows between consecutive blocks
    print(f"\n--- Overlap Analysis ---")
    for i in range(len(blocks) - 1):
        b1 = blocks[i]
        b2 = blocks[i + 1]

        overlap_start = max(b1["block_start"], b2["block_start"])
        overlap_end = min(b1["block_end"], b2["block_end"])

        if overlap_start >= overlap_end:
            print(f"  Blocks {i}-{i+1}: NO OVERLAP")
            continue

        print(f"\n  Blocks {i} & {i+1} overlap: {overlap_start} → {overlap_end}")

        # Convert overlap to epoch seconds (same way as UnitMatching.make)
        overlap_start_s = overlap_start.timestamp()
        overlap_end_s = overlap_end.timestamp()
        print(f"    overlap_start.timestamp() = {overlap_start_s:.3f}")
        print(f"    overlap_end.timestamp()   = {overlap_end_s:.3f}")
        print(f"    overlap duration = {overlap_end_s - overlap_start_s:.1f} seconds")

        # Check block_start/end as timestamp too
        print(f"    block {i} start.timestamp() = {b1['block_start'].timestamp():.3f}")
        print(f"    block {i} end.timestamp()   = {b1['block_end'].timestamp():.3f}")
        print(f"    block {i+1} start.timestamp() = {b2['block_start'].timestamp():.3f}")
        print(f"    block {i+1} end.timestamp()   = {b2['block_end'].timestamp():.3f}")

        if i in block_spike_data:
            sd = block_spike_data[i]
            print(f"    block {i} spike epoch_s: {sd['epoch_s_min']:.3f} → {sd['epoch_s_max']:.3f}")
            print(f"    overlap_start_s in spike range? {sd['epoch_s_min'] <= overlap_start_s <= sd['epoch_s_max']}")
            print(f"    overlap_end_s in spike range?   {sd['epoch_s_min'] <= overlap_end_s <= sd['epoch_s_max']}")

        if (i + 1) in block_spike_data:
            sd = block_spike_data[i + 1]
            print(f"    block {i+1} spike epoch_s: {sd['epoch_s_min']:.3f} → {sd['epoch_s_max']:.3f}")
            print(f"    overlap_start_s in spike range? {sd['epoch_s_min'] <= overlap_start_s <= sd['epoch_s_max']}")

        # Count spikes in overlap window for each block
        for bi in [i, i + 1]:
            if bi not in block_spike_data:
                continue
            block_key = {
                **insertion_key,
                "block_start": blocks[bi]["block_start"],
                "block_end": blocks[bi]["block_end"],
            }
            unit_data = (spike_sorting.SyncedSpikes.Unit & block_key).fetch(
                "unit", "spike_times", as_dict=True
            )
            units_with_spikes_in_overlap = 0
            total_spikes_in_overlap = 0
            for entry in unit_data:
                times = entry["spike_times"]
                if len(times) == 0:
                    continue
                if times.dtype.kind == "M":
                    times_s = times.astype("datetime64[ns]").astype(np.int64) / 1e9
                else:
                    times_s = times
                mask = (times_s >= overlap_start_s) & (times_s <= overlap_end_s)
                n_in = mask.sum()
                if n_in > 0:
                    units_with_spikes_in_overlap += 1
                    total_spikes_in_overlap += n_in
            print(f"    block {bi}: {units_with_spikes_in_overlap} units with spikes in overlap, "
                  f"{total_spikes_in_overlap} total spikes in overlap")

    # 4. Check UnitMatching results
    print(f"\n--- UnitMatching Results ---")
    matching_entries = (spike_sorting.UnitMatching & insertion_key).fetch(as_dict=True, order_by="block_start")
    for entry in matching_entries:
        block_start = entry["block_start"]
        units = (spike_sorting.UnitMatching.Unit & entry).fetch(as_dict=True)
        global_units = set(u["global_unit"] for u in units)
        print(f"  Block {block_start}: {len(units)} units → {len(global_units)} unique global units")

    # 5. Check GlobalUnit distribution
    all_gus = (spike_sorting.GlobalUnit & insertion_key).fetch("global_unit")
    print(f"\n  Total GlobalUnit entries: {len(all_gus)}")
    print(f"  Unique global_unit IDs: {len(set(all_gus))}")

    # 6. Quick timestamp sanity check
    print(f"\n--- Timestamp Sanity Check ---")
    import datetime as dt
    test_dt = dt.datetime(2024, 6, 4, 13, 0, 0)  # overlap start
    print(f"  Python datetime(2024,6,4,13,0,0).timestamp() = {test_dt.timestamp():.3f}")
    test_np = np.datetime64("2024-06-04T13:00:00", "ns")
    print(f"  numpy datetime64('2024-06-04T13:00:00').astype(int64)/1e9 = {test_np.astype(np.int64)/1e9:.3f}")
    print(f"  Difference = {test_dt.timestamp() - test_np.astype(np.int64)/1e9:.3f} seconds")
    if abs(test_dt.timestamp() - test_np.astype(np.int64)/1e9) > 1:
        print(f"  *** TIMEZONE MISMATCH DETECTED! ***")
        print(f"  Python .timestamp() interprets naive datetime as local time.")
        print(f"  numpy datetime64 is always UTC.")
        import time
        print(f"  Local timezone offset: {-time.timezone} seconds from UTC")


if __name__ == "__main__":
    main()
