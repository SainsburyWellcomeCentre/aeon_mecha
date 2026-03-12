"""Detailed diagnostic: replicate UnitMatching.make() logic for blocks 0 and 1.

Traces every step of the matching process to find where block 0↔1 matching fails,
while blocks 1↔2 and 2↔3 succeed.
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


def load_block_spike_trains(block_key):
    """Load spike trains exactly as UnitMatching.make() does.

    Returns dict: unit_id -> epoch_seconds_array
    """
    units = {}
    for entry in (spike_sorting.SyncedSpikes.Unit & block_key).fetch(as_dict=True):
        uid = entry["unit"]
        if uid not in units:
            units[uid] = []
        units[uid].append(entry["spike_times"])

    for uid in units:
        concatenated = np.sort(np.concatenate(units[uid]))
        if concatenated.dtype.kind == "M":
            concatenated = concatenated.astype("datetime64[ns]").astype(np.int64) / 1e9
        units[uid] = concatenated

    return units


def restrict_to_overlap(spike_times_s, start_s, end_s):
    """Same as UnitMatching._restrict_to_overlap."""
    if len(spike_times_s) == 0:
        return np.array([])
    mask = (spike_times_s >= start_s) & (spike_times_s <= end_s)
    return spike_times_s[mask] - start_s


def compare_blocks(block_a_idx, block_b_idx, blocks, delta_times=None):
    """Run comparison between two blocks at various delta_time values."""
    from spikeinterface.comparison import compare_two_sorters
    from spikeinterface.core import NumpySorting

    if delta_times is None:
        delta_times = [0.4, 1.0, 2.0, 5.0, 10.0, 50.0]

    ba = blocks[block_a_idx]
    bb = blocks[block_b_idx]

    print(f"\n{'='*60}")
    print(f"  Comparing Block {block_a_idx} vs Block {block_b_idx}")
    print(f"{'='*60}")
    print(f"  Block {block_a_idx}: {ba['block_start']} → {ba['block_end']}")
    print(f"  Block {block_b_idx}: {bb['block_start']} → {bb['block_end']}")

    overlap_start = max(ba["block_start"], bb["block_start"])
    overlap_end = min(ba["block_end"], bb["block_end"])
    if overlap_start >= overlap_end:
        print("  NO OVERLAP!")
        return

    overlap_start_s = overlap_start.timestamp()
    overlap_end_s = overlap_end.timestamp()
    print(f"  Overlap: {overlap_start} → {overlap_end}")
    print(f"  Overlap epoch_s: {overlap_start_s:.3f} → {overlap_end_s:.3f}")
    print(f"  Overlap duration: {overlap_end_s - overlap_start_s:.0f} seconds")

    # Load spike trains
    key_a = {
        **insertion_key,
        "block_start": ba["block_start"],
        "block_end": ba["block_end"],
    }
    key_b = {
        **insertion_key,
        "block_start": bb["block_start"],
        "block_end": bb["block_end"],
    }

    print(f"\n  Loading spike trains for block {block_a_idx}...")
    units_a = load_block_spike_trains(key_a)
    print(f"  → {len(units_a)} units loaded")

    print(f"  Loading spike trains for block {block_b_idx}...")
    units_b = load_block_spike_trains(key_b)
    print(f"  → {len(units_b)} units loaded")

    # Show a few units' time ranges
    for label, units_dict, bidx in [("A", units_a, block_a_idx), ("B", units_b, block_b_idx)]:
        print(f"\n  Block {bidx} ({label}) sample unit time ranges (epoch_s):")
        for i, (uid, times) in enumerate(sorted(units_dict.items())):
            if i < 3 or i == len(units_dict) - 1:
                print(f"    unit {uid}: {times.min():.3f} → {times.max():.3f} ({len(times)} spikes)")
            elif i == 3:
                print(f"    ... ({len(units_dict) - 4} more units) ...")

    # Restrict to overlap
    print(f"\n  Restricting to overlap window [{overlap_start_s:.3f}, {overlap_end_s:.3f}]...")

    trains_a = {}
    for uid, times in units_a.items():
        restricted = restrict_to_overlap(times, overlap_start_s, overlap_end_s)
        if len(restricted) > 0:
            trains_a[uid] = (restricted * 30000).astype(np.int64)

    trains_b = {}
    for uid, times in units_b.items():
        restricted = restrict_to_overlap(times, overlap_start_s, overlap_end_s)
        if len(restricted) > 0:
            trains_b[uid] = (restricted * 30000).astype(np.int64)

    print(f"  Block {block_a_idx}: {len(trains_a)}/{len(units_a)} units have spikes in overlap")
    print(f"  Block {block_b_idx}: {len(trains_b)}/{len(units_b)} units have spikes in overlap")

    if not trains_a or not trains_b:
        print("  *** NO SPIKE TRAINS IN OVERLAP — cannot compare ***")
        return

    # Show sample counts in overlap
    for label, trains, bidx in [("A", trains_a, block_a_idx), ("B", trains_b, block_b_idx)]:
        total = sum(len(t) for t in trains.values())
        print(f"  Block {bidx} overlap: {total} total spikes across {len(trains)} units")
        for i, (uid, t) in enumerate(sorted(trains.items())):
            if i < 3:
                print(f"    unit {uid}: {len(t)} spikes, samples {t[0]}..{t[-1]}")

    # Show max sample value — should be ~overlap_duration * 30000
    expected_max = (overlap_end_s - overlap_start_s) * 30000
    for label, trains, bidx in [("A", trains_a, block_a_idx), ("B", trains_b, block_b_idx)]:
        max_sample = max(t[-1] for t in trains.values())
        print(f"  Block {bidx} max sample: {max_sample} (expected ~{expected_max:.0f})")

    # Run comparison at various delta_time values
    for dt_ms in delta_times:
        print(f"\n  --- compare_two_sorters(delta_time={dt_ms} ms) ---")
        sorting_a = NumpySorting.from_unit_dict(trains_a, sampling_frequency=30000)
        sorting_b = NumpySorting.from_unit_dict(trains_b, sampling_frequency=30000)
        comparison = compare_two_sorters(
            sorting1=sorting_a,
            sorting2=sorting_b,
            sorting1_name="block_a",
            sorting2_name="block_b",
            delta_time=dt_ms,
        )

        a_to_b, b_to_a = comparison.get_matching()
        n_matched_ab = sum(1 for v in a_to_b if v != -1)
        n_matched_ba = sum(1 for v in b_to_a if v != -1)
        print(f"    A→B matches: {n_matched_ab}/{len(a_to_b)}")
        print(f"    B→A matches: {n_matched_ba}/{len(b_to_a)}")

        if n_matched_ab > 0:
            print(f"    Matched pairs (A unit → B unit):")
            for a_uid, b_uid in a_to_b.items():
                if b_uid != -1:
                    print(f"      {a_uid} → {b_uid}")

        # Also print agreement matrix summary
        try:
            agreement = comparison.get_agreement_fraction()
            max_agree = agreement.values.max() if agreement.size > 0 else 0
            mean_agree = agreement.values[agreement.values > 0].mean() if (agreement.values > 0).any() else 0
            print(f"    Agreement fraction: max={max_agree:.4f}, mean(>0)={mean_agree:.4f}")
            if max_agree > 0 and max_agree < 0.5:
                print(f"    *** Low agreement — units may exist in overlap but spikes don't align ***")
        except Exception as e:
            print(f"    (Could not get agreement fraction: {e})")


def main():
    # Fetch blocks
    blocks = (ephys.EphysBlock & insertion_key).fetch(
        "block_start", "block_end", as_dict=True, order_by="block_start"
    )
    print(f"Found {len(blocks)} blocks")
    for i, b in enumerate(blocks):
        print(f"  Block {i}: {b['block_start']} → {b['block_end']}")

    # Compare block 0 vs 1 (the failing pair)
    compare_blocks(0, 1, blocks)

    # Compare block 1 vs 2 (a working pair) for reference
    compare_blocks(1, 2, blocks)


if __name__ == "__main__":
    main()
