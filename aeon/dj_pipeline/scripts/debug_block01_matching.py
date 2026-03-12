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


def measure_offset(block_a_idx, block_b_idx, blocks):
    """For matched units at 50ms, measure the actual temporal offset between spikes."""
    from spikeinterface.comparison import compare_two_sorters
    from spikeinterface.core import NumpySorting

    ba = blocks[block_a_idx]
    bb = blocks[block_b_idx]
    overlap_start = max(ba["block_start"], bb["block_start"])
    overlap_end = min(ba["block_end"], bb["block_end"])
    overlap_start_s = overlap_start.timestamp()
    overlap_end_s = overlap_end.timestamp()

    key_a = {**insertion_key, "block_start": ba["block_start"], "block_end": ba["block_end"]}
    key_b = {**insertion_key, "block_start": bb["block_start"], "block_end": bb["block_end"]}

    units_a = load_block_spike_trains(key_a)
    units_b = load_block_spike_trains(key_b)

    # Get overlap spike trains in SECONDS (not samples) relative to overlap start
    trains_a_sec = {}
    for uid, times in units_a.items():
        restricted = restrict_to_overlap(times, overlap_start_s, overlap_end_s)
        if len(restricted) > 0:
            trains_a_sec[uid] = restricted

    trains_b_sec = {}
    for uid, times in units_b.items():
        restricted = restrict_to_overlap(times, overlap_start_s, overlap_end_s)
        if len(restricted) > 0:
            trains_b_sec[uid] = restricted

    # Get matches at 50ms
    trains_a_samp = {uid: (t * 30000).astype(np.int64) for uid, t in trains_a_sec.items()}
    trains_b_samp = {uid: (t * 30000).astype(np.int64) for uid, t in trains_b_sec.items()}

    sorting_a = NumpySorting.from_unit_dict(trains_a_samp, sampling_frequency=30000)
    sorting_b = NumpySorting.from_unit_dict(trains_b_samp, sampling_frequency=30000)
    comparison = compare_two_sorters(
        sorting1=sorting_a, sorting2=sorting_b,
        sorting1_name="block_a", sorting2_name="block_b",
        delta_time=50.0,
    )
    a_to_b, _ = comparison.get_matching()

    print(f"\n{'='*60}")
    print(f"  Measuring temporal offset: Block {block_a_idx} vs Block {block_b_idx}")
    print(f"{'='*60}")

    offsets = []
    for a_uid, b_uid in a_to_b.items():
        if b_uid == -1:
            continue
        times_a = trains_a_sec[a_uid]
        times_b = trains_b_sec[b_uid]

        # For each spike in A, find nearest spike in B
        nearest_offsets = []
        # Sample 1000 spikes evenly spaced through the overlap
        step = max(1, len(times_a) // 1000)
        for t_a in times_a[::step]:
            idx = np.searchsorted(times_b, t_a)
            candidates = []
            if idx > 0:
                candidates.append(times_b[idx - 1] - t_a)
            if idx < len(times_b):
                candidates.append(times_b[idx] - t_a)
            if candidates:
                nearest = min(candidates, key=abs)
                if abs(nearest) < 0.1:  # within 100ms
                    nearest_offsets.append(nearest)

        if nearest_offsets:
            arr = np.array(nearest_offsets)
            median_off = np.median(arr) * 1000  # convert to ms
            offsets.append(median_off)
            print(f"  Unit {a_uid}→{b_uid}: median offset = {median_off:.3f} ms "
                  f"(from {len(nearest_offsets)} matched spikes)")

    if offsets:
        arr = np.array(offsets)
        print(f"\n  Overall: median offset = {np.median(arr):.3f} ms, "
              f"mean = {np.mean(arr):.3f} ms, std = {np.std(arr):.3f} ms")


def compare_shared_chunk(block_a_idx, block_b_idx, blocks):
    """Compare spike timestamps for the SAME chunk across two blocks.

    If the same chunk's spike times differ between blocks, it's a sync issue.
    If they're the same, it's a sorting (spike detection) issue.
    """
    ba = blocks[block_a_idx]
    bb = blocks[block_b_idx]

    key_a = {**insertion_key, "block_start": ba["block_start"], "block_end": ba["block_end"]}
    key_b = {**insertion_key, "block_start": bb["block_start"], "block_end": bb["block_end"]}

    print(f"\n{'='*60}")
    print(f"  Chunk-level comparison: Block {block_a_idx} vs Block {block_b_idx}")
    print(f"{'='*60}")

    # Get chunk_starts for each block from SyncedSpikes.Unit
    chunks_a = set((spike_sorting.SyncedSpikes.Unit & key_a).fetch("chunk_start"))
    chunks_b = set((spike_sorting.SyncedSpikes.Unit & key_b).fetch("chunk_start"))
    shared_chunks = sorted(chunks_a & chunks_b)

    print(f"  Block {block_a_idx} chunks: {sorted(chunks_a)}")
    print(f"  Block {block_b_idx} chunks: {sorted(chunks_b)}")
    print(f"  Shared chunks: {shared_chunks}")

    if not shared_chunks:
        print("  NO SHARED CHUNKS — cannot compare at chunk level")
        return

    # For each shared chunk, compare spike counts and times per unit
    for chunk_start in shared_chunks:
        print(f"\n  --- Chunk {chunk_start} ---")

        # Fetch per-unit spike_times from each block for this chunk
        entries_a = (spike_sorting.SyncedSpikes.Unit & key_a & {"chunk_start": chunk_start}).fetch(
            "unit", "spike_times", "spike_count", as_dict=True
        )
        entries_b = (spike_sorting.SyncedSpikes.Unit & key_b & {"chunk_start": chunk_start}).fetch(
            "unit", "spike_times", "spike_count", as_dict=True
        )
        print(f"    Block {block_a_idx}: {len(entries_a)} unit entries, "
              f"{sum(e['spike_count'] for e in entries_a)} total spikes")
        print(f"    Block {block_b_idx}: {len(entries_b)} unit entries, "
              f"{sum(e['spike_count'] for e in entries_b)} total spikes")

        # Build per-unit spike arrays (in epoch seconds)
        def to_epoch_sec(times):
            if len(times) == 0:
                return np.array([])
            if times.dtype.kind == "M":
                return times.astype("datetime64[ns]").astype(np.int64) / 1e9
            return times

        units_a = {e["unit"]: to_epoch_sec(e["spike_times"]) for e in entries_a}
        units_b = {e["unit"]: to_epoch_sec(e["spike_times"]) for e in entries_b}

        # Concatenate ALL spikes in this chunk (regardless of unit) and compare
        all_a = np.sort(np.concatenate([t for t in units_a.values() if len(t) > 0])) if units_a else np.array([])
        all_b = np.sort(np.concatenate([t for t in units_b.values() if len(t) > 0])) if units_b else np.array([])

        if len(all_a) > 0 and len(all_b) > 0:
            print(f"    Block {block_a_idx} time range: {all_a[0]:.6f} → {all_a[-1]:.6f}")
            print(f"    Block {block_b_idx} time range: {all_b[0]:.6f} → {all_b[-1]:.6f}")

            # Find nearest-neighbor distances between ALL spikes (pooled across units)
            # Sample 2000 spikes from A
            step = max(1, len(all_a) // 2000)
            nn_dists = []
            for t in all_a[::step]:
                idx = np.searchsorted(all_b, t)
                dists = []
                if idx > 0:
                    dists.append(abs(all_b[idx - 1] - t))
                if idx < len(all_b):
                    dists.append(abs(all_b[idx] - t))
                if dists:
                    nn_dists.append(min(dists))

            nn_arr = np.array(nn_dists) * 1000  # ms
            pct_below_04 = np.mean(nn_arr < 0.4) * 100
            pct_below_1 = np.mean(nn_arr < 1.0) * 100
            pct_below_10 = np.mean(nn_arr < 10.0) * 100
            print(f"    Nearest-neighbor distances (pooled, all units):")
            print(f"      median: {np.median(nn_arr):.3f} ms")
            print(f"      <0.4ms: {pct_below_04:.1f}%")
            print(f"      <1.0ms: {pct_below_1:.1f}%")
            print(f"      <10ms:  {pct_below_10:.1f}%")
            print(f"      p25/p75: {np.percentile(nn_arr, 25):.3f} / {np.percentile(nn_arr, 75):.3f} ms")


def main():
    # Fetch blocks
    blocks = (ephys.EphysBlock & insertion_key).fetch(
        "block_start", "block_end", as_dict=True, order_by="block_start"
    )
    print(f"Found {len(blocks)} blocks")
    for i, b in enumerate(blocks):
        print(f"  Block {i}: {b['block_start']} → {b['block_end']}")

    # Skip the full comparison (already done), go straight to chunk analysis
    compare_shared_chunk(0, 1, blocks)
    compare_shared_chunk(1, 2, blocks)


if __name__ == "__main__":
    main()
