"""Compression testing report.

Queries CompressionTest (Workstream 1) and pipeline execution tables (Workstream 2)
to print a summary of compression testing results.

Usage:
    python scripts/compression_report.py
"""

import datajoint as dj
import numpy as np

from aeon.dj_pipeline import ephys, get_schema_name
from aeon.dj_pipeline.spike_sorting import (
    PreProcessing,
    SpikeSorting,
    PostProcessing,
    SortingParamSet,
)
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir


def report_workstream1():
    """Raw data compression test results from CompressionTest table."""
    print("=" * 70)
    print("WORKSTREAM 1: Raw Data Compression Test (CompressionTest)")
    print("=" * 70)

    ct = ephys.CompressionTest()
    n_total = len(ct)
    if n_total == 0:
        print("No CompressionTest entries found.\n")
        return

    data = ct.to_pandas()

    n_pass = int(data["checksum_match"].sum())
    n_fail = n_total - n_pass

    print(f"\nChunks tested:  {n_total}")
    print(f"Checksum pass:  {n_pass}")
    print(f"Checksum FAIL:  {n_fail}")

    if n_fail > 0:
        print("\n--- FAILURES ---")
        failures = data[~data["checksum_match"]]
        for _, row in failures.iterrows():
            print(f"  {row['subject']} ins{row['insertion_number']} "
                  f"chunk {row['chunk_start']}: {row['mismatch_details']}")

    ratios = data["compression_ratio"]
    print(f"\nCompression ratio:")
    print(f"  Mean:  {ratios.mean():.3f}x")
    print(f"  Min:   {ratios.min():.3f}x")
    print(f"  Max:   {ratios.max():.3f}x")

    print(f"\nCompression time (seconds per chunk):")
    print(f"  Mean:  {data['compression_time_s'].mean():.1f}")
    print(f"  Min:   {data['compression_time_s'].min():.1f}")
    print(f"  Max:   {data['compression_time_s'].max():.1f}")

    print(f"\nDecompression time (seconds per chunk):")
    print(f"  Mean:  {data['decompression_time_s'].mean():.1f}")
    print(f"  Min:   {data['decompression_time_s'].min():.1f}")
    print(f"  Max:   {data['decompression_time_s'].max():.1f}")

    total_orig = data["original_size_bytes"].sum()
    total_comp = data["compressed_size_bytes"].sum()
    print(f"\nTotal sizes:")
    print(f"  Original:    {total_orig / 1e9:.2f} GB")
    print(f"  Compressed:  {total_comp / 1e9:.2f} GB")
    print(f"  Saved:       {(total_orig - total_comp) / 1e9:.2f} GB ({100 * (1 - total_comp / total_orig):.1f}%)")

    print(f"\nCodec: {data['codec_name'].iloc[0]}")
    print()


def report_workstream2():
    """Compressed intermediates + sorting speed comparison from pipeline tables."""
    print("=" * 70)
    print("WORKSTREAM 2: Compressed Intermediates Speed Comparison")
    print("=" * 70)

    # Find paramsets with save_format
    all_paramsets = SortingParamSet.to_dicts()
    binary_ids = []
    zarr_ids = []
    for ps in all_paramsets:
        fmt = ps["params"].get("save_format", "binary")
        if fmt == "zarr":
            zarr_ids.append(ps["paramset_id"])
        else:
            binary_ids.append(ps["paramset_id"])

    if not zarr_ids:
        print("\nNo zarr-format SortingParamSet entries found.")
        print("Insert a paramset with save_format='zarr' to run Workstream 2.\n")
        return

    print(f"\nBinary paramset IDs: {binary_ids}")
    print(f"Zarr paramset IDs:   {zarr_ids}")

    # Gather execution times for each stage
    sorting_root = get_sorting_root_dir()
    stages = [
        ("PreProcessing", PreProcessing),
        ("SpikeSorting", SpikeSorting),
        ("PostProcessing", PostProcessing),
    ]

    for stage_name, table in stages:
        print(f"\n--- {stage_name} execution_duration (hours) ---")
        for label, ids in [("binary", binary_ids), ("zarr", zarr_ids)]:
            restriction = [{"paramset_id": pid} for pid in ids]
            entries = (table & restriction)
            if not entries:
                print(f"  {label}: no entries")
                continue
            durations = entries.to_pandas()["execution_duration"]
            print(f"  {label}: n={len(durations)}, "
                  f"mean={durations.mean():.4f}h, "
                  f"min={durations.min():.4f}h, "
                  f"max={durations.max():.4f}h")

    # Compare output sizes on disk
    print(f"\n--- Output sizes on disk ---")
    for label, ids in [("binary", binary_ids), ("zarr", zarr_ids)]:
        restriction = [{"paramset_id": pid} for pid in ids]
        entries = (PreProcessing & restriction)
        if not entries:
            print(f"  {label}: no entries")
            continue

        recording_sizes = []
        analyzer_sizes = []
        for entry in entries.to_dicts():
            output_dir = sorting_root / entry["sorting_output_dir"]
            rec_dir = output_dir.parent / "recording"
            ana_dir = output_dir / "sorting_analyzer"
            if rec_dir.exists():
                size = sum(f.stat().st_size for f in rec_dir.rglob("*") if f.is_file())
                recording_sizes.append(size)
            if ana_dir.exists():
                size = sum(f.stat().st_size for f in ana_dir.rglob("*") if f.is_file())
                analyzer_sizes.append(size)

        if recording_sizes:
            avg = np.mean(recording_sizes)
            print(f"  {label} recording dir: n={len(recording_sizes)}, avg={avg / 1e9:.2f} GB")
        if analyzer_sizes:
            avg = np.mean(analyzer_sizes)
            print(f"  {label} analyzer dir:  n={len(analyzer_sizes)}, avg={avg / 1e9:.2f} GB")

    print()


if __name__ == "__main__":
    report_workstream1()
    report_workstream2()
