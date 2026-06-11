# Ephys Compression Test Results

Branch: `es/compression-spec`
Date: 2026-06-11
Test data: `abcGolden01`, subject `IAA-1147881`, insertion 1, electrode group `shank0`

---

## Workstream 1: Raw Data Compression

Compresses each raw ephys chunk to zarr, decompresses, and verifies the result matches the original byte-for-byte.

**Codec:** Blosc-zstd, compression level 5, bitshuffle (SpikeInterface default)

### Results

| Metric | Value |
|--------|-------|
| Chunks tested | 12 |
| Checksum pass | 12 (100%) |
| Mean compression ratio | 1.95x |
| Min / Max ratio | 1.95x / 1.98x |
| Total original size | 160.75 GB |
| Total compressed size | 82.32 GB |
| Space saved | 78.43 GB (48.8%) |

### Timing per chunk (~10 min recording, 96 channels)

| Operation | Mean | Min | Max |
|-----------|------|-----|-----|
| Compression | 152.4s | 75.9s | 192.3s |
| Decompression | 36.1s | 12.4s | 182.3s |

### Scaling rates (per shank)

| Metric | Per hour of recording |
|--------|----------------------|
| Raw data size | ~80 GB/h |
| Compressed size | ~41 GB/h |
| Space saved | ~39 GB/h |
| Compression time | ~15 min/h |
| Decompression time | ~4 min/h |

Multiply by number of shanks and experiment duration to estimate totals.

---

## Workstream 2: Compressed Pipeline Intermediates

Runs the full spike sorting pipeline using zarr-compressed intermediates instead of uncompressed binary, and compares execution time and storage.

Both paramsets use identical sorting parameters (Kilosort 4). The only difference is the storage format for `recording.dat`/`recording.zarr` and `sorting_analyzer`/`sorting_analyzer.zarr`.

### Pipeline execution time

| Stage | Binary (n=12, mean) | Zarr (n=1) | Difference |
|-------|---------------------|------------|------------|
| PreProcessing | 10.5 min | 4.1 min | zarr 2.6x faster |
| SpikeSorting | 6.2 min | 7.8 min | zarr 26% slower |
| PostProcessing | 29.5 min | 25.0 min | zarr 15% faster |
| **Total** | **46.2 min** | **36.9 min** | **zarr 20% faster overall** |

SpikeSorting is slightly slower with zarr because Kilosort4 requires binary input — SpikeInterface converts zarr to a temporary binary internally. The zarr time (7.8 min) is within the range of binary runs (min 4.1 min, max 8.3 min).

### Storage per block (30-minute recording, 96 channels)

| File | Binary | Zarr | Reduction |
|------|--------|------|-----------|
| Recording | 12.67 GB | 5.03 GB | 60% |
| Sorting analyzer | 0.50 GB | 0.30 GB | 40% |
| **Total** | **13.17 GB** | **5.33 GB** | **60%** |

### Scaling rates (per shank)

| File | Binary per hour | Zarr per hour | Saved per hour |
|------|----------------|---------------|----------------|
| Recording | ~25 GB/h | ~10 GB/h | ~15 GB/h |
| Analyzer | ~1 GB/h | ~0.6 GB/h | ~0.4 GB/h |

Multiply by number of shanks and experiment duration to estimate totals.

---

## Summary

| | Compression ratio | Speed impact | Lossless verified |
|--|-------------------|-------------|-------------------|
| Raw data (Tier 2) | 1.95x (49% savings) | N/A — offline process | Yes, all 12 chunks |
| Intermediates (Tier 1) | 2.5x recording, 1.7x analyzer | 20% faster overall | Same sorting results |

**Tier 1 (intermediates):** Ready to adopt. Zarr intermediates save 60% disk space with no speed penalty — in fact the pipeline runs faster overall because compressed I/O reduces Ceph bandwidth. The only code changes are in `PreProcessing` and `PostProcessing` to save in zarr format instead of binary.

**Tier 2 (raw data):** The 1.95x compression ratio and verified losslessness make this viable for archival. Compression takes about 15 min per hour of recording per shank — slow but a one-time offline operation. Decompression is ~4 min per hour of recording, fast enough for on-demand access.

---

## Implementation details

All code is on the `es/compression-spec` branch. Key files:

- `aeon/dj_pipeline/ephys.py` — `CompressionTest` table
- `aeon/dj_pipeline/spike_sorting.py` — zarr format branching in `PreProcessing`, `SpikeSorting`, `PostProcessing`
- `docs/specs/SPEC_EPHYS_COMPRESSION.md` — full design specification
- `scripts/compression_report.py` — script that generated these numbers
