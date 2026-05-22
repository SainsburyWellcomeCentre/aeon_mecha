# Ephys Data Compression Specification

## Table of Contents
1. [Problem Statement & Scope](#problem-statement--scope)
2. [Background: How SpikeInterface Compression Works](#background-how-spikeinterface-compression-works)
3. [Why Not Adopt the Allen Brain Institute's Approach Directly](#why-not-adopt-the-allen-brain-institutes-approach-directly)
4. [Tier 1 Recommendation: Compress Intermediate Files](#tier-1-recommendation-compress-intermediate-files)
5. [Tier 2 Evaluation: Compressing Raw Acquisition Files](#tier-2-evaluation-compressing-raw-acquisition-files)
6. [Implementation Considerations](#implementation-considerations)
7. [Verification & Rollout Plan](#verification--rollout-plan)
8. [Summary & Recommendations](#summary--recommendations)

---

## Problem Statement & Scope

### Problem

Project Aeon's ephys pipeline generates massive amounts of data that is exhausting disk space on Ceph. A single experiment (e.g., abcGolden01: 131 hours, Neuropixels 2.0, 384 channels at 30 kHz) produces:

| Data tier | Per 10-min chunk | Per experiment (est.) |
|-----------|------------------|-----------------------|
| Raw acquisition (`*_AmplifierData*.bin`) | ~13.8 GB | ~10.8 TB |
| Preprocessed recording (`recording.dat`) | ~10.4 GB | ~8.1 TB |
| Sorting output (Kilosort) | 1-5 GB | ~1-4 TB |
| Sorting analyzer (extensions) | 2-10 GB | ~2-8 TB |

The preprocessed `recording.dat` is essentially a filtered copy of the raw data (the size difference between raw and preprocessed reflects that preprocessing may select a subset of electrode groups and that raw ONIX files include additional overhead), so the pipeline roughly doubles disk usage before spike sorting even begins. The sorting analyzer and Kilosort outputs add further on top. None of these files are currently compressed.

### Scope

This spec evaluates compression options at two tiers:

1. **Tier 1 (Intermediate files):** `recording.dat` and `sorting_analyzer/` -- the large outputs of PreProcessing and PostProcessing. **Recommended for immediate implementation.**
2. **Tier 2 (Raw acquisition files):** The original `*_AmplifierData*.bin` files on Ceph. **Evaluated here for future consideration**, with risk analysis.

---

## Background: How SpikeInterface Compression Works

SpikeInterface supports compression exclusively through the **zarr** format. The current Aeon pipeline uses `binary_folder` format everywhere, which has no compression capability.

### The mechanism

Zarr is a chunked array storage format that applies a compressor to each chunk independently. SpikeInterface wraps zarr with a default compressor tuned for neural data:

```python
# SpikeInterface's built-in default (from zarrextractors.py)
Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
```

This uses Zstandard compression with bit-level shuffling -- a lossless rearrangement that groups corresponding bits across successive samples together before compression. This makes the data much more compressible because many of those bit positions are identical across samples.

### What can be compressed

| Component | Current format | Zarr-compressed format |
|-----------|---------------|----------------------|
| Preprocessed recording | `write_binary_recording()` -> `recording.dat` | `recording.save(format="zarr")` -> `recording.zarr/` |
| Sorting analyzer | `create_sorting_analyzer(format="binary_folder")` | `create_sorting_analyzer(format="zarr")` |

Any **numcodecs-compatible codec** can be substituted for the default. The two most relevant options for Neuropixels data:

1. **Blosc-zstd** (SpikeInterface default) -- built into numcodecs, no extra dependencies, extremely fast (multi-threaded C library with SIMD optimization).
2. **WavPack** (Allen Neural Dynamics' choice) -- audio codec adapted for neural data via the `wavpack-numcodecs` package, better compression ratios but requires extra installation.

### Published compression ratios for Neuropixels 2.0

From Buccino et al. 2023, "Compression strategies for large-scale electrophysiology data" (J. Neural Eng.):

| Codec | NP2 compression ratio | Disk reduction | Notes |
|-------|----------------------|----------------|-------|
| Blosc-zstd (level 5, BITSHUFFLE) | ~1.76x | ~43% | Built-in, fastest |
| WavPack lossless (level 3) | ~2.26x | ~56% | Requires `wavpack-numcodecs` |
| WavPack lossy hybrid (2.25 bps) | ~7.04x | ~86% | Lossy -- not recommended for archival |

Note: Neuropixels 1.0 probes compress significantly better (~3.6x with WavPack lossless). Aeon uses NP2 exclusively, so the NP2 numbers are what apply here. These ratios are averages from the published benchmarks and may vary depending on signal characteristics. Step 3 of the rollout plan measures the actual ratio on Aeon's data.

### LSB (Least Significant Bit) correction

Allen Neural Dynamics (AIND) applies a preprocessing step called LSB correction (`correct_lsb()`) before compression. This fixes a data representation issue specific to recordings made with the Open Ephys PXI system: the GUI rescales raw ADC (analog-to-digital converter) values in a way that introduces gaps in the stored integers (e.g., only multiples of 3 are used for NP2 data), wasting bits. LSB correction removes these gaps, reducing data entropy and improving compression ratios.

Aeon uses ONIX, which stores raw ADC counts directly without rescaling. The data already has no gaps (LSB = 1). AIND's own code explicitly skips LSB correction for ONIX recordings. This means one of AIND's compression advantages doesn't carry over -- Aeon's achievable compression ratios are the baseline NP2 numbers reported in the benchmarks, with no further improvement available from LSB correction.

---

## Why Not Adopt the Allen Brain Institute's Approach Directly

AIND's compression stack is: LSB correction + WavPack (level 3) + zarr, applied to raw data at ingest time. This is the gold standard for Neuropixels compression. Three things prevent Aeon from adopting it directly:

### 1. The Singularity container problem

Aeon runs Kilosort inside a Singularity container on the SWC HPC (`spikeinterface/kilosort4-base`). This is a pre-built image that Aeon doesn't control. WavPack requires two things installed inside the container: the Python package `wavpack-numcodecs` and the C shared library `libwavpack.so.1`. If the sorter tries to read WavPack-compressed data and either is missing, it fails with `ValueError: codec not available: 'wavpack'`. This exact scenario is documented in SpikeInterface issue #3618 -- a user compressed their data with WavPack, then Kilosort in Singularity couldn't read it. The resolution required building a custom Singularity image with the WavPack C library and Python package.

AIND avoids this because they control their entire container stack and can ensure WavPack is installed everywhere. Aeon would need to either build a custom Singularity image or get `wavpack-numcodecs` added to the upstream SpikeInterface container. Both are possible but add ongoing maintenance burden.

### 2. LSB correction does not apply to Aeon's data

As described above, LSB correction fixes a data representation issue introduced by the Open Ephys PXI system. Aeon uses ONIX, which doesn't have this issue. This means one of AIND's key steps for improving compression ratios simply doesn't apply.

### 3. The compression ratio gap is smaller than it looks

The headline numbers for WavPack (~3.6x lossless) come from Neuropixels 1.0 data. Aeon uses Neuropixels 2.0, where published benchmarks show:
- WavPack lossless: **~2.26x** on NP2
- Blosc-zstd: **~1.76x** on NP2

That's a difference of ~0.5x, not the ~2x gap you'd infer from NP1 numbers. On 8 TB of intermediate data, that translates to roughly 1 TB of additional savings -- meaningful, but not enough to justify the container risk.

### What it would take to adopt AIND's approach

If the team wants to pursue WavPack in the future, the steps would be:

1. Build a custom Singularity image for Kilosort that includes `wavpack-numcodecs` and `libwavpack`
2. Update the HPC deployment to use the custom image
3. Swap the compressor from Blosc-zstd to WavPack -- a one-line change once the infrastructure is ready

The Tier 1 recommendation (Blosc-zstd) is designed so this upgrade path is straightforward. The zarr format and pipeline changes are identical regardless of which codec is used; only the compressor object changes.

---

## Tier 1 Recommendation: Compress Intermediate Files

### What changes

Switch PreProcessing and PostProcessing from `binary_folder` to `zarr` format, using SpikeInterface's built-in Blosc-zstd compressor.

### Codec choice: Blosc-zstd

We recommend Blosc-zstd over WavPack for Aeon because:

- It is built into numcodecs -- no extra packages to install anywhere.
- No Singularity container compatibility issues (see above).
- The compression ratio difference on NP2 data is modest: ~1.76x (Blosc) vs ~2.26x (WavPack). On an 8 TB intermediate dataset, that's ~3.5 TB saved (Blosc) vs ~4.5 TB saved (WavPack). The extra 1 TB is not worth the dependency risk.
- Blosc is faster -- it's a multi-threaded C library with SIMD optimizations, designed for speed over ratio.

### Code changes required

**PreProcessing** (`spike_sorting.py`, ~lines 266-281):

Currently:
```python
si.core.write_binary_recording(
    recording=si_recording,
    file_paths=[binary_file_path],
    dtype="int16",
    **job_kwargs,
)
```

Becomes:
```python
si_recording.save(
    format="zarr",
    folder=recording_file.parent / "recording.zarr",
    **job_kwargs,
)
```

**PostProcessing** (`spike_sorting.py`, ~lines 528-535):

Currently:
```python
sorting_analyzer = si.create_sorting_analyzer(
    sorting=si_sorting,
    recording=si_recording,
    format="binary_folder",
    folder=analyzer_output_dir,
    sparse=True,
    overwrite=True,
)
```

Becomes:
```python
sorting_analyzer = si.create_sorting_analyzer(
    sorting=si_sorting,
    recording=si_recording,
    format="zarr",
    folder=analyzer_output_dir,
    sparse=True,
    overwrite=True,
)
```

### Impact on the SpikeSorting step

There is a trade-off here. Currently, the SpikeSorting step loads `recording.dat` directly and passes it to `run_sorter()`. Because the file is already binary-compatible, the Kilosort wrapper reuses it without re-writing.

If PreProcessing saves to zarr instead, the SpikeSorting step would need to load from zarr. The Kilosort wrapper internally always needs binary data, so it would decompress from zarr and write a temporary binary in the sorter's working folder. This means:

- **Long-term storage savings are real** -- the zarr on Ceph is ~43% smaller.
- **During sorting, a temporary binary still gets written** in the sorting output folder, cleaned up after.
- **Sorting I/O increases slightly** -- the sorter reads compressed data and writes a temporary binary.

Alternatively, `recording.dat` could be kept as a temporary file that gets written for sorting and deleted after sorting completes, while the zarr is the persistent copy. This would need some pipeline orchestration but avoids the double-storage problem.

### Expected savings

Blosc-zstd, NP2, per experiment (estimated):

| Component | Current size | Compressed | Savings |
|-----------|-------------|------------|---------|
| Preprocessed recording | ~8.1 TB | ~4.6 TB | ~3.5 TB |
| Sorting analyzer | ~2-8 TB | ~1.1-4.5 TB | ~0.9-3.5 TB |
| **Total intermediate savings** | | | **~4.4-7.0 TB** |

---

## Tier 2 Evaluation: Compressing Raw Acquisition Files

### What this means

Replacing the original `*_AmplifierData*.bin` files on Ceph with compressed zarr versions. The originals would be deleted after compression and verification.

### Why consider it

The raw data is the single largest consumer of disk space (~10.8 TB per experiment). It's also the most redundant -- the PreProcessing step reads these files, applies minimal transformations (unsigned-to-signed conversion, bandpass filter, common average reference), and writes a near-copy. If the raw data were compressed in place, the total disk footprint would drop substantially.

### How it would work

A standalone compression script (not part of the DataJoint pipeline) would:

1. Read raw binary chunks via `spikeinterface.extractors.read_binary()`
2. Save as zarr with Blosc-zstd: `recording.save(format="zarr", folder=...)`
3. Verify the compressed data by reading it back and comparing checksums or sample counts against the original
4. Delete the original `.bin` file only after verification passes

All downstream code that reads raw files (`PreProcessing.make_compute`, `EphysChunk` discovery, clock synchronization) would need to be updated to read from zarr instead of raw binary.

### Risk assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Data loss if compression has bugs | High | Blosc-zstd is lossless and battle-tested. Verify round-trip integrity before deleting originals. |
| Downstream code breaks | Medium | `EphysChunk` discovery uses `rglob("*_AmplifierData*.bin")`. This pattern would no longer match. All raw-file access paths need auditing. |
| Loss of original instrument output | Medium | The compressed data is bit-for-bit reconstructible (lossless), but the file is no longer in the format the acquisition system produced. Some teams consider this unacceptable on principle. |
| Ad-hoc scripts break | Low-Medium | Any analysis scripts outside the pipeline that directly read `.bin` files would need updating. |
| Clock files | Low | `*_Clock*.bin` files (uint64 timestamps) are small and could remain uncompressed, or be compressed separately. |

### What AIND does for reference

Allen Neural Dynamics compresses raw data as a routine step in their ingest pipeline, upstream of all processing. They use WavPack + LSB correction and store as zarr. However, their setup differs from Aeon's in a key way: AIND uses Open Ephys PXI for acquisition and benefits from LSB correction, while Aeon uses ONIX where LSB correction is not applicable.

### Our recommendation for Tier 2

Do not compress raw data in the initial implementation. The risk-to-reward ratio is less favorable than for intermediate files:

- Intermediate compression is low-risk (originals still exist on Ceph) and saves 4-7 TB per experiment.
- Raw compression saves an additional ~4-6 TB but requires deleting irreplaceable originals and updating all downstream file access.
- Revisit after Tier 1 is working and the team has gained confidence in zarr-based storage.

If the team does decide to pursue raw compression later, we recommend:

- Starting with a single experiment as a pilot
- Running the full pipeline on compressed raw data and comparing results to the uncompressed run
- Only proceeding to bulk compression after the pilot validates end-to-end correctness

---

## Implementation Considerations

### Backward compatibility with existing data

All data already processed and stored as `recording.dat` and `binary_folder` sorting analyzers remains readable. `si.load()` detects the format from the folder contents and handles both transparently. No existing data needs to be reprocessed.

### Historical data: compress or leave in place?

**Option A: Forward-only (recommended).** New processing runs save to zarr. Existing `recording.dat` files and `binary_folder` analyzers remain as-is. The pipeline code handles both formats when loading -- `si.load()` already does this transparently. The downside is that old data continues consuming the same disk space.

**Option B: Migrate historical data.** A standalone migration script converts existing files to zarr without reprocessing:

- For `recording.dat`: load via `se.read_binary()`, save with `recording.save(format="zarr")`, delete the original `.dat`
- For sorting analyzers: load via `si.load()`, convert with `analyzer.save_as(format="zarr")`, delete the original folder

This is a format conversion, not a reprocessing step -- no spike sorting or postprocessing is re-run. Once converted, any future reprocessing from any stage would work with the new code as-is.

The complication is DataJoint's File table tracking. The `PostProcessing.make_insert` method registers every file in the analyzer directory via `rglob("*")`. If we convert an existing analyzer from `binary_folder` to zarr, the registered files no longer match what's on disk. We'd need to delete the old File entries and re-insert new ones for the zarr contents. This is a metadata update, not difficult, but it's an extra step per converted analyzer.

**Our take:** Forward-only is simpler and lower-risk. Historical data migration is feasible but adds complexity (the migration script, File table updates, verification that converted data matches originals). Whether it's worth doing depends on how much historical processed data exists and how much disk pressure it's causing. If the bulk of the disk space problem is coming from ongoing experiments rather than legacy data, forward-only may be sufficient.

### SpikeSorting step changes

The SpikeSorting step currently:

1. Loads `si_recording.pkl` via `si.load()`
2. Loads `recording.dat` via `se.read_binary()` and verifies it matches
3. Passes the binary recording to `run_sorter()`

With zarr-based preprocessing output, this simplifies: load the zarr recording directly via `si.load()` and pass it to `run_sorter()`. The Kilosort wrapper handles the binary conversion internally (it always writes a temporary binary in its working folder regardless of input format). The `load_and_verify_binary_file` function and the `binary_compatible_with` check become unnecessary for new runs.

One detail: the current code sets `skip_kilosort_preprocessing = False` specifically because this allows the Kilosort wrapper to reuse the existing binary file without re-writing it. With a zarr input, the recording is no longer binary-compatible, so the wrapper will always write a temporary binary regardless of this flag's value. The flag's behavior doesn't change, but the optimization it currently enables (skipping the binary write) no longer applies. This is expected and correct -- it just means sorting I/O includes an extra decompression-and-write step.

### PostProcessing step changes

The PostProcessing step currently loads the sorting analyzer from a `binary_folder`. With zarr, `si.load()` handles both formats transparently. The main change is in `create_sorting_analyzer()` where `format="zarr"` replaces `format="binary_folder"`.

One additional consideration: the `make_insert` method does `analyzer_output_dir.rglob("*")` to register all files with DataJoint. Zarr stores data in a different directory structure (chunks as individual files inside array directories), so the set of files registered will change. This shouldn't break anything, but the File table entries will look different.

### Curation compatibility

The curation step (`spike_sorting_curation.py`) loads the raw sorting analyzer and saves a curated copy via `curated_analyzer.save(folder=..., overwrite=True)`. When no `format` is specified, SpikeInterface defaults to `binary_folder` regardless of the source analyzer's format. This means curated analyzers would still save as `binary_folder` even if the source is zarr. To get zarr output for curated analyzers, the curation code needs to be updated to use `save_as(format="zarr", folder=...)` instead. This is a small change but must not be overlooked.

### Chunk size tuning

Zarr compression operates on chunks. The chunk size along the time axis is controlled by `chunk_duration` in job kwargs. The current settings (`"2s"` for preprocessing, `"1s"` for postprocessing) are reasonable starting points and match what AIND uses. Larger chunks give slightly better compression but use more memory. We recommend keeping the current values initially and tuning only if needed.

### Zarr directory structure on Ceph

Zarr stores each chunk as a separate file on disk. For a 384-channel, 10-minute recording at 2-second chunk duration, that's roughly 300 chunk files plus metadata. Over hundreds of time blocks per experiment, this could create hundreds of thousands of small files. Some distributed filesystems handle large numbers of small files poorly (metadata overhead, inode limits). Whether this is a concern for SWC's Ceph deployment should be checked before committing to this approach. If it is a problem, larger chunk durations would reduce file count at the cost of higher memory usage during reads.

### Disk space during processing

During the SpikeSorting step, both the compressed zarr and a temporary binary (written by the Kilosort wrapper) will exist simultaneously. This is a transient increase in disk usage during sorting, but the temporary binary lives in the sorter's output folder and is not persisted long-term. The net long-term effect is still a ~43% reduction in preprocessed recording storage.

---

## Verification & Rollout Plan

Before deploying compression to production processing, we need to confirm two things: that the compressed data is bit-for-bit identical when decompressed, and that all downstream pipeline steps produce the same results.

### Step 1: Round-trip verification on a test block

Pick a single already-processed time block. Load the existing `recording.dat`, save it as zarr with Blosc-zstd, load the zarr back, and compare sample-by-sample against the original. This confirms lossless round-trip integrity. Do the same for a sorting analyzer: load from `binary_folder`, save as zarr, load back, and verify all extension data matches.

### Step 2: End-to-end pipeline comparison

Run the full pipeline (PreProcessing through PostProcessing) on one time block using the zarr-based code, and compare outputs against the existing results from the binary-based pipeline. Specifically:

- Do the spike sorting results (spike times, cluster assignments) match?
- Do the quality metrics and templates match?
- Does the curation step produce the same output?

If sorting results don't match exactly (which is possible -- Kilosort has stochastic elements), verify that the differences are within the normal run-to-run variation rather than caused by the format change.

### Step 3: Disk usage measurement

Measure the actual compression ratio achieved on Aeon's NP2 data. The published benchmark (~1.76x for Blosc-zstd on NP2) is an average across datasets. Aeon's actual ratio may vary depending on signal characteristics. This gives the team concrete numbers to decide whether the savings justify the change.

### Rollout

1. Implement the code changes on a feature branch
2. Run Steps 1-3 on the HPC using a single time block from an active experiment
3. Review results with the team
4. If approved, merge and apply to all new processing going forward
5. Decide on historical data migration based on measured savings and disk pressure

---

## Summary & Recommendations

This spec proposes adding lossless compression to Project Aeon's ephys data pipeline using SpikeInterface's zarr format with the built-in Blosc-zstd compressor. The changes are minimal (format parameter swaps in two pipeline steps) and the compression is fully lossless and transparent to downstream code.

### What we recommend

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Compress intermediate files? | **Yes -- implement now** | Low-risk, saves ~4-7 TB per experiment, originals still exist |
| Which codec? | **Blosc-zstd (SpikeInterface default)** | No extra dependencies, no container issues, fast |
| Compress raw acquisition files? | **Not yet** | Higher risk (deleting originals), revisit after Tier 1 is proven |
| Migrate historical data? | **Team decision** | Feasible but adds complexity; depends on disk pressure from old vs new data |
| Adopt AIND's WavPack approach? | **Not yet, but keep the door open** | Requires custom Singularity image; upgrade path is straightforward once infra is ready |

### Expected disk savings (Tier 1 only, per experiment)

- **Preprocessed recordings:** ~43% reduction (~3.5 TB saved on an 8.1 TB dataset)
- **Sorting analyzers:** ~43% reduction (~0.9-3.5 TB saved depending on extensions computed)
- **Total:** ~4.4-7.0 TB saved per experiment

### Open questions for team discussion

1. **Historical data migration:** Is the disk pressure coming mainly from ongoing experiments or legacy data? This determines whether migrating old data is worth the effort.
2. **Raw data compression appetite:** How does the team feel about replacing original acquisition files with lossless compressed versions? This is a policy decision as much as a technical one.
3. **WavPack upgrade path:** Is there interest in building a custom Singularity image to get the better compression ratios (~2.26x vs ~1.76x on NP2)? Or is the Blosc-zstd ratio sufficient?

### References

- Buccino AP et al. (2023). "Compression strategies for large-scale electrophysiology data." J. Neural Eng. DOI: 10.1088/1741-2552/acf5a4
- SpikeInterface zarr compression: `spikeinterface.core.zarrextractors` ([source](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/core/zarrextractors.py))
- AIND ephys transformation pipeline: [aind-ephys-transformation](https://github.com/AllenNeuralDynamics/aind-ephys-transformation)
- WavPack numcodecs wrapper: [wavpack-numcodecs](https://github.com/AllenNeuralDynamics/wavpack-numcodecs)
- SpikeInterface issue #3618 -- WavPack/Singularity container failure: [GitHub](https://github.com/SpikeInterface/spikeinterface/issues/3618)
