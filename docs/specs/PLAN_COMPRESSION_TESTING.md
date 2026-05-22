# Compression Testing Plan

Branch: `es/compression-spec`
Goal: Empirically test lossless compression on both raw ephys data and intermediate pipeline files, generate a report for the team, and leave the branch clean for Thinh to continue testing.

**Out of scope:** File deletion, S3/Glacier archival, cleanup scripts. This plan is testing only -- we create new files, measure things, and report results. No originals are modified or deleted.

---

## Workstream 1: Raw Data Compression Test

### What

A new DataJoint table (`CompressionTest`) that compresses each raw ephys chunk, decompresses it, and verifies the result matches the original. Records compression ratio, timing, and checksum pass/fail.

### Table definition

```
CompressionTest (dj.Computed)
    -> EphysChunk
    ---
    original_size_bytes    : bigint        # size of the raw AmplifierData .bin file
    compressed_size_bytes  : bigint        # size of the zarr output
    compression_ratio      : float         # original / compressed
    compression_time_s     : float         # wall-clock seconds to compress
    decompression_time_s   : float         # wall-clock seconds to decompress
    checksum_match         : bool          # True if decompressed == original
    mismatch_details       : varchar(1000) # empty if match, error details if not
    num_channels           : int           # channels in this chunk
    num_samples            : bigint        # total samples in this chunk
    sampling_frequency     : float         # Hz
    codec_name             : varchar(64)   # e.g. "blosc-zstd-5-bitshuffle"
    execution_time         : datetime      # when this test ran
```

### make() logic

1. Find the `*_AmplifierData*.bin` file for this chunk from `EphysChunk.File`
2. Use the same hardcoded NP2 constants as `PreProcessing.make_compute()` (fs=30kHz, gain=3.05176, dtype=uint16, etc.) for consistency with the existing pipeline
3. Load via `se.read_binary()` with those parameters
4. **Compress:** `recording.save(format="zarr", folder=temp_zarr_path)` with the default Blosc-zstd compressor. Time this.
5. Measure compressed size (sum of all files in the zarr directory)
6. **Decompress:** Load the zarr via `si.load()`, write back to binary via `write_binary_recording()` to a temp file. Time this.
7. **Verify:** Compare the original `.bin` file and decompressed `.bin` file byte-for-byte. Record match/mismatch. On mismatch, record the first N differing positions and values.
8. **Clean up:** Delete the temporary zarr directory and decompressed binary. Only the table entry persists. Use `try/finally` to ensure temp files are cleaned up even if verification or earlier steps fail.

### Where temp files go

Under the sorting root directory in a `compression_test_tmp/` subdirectory, organized by chunk. Cleaned up after each chunk completes (both success and failure).

### Codec name format

Construct the `codec_name` string from the Blosc compressor parameters: `f"blosc-{cname}-{clevel}-{shuffle_name}"` where shuffle_name maps `Blosc.BITSHUFFLE` to `"bitshuffle"`. For the default compressor this produces `"blosc-zstd-5-bitshuffle"`.

### Implementation steps

1. Add `CompressionTest` class to `ephys.py` (after `EphysBlockInfo`)
2. Implement `make()` with the logic above
3. Test on a single chunk to validate

---

## Workstream 2: Compressed Intermediate Files + Sorting Speed Test

### What

Add support for saving PreProcessing output and SortingAnalyzer in compressed zarr format, controlled by a parameter in `SortingParamSet`. Then run the pipeline with both formats on the same data and compare execution times.

### Parameter mechanism

Add a `"save_format"` key to the `SortingParamSet.params` dictionary. Values: `"binary"` (default, current behavior) or `"zarr"` (compressed). No schema change needed -- this is a new key in the existing params blob.

To test, insert a new `SortingParamSet` entry:
```python
SortingParamSet.insert1({
    "paramset_id": "ks4_zarr",
    "sorting_method": "kilosort4",
    "paramset_description": "Kilosort4 with zarr-compressed intermediates",
    "params": {
        # ... same SI_SORTING_PARAMS as the binary version ...
        "save_format": "zarr",
    },
})
```

### Code changes

**PreProcessing.make_compute() (~lines 266-281 of spike_sorting.py):**

Check `params.get("save_format", "binary")`:
- `"binary"`: current code path (`write_binary_recording()` to `recording.dat`)
- `"zarr"`: `si_recording.save(format="zarr", folder=recording_dir / "recording.zarr", **job_kwargs)` using `get_job_kwargs()` for proper n_jobs handling. Add an idempotency check (skip if `recording.zarr/` already exists, similar to the existing `recording.dat` check).

**SpikeSorting.make_compute() (~lines 370-410):**

Check `params.get("save_format", "binary")`:
- `"binary"`: current code path (load `recording.dat` via `se.read_binary()`, verify, pass to sorter)
- `"zarr"`: load `recording.zarr` via `si.load()`, pass directly to `run_sorter()`. Skip `load_and_verify_binary_file` and `binary_compatible_with` check (not applicable for zarr). The zarr recording preserves probe geometry from when PreProcessing saved it, so no re-attachment is needed. The Kilosort wrapper's `_setup_recording` will handle writing a temporary binary internally.

**PostProcessing.make_compute() (~lines 528-535):**

Check `params.get("save_format", "binary")`:
- `"binary"`: `create_sorting_analyzer(format="binary_folder", ...)`
- `"zarr"`: `create_sorting_analyzer(format="zarr", ...)`

**Curation step (spike_sorting_curation.py):**

`ApplyOfficialCuration.make()` calls `curated_analyzer.save(folder=..., overwrite=True)` which defaults to `binary_folder` regardless of the source analyzer's format. When the source paramset specifies zarr, update this to use `curated_analyzer.save_as(format="zarr", folder=...)` instead.

**Downstream tables (QualityMetrics, CuratedSpikeSorting, SIExport, etc.):**

These load the sorting analyzer via `si.load()`, which auto-detects the format. Review each downstream table to confirm it works with both formats, including `SIExport` which calls `si.exporters.export_report()`. The main thing to watch for: any code that constructs file paths assuming `binary_folder` structure.

**PreProcessing.make_insert file registration:**

The current `make_insert` explicitly excludes `recording.dat` from the File table. With zarr output, there's no `recording.dat`, but a zarr directory contains many chunk files. Decide whether to register individual zarr chunk files or treat the zarr directory as a single logical unit (e.g., register only the top-level `recording.zarr` path). Registering hundreds of chunk files per block would bloat the File table.

### Speed comparison

Run the same `EphysBlock` with two `SortingParamSet` entries (one binary, one zarr). The `execution_duration` fields already recorded by `PreProcessing`, `SpikeSorting`, and `PostProcessing` provide the timing comparison directly from the database. No extra instrumentation needed.

### Implementation steps

1. Modify `PreProcessing.make_compute()` with save_format branching
2. Modify `PreProcessing.make_insert()` to handle zarr file registration
3. Modify `SpikeSorting.make_compute()` with save_format branching
4. Modify `PostProcessing.make_compute()` with save_format branching
5. Modify `ApplyOfficialCuration.make()` in `spike_sorting_curation.py` to use `save_as(format="zarr")` when paramset specifies zarr
6. Review downstream tables for format compatibility (QualityMetrics, CuratedSpikeSorting, SIExport, SortedSpikes)
7. Verify that `si.load_sorting_analyzer()` works on zarr-format analyzer directories (check whether `.zarr` extension is required)
8. Test end-to-end on a single block with zarr format

---

## Report Script

A standalone script at `scripts/compression_report.py` that queries both workstreams and prints a summary:

**From CompressionTest (Workstream 1):**
- Number of chunks tested, pass/fail counts
- Average/min/max compression ratio
- Average/min/max compression time per chunk
- Average/min/max decompression time per chunk
- Total original size vs total compressed size
- Any checksum failures with details

**From pipeline tables (Workstream 2):**
- PreProcessing execution_duration: binary vs zarr
- SpikeSorting execution_duration: binary vs zarr
- PostProcessing execution_duration: binary vs zarr
- Total pipeline time: binary vs zarr
- Size of recording.dat vs recording.zarr
- Size of binary_folder analyzer vs zarr analyzer

---

## Execution Order

### Before Wednesday (Elissa + Claude)

1. **Implement CompressionTest table** in `ephys.py`
2. **Implement PreProcessing/SpikeSorting/PostProcessing format branching** in `spike_sorting.py`
3. **Implement report script** in `scripts/compression_report.py`
4. **Push to `es/compression-spec`**
5. **On HPC:** Run `CompressionTest.populate()` on a handful of chunks to validate raw compression testing works
6. **On HPC:** Insert a zarr SortingParamSet entry, create a SortingTask, run one block through the pipeline to validate end-to-end
7. **Push any fixes** from HPC testing

### After Wednesday (Thinh picks up)

1. Run `CompressionTest.populate()` on a larger set of chunks (or all chunks from a test experiment)
2. Run more blocks through the pipeline with zarr format
3. Run `scripts/compression_report.py` to generate the full report
4. Share report with team for decision-making

### Cleaning up failed test runs

If the pipeline fails partway through a zarr test run, clean up from the deepest table first. For example: delete `PostProcessing` entries before `SpikeSorting` entries before `PreProcessing` entries. Then delete the partially-written output directories on disk.
