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

### Completed (Elissa, 2026-05-26)

1. **Implemented** CompressionTest table in `ephys.py`
2. **Implemented** PreProcessing/SpikeSorting/PostProcessing/Curation format branching in `spike_sorting.py` and `spike_sorting_curation.py`
3. **Implemented** report script at `scripts/compression_report.py`
4. **Pushed** two commits to `es/compression-spec`:
   - `647dbfa` — All spec docs + code implementation
   - `9050730` — Fix zarr object-codec error (probe `contact_shapes` property is object dtype, stripped before zarr save) and add `n_jobs` field to CompressionTest
5. **Workstream 1 tested (3/12 chunks):**
   - Compression ratio: ~1.95x (better than the spec's 1.76x estimate from published benchmarks)
   - Compress: ~77s per 13.82 GB chunk (n_jobs=1, single core)
   - Decompress: ~13s per chunk
   - All checksums pass (lossless verified)
   - Codec: blosc-zstd-5-bitshuffle
6. **Workstream 2 partially tested:**
   - Inserted zarr paramset `400_zarr` (copy of `400` with `save_format: "zarr"`)
   - Inserted SortingTask for block 2026-05-11T07:49:46, shank0
   - PreProcessing completed (zarr write successful)
   - SpikeSorting submitted as SLURM GPU job 3022011, pending GPU node availability
   - PostProcessing and Curation not yet run

### Remaining work (Thinh picks up)

**What still needs to happen:**

1. SpikeSorting GPU job (3022011) needs to complete. If it's still pending or failed, resubmit it (see instructions below).
2. Run PostProcessing on the zarr result (CPU only).
3. Run Curation on the zarr result (CPU only).
4. Run `scripts/compression_report.py` to get the full timing comparison between binary (paramset `400`) and zarr (`400_zarr`).
5. Optionally, run `CompressionTest.populate()` on more chunks for a larger sample of compression ratios.

**HPC setup:**

The testing environment is already set up at `~/ProjectAeon/foragingABC_analysis` with `aeon_mecha` as a submodule on `es/compression-spec`. The database is `u_elissas_` on `aeon-db`. To get started:

```bash
ssh aeon-hpc

srun --pty -c 4 --mem 32G -t 4:00:00 bash

cd ~/ProjectAeon/foragingABC_analysis

# Make sure the submodule is on the right branch
git -C submodules/aeon_mecha fetch origin
git -C submodules/aeon_mecha checkout es/compression-spec
git -C submodules/aeon_mecha pull

# Install
module load uv
uv pip install -e "./submodules/aeon_mecha[spike_sorting]"
```

**Check SpikeSorting job status:**

```bash
squeue -u elissas    # check if job 3022011 is still queued/running
cat ~/ProjectAeon/foragingABC_analysis/slurm_output/*3022011*    # check output if finished
```

If the job failed or needs to be resubmitted, the SLURM script and Python script are already on the HPC at `~/zarr_spike_sorting.sh` and `~/zarr_spike_sorting.py`. Resubmit with `sbatch ~/zarr_spike_sorting.sh`.

**Run PostProcessing (after SpikeSorting completes):**

```bash
.venv/bin/python << 'EOF'
from aeon.dj_pipeline import spike_sorting as ss

import sys, traceback
sys.excepthook = lambda *args: traceback.print_exception(*args)

ss.PostProcessing.populate(
    ss.SortingTask & "paramset_id='400_zarr'",
    display_progress=True,
)
EOF
```

**Run Curation (after PostProcessing completes):**

```bash
.venv/bin/python << 'EOF'
from aeon.dj_pipeline import spike_sorting_curation as ssc

import sys, traceback
sys.excepthook = lambda *args: traceback.print_exception(*args)

ssc.ApplyOfficialCuration.populate(display_progress=True)
EOF
```

**Generate the report:**

```bash
.venv/bin/python scripts/compression_report.py
```

**Important notes:**

- The `sys.excepthook` override must come AFTER the DataJoint import. DJ 2.x installs its own excepthook that swallows tracebacks.
- Use `.venv/bin/python` directly, not `uv run` (which tries to resolve all extras and hits dependency conflicts).
- The `u_elissas_` database prefix is set via `dj_local_conf.json` in the project directory.

### Cleaning up failed test runs

If the pipeline fails partway through a zarr test run, clean up from the deepest table first. For example: delete `PostProcessing` entries before `SpikeSorting` entries before `PreProcessing` entries. Then delete the partially-written output directories on disk.
