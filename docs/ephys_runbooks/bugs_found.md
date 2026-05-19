# Bugs Found During Ephys Runbook HPC Testing

> These notes were written by Claude (AI assistant) as Elissa tested the pipeline end-to-end on the HPC.

All of these were discovered by running the ephys pipeline against Adrian's golden baseline dataset (abcGolden01). The pipeline was previously tested on real data with DJ 0.14.x (social-ephys0.1, single-shank NP2004 probe). This was the first test with DJ 2.x and a multi-shank NP2.0 probe.

Bugs 1-14 are real issues — logic errors, incorrect assumptions, or framework incompatibilities that produced wrong results or blocked the pipeline. The DJ 2.x syntax migration and operational notes at the end are included for completeness but are routine cleanup, not bugs.

## 1. `get_probe_id()` — Wrong metadata path

**File:** `aeon/dj_pipeline/utils/ephys_utils.py`
**Symptom:** ProbeB returned fallback ID `NeuropixelsV2_ProbeB` instead of serial number.
**Root cause:** Code looked at `metadata["NeuropixelsV2e"]["ProbeConfigurationB"]` but the actual Metadata.yml structure is `metadata["Devices"]["NeuropixelsV2e"]["ConfigurationB"]`. The function was written without ever checking real metadata.
**Fix:** Corrected the path to match actual Metadata.yml structure.

## 2. `get_probe_id()` — No disabled/dummy probe detection

**File:** `aeon/dj_pipeline/utils/ephys_utils.py`
**Symptom:** `EphysEpoch.populate()` failed trying to look up ProbeA (a SpoofProbe/dummy) in the probe assignments JSON, which only had the real probe's serial.
**Root cause:** ProbeA is disabled (`Devices.ProbeA = "false"` in Metadata.yml) but the code had no concept of disabled probes. It treated all discovered probes equally.
**Fix:** Added check for `Devices.ProbeA/B = "false"` enable flag. `get_probe_id()` returns None for disabled probes. `EphysEpoch.make()` filters them out before probe assignment lookup.

## 3. `get_probe_id()` — Windows paths parsed on Linux

**File:** `aeon/dj_pipeline/utils/ephys_utils.py`
**Symptom:** ProbeB returned fallback ID even after metadata path fix. Serial extraction always failed.
**Root cause:** `GainCalibrationFileName` contains Windows-format paths (backslashes) because Bonsai runs on Windows. On Linux, `Path()` treats backslashes as literal filename characters, not separators. So `Path("Z:\\aeon\\code\\23299108854\\file.csv").parent.name` gives the whole string, not `"23299108854"`.
**Fix:** Use `PureWindowsPath` which correctly parses Windows paths on any platform.

## 4. `EphysSyncModel.ingest()` — Stores HARP seconds as ONIX timestamps

**File:** `aeon/dj_pipeline/ephys.py` (EphysSyncModel.ingest)
**Symptom:** `EphysChunk.ingest_chunks()` skips ALL amplifier data files with "No EphysSyncModel row covers ONIX range." Zero chunks ingested.
**Root cause:** The `HarpSyncModel.Reader` in `aeon/schema/ephys.py` has a misleading variable name: it calls `data.index.values` "onix_clock", but the CSV index is actually HARP seconds (since 1904-01-01, ~3.86 billion). The reader returns these as `clock_start`/`clock_end`. `EphysSyncModel.ingest()` stores them as `onix_ts_start`/`onix_ts_end`. Meanwhile, the actual ONIX hardware timestamps (from the amplifier Clock binary files) are in a completely different range (millions to trillions). The matching query in `ingest_chunks()` compares real ONIX ticks against HARP seconds and nothing overlaps.
**Fix:** Changed `data.index.values` → `data.clock.values` in HarpSyncModel.Reader.read().

## 5. `step01_register_experiment.py` — Missing sync model ingestion

**File:** `docs/ephys_runbooks/step01_register_experiment.py`
**Symptom:** Step 1 reports "0 chunks" and step 2 can't define blocks.
**Root cause:** The script called `EphysChunk.ingest_chunks()` without first calling `EphysSyncModel.ingest()`. Chunks need sync models to convert ONIX timestamps to HARP clock. Without sync models, all chunks are skipped.
**Fix:** Added `ingest_sync_models()` as step 6/8 before chunk ingestion.

## 6. `EphysChunk.File` stores wrong `directory_type`

**File:** `aeon/dj_pipeline/ephys.py` (EphysChunk.ingest_chunks)
**Symptom:** `PreProcessing.populate()` fails with `FileNotFoundError: No such file or directory: '/ceph/aeon/aeon/data/raw/AEON3/...'` — looking for ephys files under the behavior directory.
**Root cause:** `ingest_chunks()` discovers ephys files using the `"raw-ephys"` directory (correctly resolving to AEONX1), and stores `file_path` relative to that directory. But line 552 hardcodes `"directory_type": "raw"` in the `EphysChunk.File` insert. When `PreProcessing` later reconstructs the full path using `get_data_directory(key, directory_type="raw")`, it gets AEON3 (behavior) instead of AEONX1 (ephys).
**Fix:** Changed `"raw"` → `"raw-ephys"` on line 552.

## 7. `<filepath@dj_store>` / `<blob@dj_store>` codec incompatible with MariaDB

**File:** `aeon/dj_pipeline/spike_sorting.py` (PreProcessing.File, SpikeSorting.File, PostProcessing.File, SIExport.File, SortedSpikes.Unit)
**Symptom:** `TypeError: dict can not be used as parameter` when inserting into File part tables or SortedSpikes.Unit.
**Root cause:** The `<filepath@dj_store>` and `<blob@dj_store>` codecs' `encode()` returns a Python dict. On MySQL 5.7+, native JSON columns handle dict serialization natively. On MariaDB 10.3.28, `JSON` aliases to `longtext`, and `attr.json` is False for codec columns. The `json.dumps()` step in `__make_placeholder()` is skipped, and the raw dict reaches pymysql, which raises `TypeError`.

DataJoint 2.2.2 ([PR #1443](https://github.com/datajoint/datajoint-python/pull/1443)) fixes plain `json` columns by recovering `attr.json` from the `:json:` comment marker. But codec columns use markers like `:<filepath@dj_store>:`, so they're not covered by that fix.

**Workaround:** pymysql monkey-patch in `aeon/dj_pipeline/__init__.py` that teaches pymysql to serialize dicts as JSON strings. Must be applied before any DataJoint connection. Inert on MySQL (DataJoint serializes dicts before pymysql sees them).
**Upstream issue:** [datajoint/datajoint-python#1451](https://github.com/datajoint/datajoint-python/issues/1451)

## 8. `write_binary_recording` fork crashes on HPC

**File:** `aeon/dj_pipeline/spike_sorting.py` (PreProcessing.make_compute)
**Symptom:** `BrokenProcessPool: A process in the process pool was terminated abruptly` at chunk 0/90 or 1/90.
**Root cause:** Two issues compounded:
1. `n_jobs=0.8` resolves to `int(0.8 * os.cpu_count())` inside SpikeInterface's `fix_job_kwargs()`. On SLURM nodes, `os.cpu_count()` returns all physical cores (not the cgroup allocation), so a 4-CPU allocation spawns 16 workers.
2. `ProcessPoolExecutor` with `fork` mp_context crashes on the SWC HPC — fork + NumPy/BLAS runtime incompatibility causes worker segfaults before any data is processed.
**Status:** Not reproducible as of 2026-05-18. Reverted to `n_jobs=-1` with a troubleshooting note in step03.

## 9. `create_electrode_config()` inserts full probe instead of active channels

**File:** `docs/ephys_runbooks/step01_register_experiment.py` (create_electrode_config)
**Symptom:** `PreProcessing.populate()` produces garbled data. Preprocessed recordings report 5120 channels and 135-second duration instead of 384 channels and 30 minutes.
**Root cause:** The probeinterface channel map JSON describes the full NP2.0 probe — all 5120 electrode sites across 4 shanks. Only 384 are actively recorded; the `device_channel_indices` array marks active contacts (value 0-383 = raw channel index) and inactive ones (value -1). `create_electrode_config()` read `contact_ids` (all 5120) but never checked `device_channel_indices`, so it inserted all 5120 sites as the electrode configuration. Downstream, SpikeInterface's `read_binary()` reshapes the flat int16 array into a (samples, 5120) matrix instead of (samples, 384).
**Fix:** Filter to contacts where `device_channel_indices >= 0` before inserting. Only the 384 active electrode sites are inserted into ElectrodeConfig.Electrode.

## 10. `EphysBlockInfo.Channel.make()` assumes channel order matches electrode site order

**File:** `aeon/dj_pipeline/ephys.py` (EphysBlockInfo.Channel, lines 682-688)
**Symptom:** Kilosort would receive a probe geometry where every channel's physical position is wrong — shank 3 neural activity placed at shank 0 coordinates, etc.
**Root cause:** The code sorts electrodes by site index and assigns channel_idx 0, 1, 2, ... in that order. But the actual hardware mapping (from `device_channel_indices` in the probeinterface JSON) is non-sequential — e.g., raw channel 0 → electrode site 3954 (shank 3), raw channel 210 → electrode site 114 (shank 0). The code never reads the probeinterface JSON to get the real channel-to-electrode mapping.
**Why it was never caught:** Previously only run on social-ephys0.1 (single-shank NP2004 probe, contiguous electrodes 0-383). For that setup, sorting by site index happens to match the raw channel order.
**Fix:** Read `device_channel_indices` from the probeinterface JSON. Build a `{electrode_site: raw_channel_idx}` mapping. Use the real raw channel index instead of enumerate position.

## 11. `SyncedSpikes.make()` queries link table without join

**File:** `aeon/dj_pipeline/spike_sorting.py` (SyncedSpikes.make, line 1003)
**Symptom:** `DataJointError: Attribute 'onix_ts_end' not found` when populating SyncedSpikes.
**Root cause:** `SyncedSpikes.make()` queries `EphysChunk.SyncModel` for `onix_ts_start`, `onix_ts_end`, and `sync_model`, but `EphysChunk.SyncModel` is a link-only part table — it only contains primary keys. The actual data attributes live on `EphysSyncModel`. The query needs a join.
**Fix:** Changed `(ephys.EphysChunk.SyncModel & ...)` to `(ephys.EphysChunk.SyncModel * ephys.EphysSyncModel & ...)`.

## 12. `fetch1("file")` returns `ObjectRef` in DJ 2.x, not a path

**Files:** `aeon/dj_pipeline/spike_sorting.py` (6 locations), `aeon/dj_pipeline/spike_sorting_curation.py` (2 locations)
**Symptom:** `TypeError: expected str, bytes or os.PathLike object, not ObjectRef` when `SpikeSorting.populate()` tries to load the preprocessed recording via `si.load()`. All 12 SLURM sorting jobs failed in under 60 seconds.
**Root cause:** In DJ 0.14.x, `fetch1("file")` on a `<filepath@dj_store>` column returned a file path string. In DJ 2.x, it returns an `ObjectRef` — a storage-agnostic handle. Code that passes the fetched value directly to `Path()` or `si.load()` breaks.
**Fix:** Added `.full_path` to all 8 `fetch1("file")` calls.
**Note:** This bug was masked by bug 7 — File part table entries didn't exist until the pymysql monkey-patch was active AND the database was rebuilt from scratch.

## 13. `step05_unit_matching.py` — DJ 2.x `to_arrays()` returns array directly for single attr

**File:** `docs/ephys_runbooks/step05_unit_matching.py` (line 152)
**Symptom:** `TypeError: object of type 'datetime.datetime' has no len()` when running unit matching.
**Root cause:** DJ 2.x `to_arrays("block_start")` with a single attribute returns the numpy array directly, not wrapped in a tuple. `.to_arrays("block_start")[0]` indexes the first *element* of the array (a single datetime) instead of unpacking the array from a tuple.
**Fix:** Removed the `[0]` index.

## 14. `PostProcessing.make()` crashes on orphaned `sorting_analyzer` directory

**File:** `aeon/dj_pipeline/spike_sorting.py` (PostProcessing.make)
**Symptom:** `FileExistsError` when SpikeInterface tries to create the `sorting_analyzer` output directory.
**Root cause:** Killed `srun` sessions or SLURM time limits leave behind partially-written `sorting_analyzer` directories on Ceph. SpikeInterface refuses to overwrite existing directories. The directory exists but has no corresponding PostProcessing DB entry (since the insert never completed), so `populate()` retries and hits the safety check.
**Fix:** Manual: identify orphaned directories (exist on disk but have no DB entry) and `rm -rf` them before re-running.

---

## DJ 2.x Syntax Migration

These are not bugs — they're mechanical updates needed because `spike_sorting.py` and `spike_sorting_curation.py` were written for DJ 0.14.x and the syntax changed in DJ 2.x. Grouped here for completeness.

- **Codec syntax** (`spike_sorting.py`, `spike_sorting_curation.py`): DJ 2.x requires angle brackets around external store codecs. Changed 4 `filepath@dj_store` and 5 `blob@dj_store` to `<filepath@dj_store>` / `<blob@dj_store>`. Same change in `ManualCuration.File`.
- **Core type aliases** (`spike_sorting.py`): DJ 2.x introduced `float64`, `int32`, etc. Replaced bare `float` and `int` in table definitions to suppress warnings.
- **JSON case sensitivity** (`spike_sorting.py`, SortingQuality.Metric): DJ 2.x's `CORE_TYPE_MAP` uses lowercase keys. Changed `JSON` → `json`.

## Operational Note: Pre-allocated `recording.dat` passes verification when corrupt

SpikeInterface's `write_binary_recording` pre-allocates the output file at full size before writing chunks. A crash at chunk 4/90 leaves a full-sized file that is mostly zeros. `load_and_verify_binary_file` only checks `get_num_samples()` (derived from file size), not content. The file passes verification. Manual workaround: always delete `recording.dat` after a crashed write before re-running.

## Summary

Bugs 1-3 are all in `get_probe_id()`, which was merged to main but never tested against actual Neuropixels data. Bug 4 is in the upstream schema reader. Bug 5 is a missing step in the guide script. Bug 6 is a hardcoded directory type. Bug 7 is a DJ 2.x + MariaDB framework incompatibility (reported upstream as [#1451](https://github.com/datajoint/datajoint-python/issues/1451)). Bug 8 is a SpikeInterface + HPC environment issue. Bugs 9-10 are probeinterface channel map parsing errors that only manifest with multi-shank probes. Bugs 11-13 are DJ 2.x API changes. Bug 14 is an operational issue with orphaned directories from killed SLURM jobs.

The common thread: the pipeline was previously tested on DJ 0.14.x with a single-shank probe (social-ephys0.1). The DJ 2.x migration changed type syntax, external store codecs, fetch return types, and query APIs. The multi-shank probe exposed channel mapping assumptions that were correct for single-shank but wrong in general. And several upstream components (metadata parsing, sync model reader, split raw directories) had gaps that only surfaced when exercised end-to-end against this dataset.
