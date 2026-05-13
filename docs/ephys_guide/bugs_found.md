# Bugs Found During Ephys Guide HPC Testing (2026-05-12)

All of these were discovered by running the ephys pipeline (PR #548, merged to main) against real data for the first time using Adrian's golden baseline dataset (abcGolden01).

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
**Status:** CONFIRMED and FIXED. Changed `data.index.values` → `data.clock.values` in HarpSyncModel.Reader.read().

## 5. `step01_register_experiment.py` — Missing sync model ingestion

**File:** `docs/ephys_guide/step01_register_experiment.py`
**Symptom:** Step 1 reports "0 chunks" and step 2 can't define blocks.
**Root cause:** The script called `EphysChunk.ingest_chunks()` without first calling `EphysSyncModel.ingest()`. Chunks need sync models to convert ONIX timestamps to HARP clock. Without sync models, all chunks are skipped.
**Fix:** Added `ingest_sync_models()` as step 6/8 before chunk ingestion.

## 6. `spike_sorting.py` — DJ 0.14.x type syntax rejected by DJ 2.x

**File:** `aeon/dj_pipeline/spike_sorting.py`
**Symptom:** `from aeon.dj_pipeline import spike_sorting` fails with `DataJointError: Unsupported attribute type filepath@dj_store`.
**Root cause:** DJ 2.x replaced the bare `filepath@store` / `blob@store` / `attach@store` syntax from DJ 0.14.x with angle-bracket codec syntax: `<filepath@store>`, `<blob@store>`, etc. The `spike_sorting.py` module had 4 `filepath@dj_store` and 5 `blob@dj_store` attributes in the old syntax.
**Fix:** Added angle brackets around all 9 occurrences. Pure syntax change, no logic changes.

## 7. `spike_sorting.py` — Native SQL types trigger DJ 2.x warnings

**File:** `aeon/dj_pipeline/spike_sorting.py`
**Symptom:** DJ 2.x emits warnings like `Native type 'float' is used in attribute 'execution_duration'. Consider using a core DataJoint type for better portability.`
**Root cause:** DJ 2.x introduced core type aliases (`float64`, `int32`, etc.) that map to SQL types portably across backends. Using bare `float` or `int` in table definitions triggers deprecation-style warnings.
**Fix:** Replaced all `float` → `float64` and `int` → `int32` in table definitions.

## 8. `spike_sorting.py` — `JSON` type case sensitivity

**File:** `aeon/dj_pipeline/spike_sorting.py` (SortingQuality.Metric)
**Symptom:** `ValueError: Unknown core type: JSON` when the schema decorator tries to declare the table.
**Root cause:** DJ 2.x's `CORE_TYPE_MAP` uses lowercase keys (`"json"` → `"json"`). The table definition had `qc_metrics: JSON` (uppercase). The regex match is case-insensitive so it matches, but the subsequent dict lookup in `core_type_to_sql()` is case-sensitive and fails.
**Fix:** Changed `JSON` → `json` in the table definition.

## 9. `ephys.py` — `EphysChunk.File` stores wrong `directory_type`

**File:** `aeon/dj_pipeline/ephys.py` (EphysChunk.ingest_chunks)
**Symptom:** `PreProcessing.populate()` fails with `FileNotFoundError: No such file or directory: '/ceph/aeon/aeon/data/raw/AEON3/...'` — looking for ephys files under the behavior directory.
**Root cause:** `ingest_chunks()` discovers ephys files using the `"raw-ephys"` directory (correctly resolving to AEONX1), and stores `file_path` relative to that directory. But line 552 hardcodes `"directory_type": "raw"` in the `EphysChunk.File` insert. When `PreProcessing` later reconstructs the full path using `get_data_directory(key, directory_type="raw")`, it gets AEON3 (behavior) instead of AEONX1 (ephys).
**Fix:** Changed `"raw"` → `"raw-ephys"` on line 552.

## 10. `spike_sorting.py` — `<filepath@dj_store>` codec incompatible with MariaDB 10.3.28

**File:** `aeon/dj_pipeline/spike_sorting.py` (PreProcessing.File, SpikeSorting.File, PostProcessing.File, SIExport.File)
**Symptom:** `TypeError: dict can not be used as parameter` when `PreProcessing.make_insert()` tries to insert into the File part table.
**Root cause:** The `<filepath@dj_store>` codec's `encode()` method (in `datajoint/builtin_codecs/filepath.py`) returns a Python dict `{path, store, size, is_dir, timestamp}`. This dict is meant to be stored in a JSON column. On MySQL 5.7+, native JSON columns handle dict serialization natively. On MariaDB 10.3.28, `JSON` aliases to `longtext`, and pymysql cannot serialize a Python dict to a text value — it raises `TypeError: dict can not be used as parameter`.

This is a DJ 2.x framework issue: the filepath codec assumes native JSON column support, which MariaDB 10.3.28 does not provide. The same issue likely affects `<blob@dj_store>` and `<attach@dj_store>` codecs.

**Workaround:** Wrap `File.insert()` calls in try/except (skip on MariaDB), add fallback path reconstruction in downstream `File.fetch()` sites. Forward-compatible: when SWC migrates to MySQL, the inserts will succeed and the fallbacks become dead code.

**Proper fix:** Either DJ 2.x needs to json.dumps() the codec output before passing to pymysql on MariaDB, or SWC needs to upgrade from MariaDB 10.3.28 to MySQL 5.7+. The MySQL upgrade is already being discussed with the IT team.

## Summary

Bugs 1-3 are all in `get_probe_id()`, which was merged to main via PR #548 but never tested against actual Neuropixels data. Bug 4 is in the upstream schema reader / ingestion pipeline. Bug 5 is in the guide script (our code). Bugs 6-8 are DJ 2.x migration issues in spike_sorting.py. Bug 9 is a split-raw-directory oversight in ephys.py. Bug 10 is a DJ 2.x + MariaDB framework incompatibility.

The common thread for bugs 1-5: none of this code was ever run against real data before it was merged. The metadata structure was assumed, the platform differences were assumed, the sync clock domains were assumed. Every assumption was wrong.

The common thread for bugs 6-10: spike_sorting.py was written for DJ 0.14.x and never updated for DJ 2.x, and the split-raw-directory feature was not propagated to all code paths.
