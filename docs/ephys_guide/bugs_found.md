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

## 11. `spike_sorting.py` — `write_binary_recording` fork crashes on HPC

**File:** `aeon/dj_pipeline/spike_sorting.py` (PreProcessing.make_compute)
**Symptom:** `BrokenProcessPool: A process in the process pool was terminated abruptly` at chunk 0/90 or 1/90. Happens with both 16 and 4 workers. Workers die immediately on first computation.
**Root cause:** Two issues compounded:
1. `n_jobs=0.8` resolves to `int(0.8 * os.cpu_count())` inside SpikeInterface's `fix_job_kwargs()`. On SLURM nodes, `os.cpu_count()` returns all physical cores (not the cgroup allocation), so a 4-CPU allocation spawns 16 workers.
2. Even with `n_jobs=4`, `ProcessPoolExecutor` with `fork` mp_context crashes immediately. This is a fork + NumPy/BLAS runtime incompatibility on the SWC HPC nodes — the BLAS threading state is not fork-safe, causing worker segfaults before any data is processed.
**Fix:** Set `n_jobs=1` (sequential, no forking). Slower (~48 min per 30-min block) but reliable.

## 12. `spike_sorting.py` — Pre-allocated `recording.dat` passes verification when corrupt

**File:** `aeon/dj_pipeline/spike_sorting.py` (`load_and_verify_binary_file`)
**Symptom:** After a crashed `write_binary_recording`, re-running `PreProcessing.populate()` silently accepts the corrupt file and inserts a PreProcessing entry. Downstream sorting produces garbage from mostly-zero data.
**Root cause:** SpikeInterface's `write_binary_recording` pre-allocates the output file at full size (seeks to end, writes one zero byte) before writing chunks. A crash at chunk 4/90 leaves a full-sized file that is mostly zeros. `load_and_verify_binary_file` only checks `get_num_samples()` (derived from file size), not content. The file passes verification.
**Status:** Known limitation. Manual workaround: always delete `recording.dat` after a crashed write before re-running. A proper fix would add a content checksum or write a completion marker file.

## 13. `step01_register_experiment.py` — `create_electrode_config()` inserts full probe instead of active channels

**File:** `docs/ephys_guide/step01_register_experiment.py` (create_electrode_config)
**Symptom:** `PreProcessing.populate()` produces garbled data. Preprocessed recordings report 5120 channels and 135-second duration instead of 384 channels and 30 minutes.
**Root cause:** The probeinterface channel map JSON (`M81_ProbeB_4Shanks_1000_to_1700_um.json`) describes the full NP2.0 probe — all 5120 electrode sites across 4 shanks. Only 384 are actively recorded; the `device_channel_indices` array marks active contacts (value 0-383 = raw channel index) and inactive ones (value -1). `create_electrode_config()` read `contact_ids` (all 5120) but never checked `device_channel_indices`, so it inserted all 5120 sites as the electrode configuration. Downstream, `PreProcessing.make()` uses `num_channels = len(ElectrodeConfig.Electrode)` = 5120 to read the 384-channel raw binary. SpikeInterface's `read_binary()` reshapes the flat int16 array into a (samples, 5120) matrix instead of (samples, 384) — each "sample" contains data from ~13 real time points interleaved across real channels.
**Fix:** Filter to contacts where `device_channel_indices >= 0` before inserting. Only the 384 active electrode sites are inserted into ElectrodeConfig.Electrode. Config named "0-383" (channel range, not site range).

## 14. `ephys.py` — `EphysBlockInfo.Channel.make()` assumes channel order matches electrode site order

**File:** `aeon/dj_pipeline/ephys.py` (EphysBlockInfo.Channel, lines 682-688)
**Symptom:** Kilosort would receive a probe geometry where every channel's physical position is wrong — shank 3 neural activity placed at shank 0 coordinates, etc.
**Root cause:** The `Channel` table's `channel_idx` column is defined as "idx of the raw data" — the column index in the binary file (0-383). The code that populates it:

```python
electrode_df = (ElectrodeConfig.Electrode & econfig).keys(order_by="electrode")
self.Channel.insert(
    {**key, "channel_idx": ch_idx, "channel_name": ch_idx, **ch_key}
    for ch_idx, ch_key in enumerate(electrode_df)
)
```

This sorts electrodes by site index (114, 115, ..., 4049) and assigns channel_idx 0, 1, 2, ... in that order. The result: channel_idx 0 → electrode site 114. But the actual hardware mapping (from `device_channel_indices` in the probeinterface JSON) is: raw channel 0 → electrode site 3954 (shank 3), raw channel 210 → electrode site 114 (shank 0). The code never reads the probeinterface JSON to get the real channel-to-electrode mapping. It just assumes electrodes sorted by site index correspond to channels 0, 1, 2, ... in order.

This matters because `PreProcessing.make_compute()` (spike_sorting.py line 258) calls `si_probe.set_device_channel_indices(electrodes_df["channel_idx"].values)`, which tells SpikeInterface which raw data column each probe contact reads from. With the wrong mapping, the probe geometry sent to Kilosort has every channel at the wrong physical location.

**Why it was never caught:** The pipeline was previously only run on social-ephys0.1, which used a single-shank NP2004 probe recording a contiguous block of electrodes 0-383. For that setup, sorting by site index happens to match the raw channel order, so the enumerate approach gave the correct mapping. Any multi-shank probe, or a single-shank probe with non-contiguous electrode selection, would have the same bug.

**Fix:** Read `device_channel_indices` from the probeinterface JSON in the epoch directory on Ceph. Build a `{electrode_site: raw_channel_idx}` mapping. Use the real raw channel index instead of enumerate position. The epoch directory path is discoverable from the EphysEpoch table. See code changes below.

**Required code changes (ephys.py, EphysBlockInfo.make, lines 680-688):**

Before:
```python
# Channel
electrode_df = (ElectrodeConfig.Electrode & econfig).keys(order_by="electrode")
self.Channel.insert(
    (
        {**key, "channel_idx": ch_idx, "channel_name": ch_idx, **ch_key}
        for ch_idx, ch_key in enumerate(electrode_df)
    ),
)
```

After:
```python
# Channel — use real hardware channel mapping from the probeinterface JSON
channel_map = _load_device_channel_map(key["experiment_name"], econfig)
electrode_df = (ElectrodeConfig.Electrode & econfig).keys(order_by="electrode")
self.Channel.insert(
    (
        {**key, "channel_idx": channel_map[ch_key["electrode"]],
         "channel_name": channel_map[ch_key["electrode"]], **ch_key}
        for ch_key in electrode_df
    ),
)
```

Where `_load_device_channel_map()` reads the probeinterface JSON from the epoch directory on Ceph and returns `{electrode_site_id: raw_channel_idx}` for all active contacts. The epoch directory is obtained via the Epoch table, and the JSON filename is found by the same discovery logic used in `create_electrode_config()`.

## Summary

Bugs 1-3 are all in `get_probe_id()`, which was merged to main via PR #548 but never tested against actual Neuropixels data. Bug 4 is in the upstream schema reader / ingestion pipeline. Bug 5 is in the guide script (our code). Bugs 6-8 are DJ 2.x migration issues in spike_sorting.py. Bug 9 is a split-raw-directory oversight in ephys.py. Bug 10 is a DJ 2.x + MariaDB framework incompatibility. Bugs 11-12 are SpikeInterface + HPC environment issues. Bug 13 is a guide script bug (our code) that caused garbled preprocessing by inserting all 5120 probe sites instead of 384 active channels. Bug 14 is a pipeline bug in ephys.py that assumes channel order matches electrode site order, which is only true for single-shank probes with contiguous electrode selection.

The common thread for bugs 1-5: none of this code was ever run against real data before it was merged. The metadata structure was assumed, the platform differences were assumed, the sync clock domains were assumed. Every assumption was wrong.

The common thread for bugs 6-10: spike_sorting.py was written for DJ 0.14.x and never updated for DJ 2.x, and the split-raw-directory feature was not propagated to all code paths.

The common thread for bugs 11-12: SpikeInterface's multiprocessing and file handling assumptions don't hold on the SWC HPC environment (SLURM cgroups, fork-unsafe BLAS, Ceph filesystem).

The common thread for bugs 13-14: the probeinterface channel map JSON was never properly parsed. Bug 13 ignored which contacts are active (`device_channel_indices`). Bug 14 ignored the actual hardware channel-to-electrode mapping, assuming site order = channel order.
