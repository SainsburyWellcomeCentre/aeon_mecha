# Ephys Pipeline Architecture

Design spec for the electrophysiology pipeline tables, ingestion flow, clock
alignment, and auxiliary ONIX streams (Bno055 IMU). Companion to
`SPEC_TESTING.md` (which covers the test infrastructure and golden-dataset integration tests).

**Status:** Reflects the schema restructure (symmetric epoch design,
per-epoch ElectrodeConfig resolution) and the ONIX IMU pipeline. Earlier
revisions described `EphysEpoch` as a child of `acquisition.Epoch` and a
`Bno055` lookup keyed on first-sample alignment — both are obsolete.

---

## Two independent acquisition systems

| Modality | Machine | Directory type | Master clock |
|---|---|---|---|
| Behavior | AEON3 | `raw` | HARP (rig is HARP-native) |
| Ephys | AEONX1 | `raw-ephys` | ONIX, synced to HARP via per-chunk `EphysSyncModel` |

The two rigs start and stop independently. Their epoch boundaries (directory
timestamps) do **not** coincide. Cross-modal alignment is post-hoc through
the shared HARP clock.

---

## Symmetric epoch design

`EphysEpoch` is a **peer** of `acquisition.Epoch` under
`acquisition.Experiment` — not a child. Each rig's epoch table has parallel
sub-tables.

```
                  acquisition.Experiment
            ┌──────────────┴──────────────┐
            ▼                              ▼
   acquisition.Epoch              ephys.EphysEpoch              ← PEERS
       (raw)                          (raw-ephys)
   ┌──── │ ────┐                  ┌──── │ ────┐
   ▼     ▼     ▼                  ▼     ▼     ▼
EpochConfig EpochEnd Chunk    EphysEpochConfig EphysEpochEnd EphysSyncModel
(Imported) (Manual) (Manual)  (Imported)       (Manual)      (Manual)
                                     │                              │
                                     │                  ┌───────────┴───────────┐
                                     ▼                  ▼                       ▼
                              ProbeInsertion       EphysChunk             OnixImuChunk
                              + per-probe            │                    (Imported)
                              ElectrodeConfig        ▼
                                                EphysBlock
                                                EphysBlockInfo
```

Symmetry: `Epoch → {EpochConfig, EpochEnd, Chunk}` mirrors
`EphysEpoch → {EphysEpochConfig, EphysEpochEnd, EphysSyncModel}`. Two
chunk-grain tables hang off `EphysSyncModel`: `EphysChunk` (probe binaries)
and `OnixImuChunk` (auxiliary Bno055 streams).

---

## Clock semantics

**`EphysEpoch.epoch_start` is HARP-native.**

It is **not** the wall-clock timestamp parsed from the raw-ephys directory
name. Instead, `EphysEpoch.ingest_epochs()` reads the **first row of the
first `*_HarpSync_*.csv`** in each epoch directory and uses its observed
`harp_start` field. That field is the HARP-clock instant at the start of
the recording.

| Field | Meaning | Clock |
|---|---|---|
| `EphysEpoch.epoch_dir` | Label only (e.g. `"2026-05-11T07-50-11"`) | ONIX wall-clock |
| `EphysEpoch.epoch_start` | The precise instant the recording began | **HARP** |
| `EphysSyncModel.sync_start` | Start of each chunk's sync window | HARP |
| `EphysChunk.chunk_start` | Start of each chunk | HARP |
| `EphysBlock.block_start` | Start of a block | HARP |
| `EphysBlock.block_end` | End of a block | HARP |

`epoch_dir` and `epoch_start` will differ by sub-second magnitude (ONIX-rig
wall-clock and HARP-clock are NTP-aligned but not bit-exact). This is
intended — `epoch_dir` is a filesystem label, `epoch_start` is the precise
timestamp used for downstream queries.

### Why HARP-native?

It eliminates the "mixed clock" ambiguity from the schema. Both
`acquisition.Epoch.epoch_start` (HARP-rig directory name, HARP-native by
construction) and `EphysEpoch.epoch_start` (first HarpSync CSV row,
HARP-native by observation) mean the same thing: HARP-clock instant of
acquisition start. Cross-modal time-range queries don't need clock
conversion at the epoch level.

### `EphysSyncModel` — per-chunk HARP↔ONIX regression

One row per `*_HarpSync_*.csv` (one per ONIX chunk window). HARP and ONIX
bounds are observed values from the CSV, not predicted — stable across
re-ingestion.

```python
@schema
class EphysSyncModel(dj.Manual):
    definition = """
    -> EphysEpoch
    sync_start: datetime(6)        # PK — observed harp_time[0]
    ---
    sync_end: datetime(6)          # observed harp_time[-1]
    onix_ts_start: bigint          # observed clock[0]
    onix_ts_end: bigint            # observed clock[-1]
    sync_model: <attach>           # joblib-serialized LinearRegression (onix→harp)
    r2: float
    n_samples: int
    unique index (experiment_name, epoch_start, onix_ts_start)
    """
```

Key design choices:

- **`sync_start` PK = observed `harp_time[0]`, not predicted.** Predicted
  values would shift across re-runs (sklearn version drift, FP precision)
  and break idempotency.
- **Both HARP and ONIX bounds stored.** HARP bounds answer "what time does
  this window cover" without loading the regression. ONIX bounds answer
  "which AmplifierData/Bno055 file overlaps" — the natural lookup axis.
- **Unique index on ONIX bounds.** Chunk-ingest queries against
  `onix_ts_start`/`onix_ts_end`; the index avoids full scans and catches
  duplicate ingestion of the same CSV.
- **Manual + classmethod, not Imported.** Discovery of new HarpSync CSVs
  requires a filesystem walk, which doesn't fit Imported's `key_source`
  (SQL-only). `EphysSyncModel.ingest(experiment_name)` is the re-runnable
  entry point.

### Sub-second alignment

For per-sample alignment between behavior video frames and ephys samples,
use `EphysSyncModel`'s per-chunk regression coefficients. Don't rely on
`epoch_start` precision below ~1 sec.

---

## Per-epoch ElectrodeConfig resolution

A single Neuropixels 2.0 probe is routinely re-banked across recordings
(different shanks, banks, depths). Each configuration becomes a separate
`ElectrodeConfig` row. Pre-restructure, `EphysChunk.ingest_chunks` couldn't
disambiguate when multiple configs existed for a `probe_type` — it errored.

**Resolution: per-(epoch, probe) FK on `EphysEpochConfig.Insertion`.**

Each `EphysEpochConfig.Insertion` row records, for one probe in one epoch,
both the `ElectrodeConfig` FK and the source JSON basename (`config_file_name`).
The mapping comes from `Metadata.yml`:

```yaml
Devices:
  NeuropixelsV2e:
    ConfigurationA: { ProbeInterfaceFileName: "Z:\...\recording_configurations\probeA.json" }
    ConfigurationB: { ProbeInterfaceFileName: "Z:\...\recording_configurations\probeB.json" }
```

`EphysEpochConfig.make()` reads `Metadata.yml` once per epoch, parses each
`ConfigurationA/B` (mapping `ProbeA/B` by convention), and:
- Calls `create_electrode_config(json_path, ...)` to dedup-insert the
  `ElectrodeConfig` (pure probe-geometry table — no filename or provenance).
- Records the JSON basename on the `Insertion` row alongside the FK
  (per-(epoch, probe) provenance lives there, not on the dedup'd config).

`ProbeInterfaceFileName: null` indicates a disabled/spoofed probe; the
ingest skips it. If every probe in an epoch is disabled/spoofed,
`EphysEpochConfig.make()` raises — there is no longer a sentinel row.

Downstream consumers (`EphysChunk.ingest_chunks`, `EphysBlockInfo.make`) read
the resolved `(probe_type, electrode_config_name, config_file_name)` directly
from `Insertion` — no further `Metadata.yml` parsing at chunk- or block-time.

### Where the JSON lives

| Layout | Location |
|---|---|
| Production | `<raw-ephys-root>/<rig>/recording_configurations/<name>.json` |
| Synthetic unit-test fixture | `tests/fixtures/ephys/synthetic_np2_multishank.json` |

`resolve_epoch_probe_json(raw_ephys_dir, epoch_path, basename)` searches the
production location first, then falls back to the epoch directory itself
(test/golden datasets typically have a local copy alongside raw data).
Raises `FileNotFoundError` if neither location has the file.

### `create_electrode_config` helper

`aeon/dj_pipeline/utils/ephys_utils.py:create_electrode_config(json_path, ...)`
is the canonical entry point. It:

1. Reads the probeinterface JSON.
2. Populates `ProbeType` + `ProbeType.Electrode` with the full contact
   geometry (e.g. 5120 contacts for NP 2.0 multishank).
3. Populates `ElectrodeConfig` + `ElectrodeConfig.Electrode` with the
   subset of contacts where `device_channel_indices != -1` (the actively-
   recorded electrodes, typically 384 per recording).
4. Sets `electrode_config_name = json_path.stem` (override via `config_name`).

The helper does NOT record the JSON basename on `ElectrodeConfig` — that
provenance belongs on the per-(epoch, probe) `EphysEpochConfig.Insertion`
row, set by the caller.

The helper is idempotent (all inserts use `skip_duplicates=True`). It does
NOT wrap its inserts in an explicit transaction, so it can be called from
inside an existing `populate()` transaction (e.g. `EphysEpochConfig.make`).

---

## Auxiliary ONIX streams (Bno055 IMU)

The Neuropixels acquisition rig writes auxiliary sensor streams alongside
the probe data, all sharing the ONIX hardware clock and the same per-chunk
`EphysSyncModel` regression. The Bno055 IMU (Euler, GravityVector,
LinearAcceleration, Quaternion) is the first such stream made queryable as
a DataJoint table.

### Schema

`OnixImuChunk` is keyed by `EphysSyncModel` — one row per HARP sync window.
Per-column summary stats live as JSON attributes on the master row; the full
DataFrame is reconstructed lazily via a custom codec.

```python
@schema
class OnixImuChunk(dj.Imported):
    definition = """
    -> EphysSyncModel
    ---
    sample_count: int
    timestamps: json                # min/max HARP, sampling rate, dt stats
    euler_x: json                   # per-column summary stats
    euler_y: json
    ...
    quaternion_z: json
    stream_df: <aeon_onix_stream>   # all 4 Bno055 streams, merged + filtered
    """
```

**Column-prefix convention:** strip `Bno055` from each stream class name and
snake_case. `Bno055Euler` → `euler_*`, `Bno055LinearAcceleration` →
`linear_acceleration_*`, etc. Summary-stat field names match `stream_df`
column names.

**No-IMU rigs:** when no Bno055 files exist (or no Bno055 chunks overlap
the sync window), `make()` inserts the row with `sample_count=0` and empty
stats. The `stream_df` reference is still valid; the codec returns an
empty DataFrame on fetch.

### Overlap-based Bno055 chunk selection

HarpSync CSVs (one per `EphysSyncModel` row, hourly cadence) and
`Bno055_Clock_N.bin` files (~10 min, firmware-flushed) partition the ONIX
clock on **independent** boundaries. Each sync window typically overlaps
several Bno055 files; a single Bno055 file may straddle two sync windows.

`find_overlapping_bno055_chunks(device_dir, device_name, onix_ts_start, onix_ts_end)`
returns the sorted list of `N`s whose `[first_sample, last_sample]` ONIX
range intersects `[onix_ts_start, onix_ts_end]`. Implementation reads only
the first and last uint64 sample of each Clock binary — O(1) I/O per file.

`OnixImuChunk.make` then loads + concatenates all overlapping chunks and
filters to the sync window's exact ONIX range before computing summary
stats. The `stream_df` reference records `chunk_indices` so the codec
reconstructs the same DataFrame on lazy fetch.

> **History.** An earlier implementation used an exact-match lookup
> (`locate_bno055_chunk_index`) assuming each sync window aligned 1:1 with
> one Bno055 file. That alignment never held on real data; the bug was
> masked by synthetic tests constructed with deliberately aligned
> fixtures. Replaced by the overlap-based selector above.

### Codec: `<aeon_onix_stream>`

Sibling to `<aeon_stream>` (`aeon/dj_pipeline/utils/codec.py`). The
existing codec handles Harp-style time-indexed binaries; Bno055 binaries
carry no embedded timestamps, so a separate codec handles ONIX-clock
reconstruction.

**Structural-only.** The codec performs file loads, column renames, and
concat on sample index. It does **not** apply the HARP sync regression
(that's `OnixImuChunk.synced_df`'s job — see below). This preserves parity
with `<aeon_stream>`'s contract (byte-deterministic given the file bytes)
and keeps the codec test surface narrow.

**Stored JSON shape** (one ref per OnixImuChunk row):

```json
{
  "experiment_name": "...",
  "epoch_start": "...",
  "sync_start": "...",
  "device_name": "NeuropixelsV2",
  "stream_group": "Bno055",
  "chunk_indices": [3, 4, 5],
  "onix_ts_start": 152811497944,
  "onix_ts_end": 1050486518185
}
```

**Decode**: load each `Bno055_Clock_N.bin` + sibling stream binaries for
the recorded `chunk_indices`, prefix-rename columns, concat on the ONIX
clock index, filter to `[onix_ts_start, onix_ts_end]`, validate column set
against `IMU_COLUMNS`, return an ONIX-indexed `DataFrame`.

**What the codec deliberately doesn't do:**

| Operation | Why excluded |
|---|---|
| Apply the regression model | Derivational, not raw. Sklearn version drift could change historical decode results silently. |
| Convert ONIX → HARP | Some downstream uses (aligning IMU with raw ONIX-clocked ephys events) need ONIX directly. |
| Download `sync_model` attach | Hidden expensive I/O on every fetch. The sync helper opts into this cost when needed. |

### `synced_df` — HARP-indexed access

```python
df_raw    = (OnixImuChunk & key).fetch1("stream_df")   # ONIX-indexed, no model fetch
df_synced = OnixImuChunk.synced_df(key)                # HARP-indexed, applies regression
```

`synced_df` is a thin `@classmethod` that fetches the ONIX-indexed
DataFrame via the codec, downloads the `sync_model` attach, and applies the
regression. One named place for sync logic — easy to find, test, mock.

---

## Ingestion order

For a fresh experiment with both behavior and ephys data:

```
1. acquisition.Experiment.insert1(...)
2. acquisition.Experiment.Directory.insert(...)   # raw + raw-ephys

3. acquisition.Epoch.ingest_epochs(exp)           # behavior epochs only
4. acquisition.EpochConfig.populate()
5. acquisition.Chunk + acquisition.EpochEnd       # from behavior side

6. ephys.EphysEpoch.ingest_epochs(exp)            # ephys epochs (HARP-native)
                                                    + EphysEpochEnd look-back
7. ephys.EphysEpochConfig.populate()              # probe discovery
                                                    + per-probe ElectrodeConfig
                                                    + ProbeInsertion creation

8. ephys.EphysSyncModel.ingest(exp)               # HarpSync CSVs → sync models
9. ephys.OnixImuChunk.populate()                  # auxiliary IMU streams

10. (Manual) ephys.EphysBlock.insert(...)         # define analysis blocks
11. ephys.EphysChunk.ingest_chunks(exp)           # probe binaries
12. ephys.EphysBlockInfo.populate()
13. spike_sorting.PreProcessing.populate()        # ... cascade
```

Behavior and ephys ingestion paths are independent — failures in one don't
block the other. Steps 8–9 are safe to re-run on a cadence (cron-friendly,
idempotent — each skips already-ingested keys).

### Why two different population mechanisms on the ephys side

| Table | Tier | Granularity | Cardinality vs SyncModel |
|---|---|---|---|
| `EphysSyncModel` | Manual + classmethod | per ONIX chunk window | self |
| `EphysChunk` | Manual + classmethod | HARP-aligned, per-probe binary | 1↔1 typical, 1↔N when straddling |
| `OnixImuChunk` | Imported (default key_source) | per sync window | 1↔1 |

`EphysChunk` keeps its 1↔N straddling semantics (one binary may span two
sync windows), which doesn't fit Imported's `key_source`. `OnixImuChunk`
is greenfield with strict 1↔1 to `EphysSyncModel`, so standard `Imported`
semantics work and no classmethod entrypoint is needed.

---

## Cross-modal alignment

There is **no pivot/join table** linking `acquisition.Epoch` to `EphysEpoch`.
Callers needing cross-modal queries use the shared HARP clock:

```python
# Behavior epoch covering a given ephys epoch
ephys_start = (EphysEpoch & key).fetch1("epoch_start")
behavior_epoch = (
    acquisition.Epoch
    & f'epoch_start <= "{ephys_start}"'
).fetch(order_by="epoch_start desc", limit=1, as_dict=True)
```

A helper function for this can be added later if it becomes common enough.

---

## Schema migration

The restructure is breaking. Production operators **drop and recreate** the
ephys schema, then re-ingest from raw data. No row-level migration script
is provided.

```python
import datajoint as dj
from aeon.dj_pipeline import ephys
ephys.schema.drop()        # confirms with the operator
# ... then run the ingestion order above ...
```

---

## File map

| File | Role |
|---|---|
| `aeon/dj_pipeline/ephys.py` | All ephys tables: `EphysEpoch`, `EphysEpochEnd`, `EphysEpochConfig`, `EphysSyncModel`, `EphysChunk`, `EphysBlock`, `EphysBlockInfo`, `OnixImuChunk`. |
| `aeon/dj_pipeline/utils/ephys_utils.py` | `create_electrode_config`, `parse_metadata_probe_configs`, `resolve_epoch_probe_json`, `load_device_channel_map`, `discover_epoch_probes`, ProbeInsertion helpers, HARP↔ONIX time helpers. |
| `aeon/dj_pipeline/utils/onix_imu.py` | `find_overlapping_bno055_chunks`, `load_and_merge_bno055`, `IMU_COLUMNS`. |
| `aeon/dj_pipeline/utils/codec.py` | `<aeon_stream>` and `<aeon_onix_stream>` codecs. |
| `aeon/dj_pipeline/acquisition.py` | `acquisition.Epoch.ingest_epochs()` — behavior side only (raw-ephys branch removed). |
| `tests/fixtures/ephys/synthetic_np2_multishank.json` | Synthetic 4-shank NP2.0 probeinterface JSON used by integration tests. |
