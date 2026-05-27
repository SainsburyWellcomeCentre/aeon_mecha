# ONIX IMU Pipeline & Ephys Sync-Model Refactor

Design spec for ingesting ONIX-clocked auxiliary data streams (initially Bno055 IMU) alongside the existing Neuropixels ephys pipeline, with a supporting refactor that promotes the per-chunk HARP↔ONIX sync model from a `EphysChunk` Part to a first-class table.

**Status:** design

**Branch:** `tn/onix-imu-pipeline`

---

## Motivation

The Neuropixels acquisition rig writes additional sensor streams alongside the probe data, all sharing the ONIX hardware clock and the same per-chunk `HarpSync_*.csv` regression to map ONIX→HARP. Today only the AmplifierData binaries flow into DataJoint; the auxiliary Bno055 IMU streams (Euler, GravityVector, LinearAcceleration, Quaternion) sit on disk unused.

The `aeon/schema/ephys.py` reader layer already covers the auxiliary streams (added incrementally past upstream commit `17dec14`). What's missing is:

1. A DataJoint table to make Bno055 data queryable.
2. A clean home for the per-chunk `LinearRegression` model so multiple downstream tables (ephys + IMU + future ONIX accessories) can share it without duplication.

This spec defines (1) by way of (2).

---

## Current state (pre-refactor)

```
EphysEpoch                 # Imported, one-shot per acquisition.Epoch
  .Insertion               # Part: which probes are active in this epoch

EphysChunk                 # Manual, populated by ingest_chunks(experiment_name)
  .File                    # Part: AmplifierData + Clock binaries
  .SyncModel               # Part: per-chunk HARP↔ONIX regression (attach + onix bounds)

EphysBlock / EphysBlockInfo  # User-defined HARP windows over chunks
```

`EphysChunk.ingest_chunks` (`aeon/dj_pipeline/ephys.py:241-372`) walks `*_AmplifierData*.bin`, lazily fits one `LinearRegression` per `HarpSync_*.csv` (cached in-memory per parent dir), predicts HARP `chunk_start`/`chunk_end` from the ONIX clock file, and inserts master + File + SyncModel rows. The fitted model is `joblib`-dumped into the SyncModel Part's `attach` field.

This is correct but couples four concerns into one ingestion path:
1. Filesystem discovery of new chunks.
2. Sync-model fitting (CPU work).
3. Sync-model persistence (disk + DB).
4. Per-probe chunk creation.

It also makes the sync model logically per-probe (because it's a Part of `EphysChunk`, which is keyed by `ProbeInsertion`) when in reality the model is per-device — both probes on the same rig share the same regression.

---

## Design goals

1. **Sync model is reusable** — both Neuropixels probes and Bno055 share one regression per chunk window, computed once.
2. **HARP is the master clock** — every chunk-level table exposes HARP `chunk_start`/`chunk_end` as its primary time reference.
3. **Cron-friendly** — ingestion paths must be re-runnable on an interval (~4hr) for ongoing epochs.
4. **Codec-style lazy fetch** — IMU data lives in raw files; DataJoint stores summary stats + a self-describing reference. Decode reconstructs the DataFrame on fetch.
5. **No new dotmap surface migration in this work** — Bno055 readers stay in `aeon/schema/ephys.py` (dotmap `social_ephys`). Pydantic migration of ephys streams is explicitly out of scope.

---

## Architecture

```
acquisition.Epoch
       │
       ▼
EphysEpoch  ──────────────► .Insertion         (one-shot Imported per epoch)
       │
       ▼
EphysSyncModel  (Manual + ingest classmethod, re-runnable)
       │
       ├─► EphysChunk  (Manual + ingest_chunks classmethod, re-runnable)
       │       ├─► .File         (AmplifierData + Clock files)
       │       └─► .SyncModel    (Part — link table only)
       │
       └─► OnixImuChunk  (Imported, default key_source = EphysSyncModel)
                                 — single row per sync window; all 4 Bno055 streams merged on sample index
```

### Why two different population mechanisms

| Table          | Pop. type                       | Granularity                        | Cardinality vs SyncModel |
|----------------|---------------------------------|------------------------------------|--------------------------|
| EphysSyncModel | Manual + classmethod            | Per ONIX chunk (~1hr, drifts)      | self                     |
| EphysChunk     | Manual + classmethod            | HARP-aligned 1hr window            | 1↔1 typically, 1↔2 when straddling |
| OnixImuChunk   | Imported (default `key_source`) | Per ONIX chunk (mirrors SyncModel) | 1↔1                      |

`EphysChunk` keeps its existing HARP 1hr-aligned semantic and the 1↔N link to SyncModel — it can't cleanly be a `dj.Imported` keyed off SyncModel because of that 1↔2 straddling case. `OnixImuChunk` is greenfield and rides the natural ONIX cadence 1:1, so standard `dj.Imported` semantics work and it doesn't need its own classmethod entrypoint.

---

## Schema

### EphysSyncModel (new)

```python
@schema
class EphysSyncModel(dj.Manual):
    definition = """
    -> EphysEpoch
    sync_start: datetime(6)            # PK — observed harp_time[0] from CSV (master clock)
    ---
    sync_end: datetime(6)              # observed harp_time[-1] from CSV
    onix_ts_start: bigint              # observed clock[0] from CSV
    onix_ts_end: bigint                # observed clock[-1] from CSV
    sync_model: attach                 # joblib-serialized LinearRegression (onix→harp)
    r2: float                          # regression fit quality
    n_samples: int                     # rows in CSV after dropna()
    unique index (experiment_name, epoch_start, onix_ts_start)
    """
```

**Key design decisions:**

- **`sync_start` PK = observed `harp_time[0]`, not predicted.** Predicted values would shift across re-runs (sklearn version drift, FP precision) and break idempotency. The CSV's first sync sample is raw observed data, stable across runs.
- **Both HARP and ONIX bounds stored as observed values.** HARP bounds answer "what time does this window cover" without loading the model. ONIX bounds answer "which AmplifierData/Bno055 file matches" — the natural lookup axis.
- **Unique index on ONIX bounds.** AmplifierData→SyncModel lookup queries against `onix_ts_start`/`onix_ts_end`; without an index, every chunk ingestion does a full scan. The unique constraint also catches duplicate ingestion of the same CSV.
- **Why Manual + classmethod, not Imported:** `key_source` is a SQL expression and can't enumerate filesystem contents. Cron-driven discovery of new HarpSync CSVs requires a file walk, which fits the Manual + `ingest()` pattern.

### EphysChunk (refactored)

```python
@schema
class EphysChunk(dj.Manual):
    definition = """
    -> ProbeInsertion
    chunk_start: datetime(6)
    ---
    -> EphysEpoch
    chunk_end: datetime(6)
    -> ElectrodeConfig
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(128)
        ---
        -> acquisition.Experiment.Directory
        file_path: varchar(255)
        """

    class SyncModel(dj.Part):           # link-only, no attach field
        definition = """
        -> master
        -> EphysSyncModel
        """
```

PK shape `(experiment_name, subject, insertion_number, chunk_start)` is preserved from the existing schema for downstream compatibility (`EphysBlock`, `EphysBlockInfo`). The `attach` field that previously stored the per-chunk model bytes is gone — `SyncModel` is now a pure link Part. Multiple link rows handle the 1↔2 straddling case.

### OnixImuChunk (new)

The four Bno055 streams (Euler, GravityVector, LinearAcceleration, Quaternion) are sample-aligned by construction — they all index off the same `Bno055_Clock_N.bin`. They naturally collapse into a single DataFrame indexed by HARP time, with columns prefixed by their source stream (`euler_x`, `gravity_vector_y`, `quaternion_w`, etc.). One row per sync window, one codec reference, one fetch.

Per-column summary stats live as JSON attributes on the master row — same pattern as `streams_maker`-generated stream tables (`utils/streams_maker.py:183-197`).

```python
@schema
class OnixImuChunk(dj.Imported):
    definition = """
    -> EphysSyncModel
    ---
    sample_count: int32
    timestamps: json                    # min/max HARP, sampling rate, dt stats; {} if no data
    euler_x: json                       # per-column summary stats (min/max/mean/etc.)
    euler_y: json
    euler_z: json
    gravity_vector_x: json
    gravity_vector_y: json
    gravity_vector_z: json
    linear_acceleration_x: json
    linear_acceleration_y: json
    linear_acceleration_z: json
    quaternion_w: json
    quaternion_x: json
    quaternion_y: json
    quaternion_z: json
    stream_df: <aeon_onix_stream>       # all 4 Bno055 streams merged on sample index
    """
```

**Column-prefix convention:** strip `Bno055` from each Stream class name, snake_case the remainder. `Bno055Euler` → `euler_*`, `Bno055GravityVector` → `gravity_vector_*`, `Bno055LinearAcceleration` → `linear_acceleration_*`, `Bno055Quaternion` → `quaternion_*`. The codec applies the same renaming on decode so summary-column names match `stream_df` columns.

**No-IMU rigs:** when `*_Bno055_Clock_*.bin` files are absent, `make()` inserts master with `sample_count=0`, empty `{}` for `timestamps` and every per-column stats field, and a `stream_df` reference that the codec resolves to an empty DataFrame. Always populates the key — no infinite retries from `populate()`.

**`key_source`:** default DJ behavior (cross-product of parent table PKs) already resolves to `EphysSyncModel`. No override needed.

**Sync helper — `synced_df` classmethod**

The codec returns ONIX-clock-indexed data (see `<aeon_onix_stream>` section). For HARP-indexed data, users call:

```python
@classmethod
def synced_df(cls, key) -> pd.DataFrame:
    """Fetch stream_df for a single chunk and apply its HARP sync regression.

    Args:
        key: A complete OnixImuChunk primary key dict — must resolve to
             exactly one row. PK fields: experiment_name, epoch_start,
             sync_start. (`fetch1()` raises if the key resolves to zero
             or multiple rows.)

    Returns:
        HARP-time-indexed DataFrame. Downloads the EphysSyncModel attach
        once (one MySQL round-trip), loads the joblib LinearRegression,
        and applies it to the ONIX index.

    For raw ONIX-clock-indexed data, fetch ``stream_df`` directly
    instead of using this helper.
    """
    df = (cls & key).fetch1("stream_df")                      # ONIX-indexed
    sync_attach = (EphysSyncModel & key).fetch1("sync_model")
    model = joblib.load(sync_attach)
    harp_seconds = model.predict(df.index.values.reshape(-1, 1)).flatten()
    df.index = io_api.to_datetime(harp_seconds)
    return df
```

Two-line user pattern (where `key` is a complete OnixImuChunk PK dict):
```python
df_raw    = (OnixImuChunk & key).fetch1("stream_df")    # ONIX-indexed, no sync model fetch
df_synced = OnixImuChunk.synced_df(key)                 # HARP-indexed, applies regression
```

Sync logic lives in this one named place — easy to find, test, mock. The codec stays narrow.

---

## Codec: `<aeon_onix_stream>`

Sibling to the existing `<aeon_stream>` codec (`aeon/dj_pipeline/utils/codec.py`). The existing codec assumes Harp-style time-indexed binaries (`io_api.load(start=, end=)` natively trims). Bno055 binaries carry no embedded timestamps, so a separate codec handles ONIX-clock alignment + sync regression.

**Design choice — structural-only.** The codec performs raw-data and structural operations only: file loads, column renames, concat on sample index. It does **not** apply the HARP sync regression. This preserves parity with the existing `<aeon_stream>` codec contract (lazy load of raw data, byte-deterministic given the file bytes) and keeps the codec test surface narrow. Sync application is exposed as an explicit `OnixImuChunk.synced_df` classmethod (defined in the OnixImuChunk schema section) — users opt in when they want HARP-indexed data.

**Stored JSON shape** (one ref per OnixImuChunk row, covers all 4 streams):
```json
{
  "experiment_name": "...",
  "epoch_start": "...",
  "sync_start": "...",
  "device_name": "NeuropixelsV2Beta",
  "stream_group": "Bno055"
}
```

**Decode algorithm:**
1. Resolve raw_dir from experiment.
2. Locate the ONIX device dir within the epoch (`NeuropixelsV2Beta/` or `NeuropixelsV2/`).
3. Identify the chunk index N by matching `EphysSyncModel.onix_ts_start` to the corresponding `Bno055_Clock_N.bin`.
4. Load `Bno055_Clock_N.bin` (uint64 ONIX timestamps) via `social_ephys.<device>.<stream_group>.Bno055Clock` reader.
5. For each non-clock Stream class on the named StreamGroup (`Bno055Euler`, `Bno055GravityVector`, `Bno055LinearAcceleration`, `Bno055Quaternion`):
   - Load `Bno055_<StreamName>_N.bin` (float32 payload) via the matching reader.
   - Rename its columns by prefixing with the snake_case stream name minus the `Bno055` prefix (e.g., `Bno055Euler`'s `x/y/z` → `euler_x/euler_y/euler_z`).
6. Concat all stream DataFrames column-wise, using the ONIX clock array as the index.
7. Validate merged columns against `IMU_COLUMNS` (set equality); raise `ValueError` on mismatch. Reorder to canonical order.
8. Return an **ONIX-clock-indexed** `pd.DataFrame` (13 columns for Bno055; index dtype `uint64`).

The `stream_group` knob makes the codec reusable for any future ONIX-clocked accessory whose dotmap StreamGroup follows the same `<Group>Clock` + sibling-stream-classes convention.

**What the codec deliberately does not do:**

| Operation | Why excluded |
|-----------|--------------|
| Apply the regression model | Derivational, not raw. Sklearn version drift could change historical decode results silently. Failure surface (model attach missing, version mismatch, extrapolation NaNs) shouldn't sit on every fetch. |
| Convert ONIX → HARP timestamps | Same as above. Some downstream uses (aligning IMU samples with raw ONIX-clocked ephys events) need ONIX directly. |
| Download `sync_model` attach | Hidden expensive I/O on every fetch. Sync helper opts into this cost when needed. |

---

## Ingestion flow

### 1. `EphysSyncModel.ingest(experiment_name)` — re-runnable classmethod

Replaces the in-memory `sync_models = {}` cache and the `process_ephys_file` model-fitting half.

```python
@classmethod
def ingest(cls, experiment_name):
    raw_dir = acquisition.Experiment.get_data_directory({...}, "raw")
    epoch_dir_to_start = build_epoch_lookup(experiment_name)

    for csv in sorted(raw_dir.rglob("*_HarpSync_*.csv")):
        epoch_start = resolve_epoch(csv, epoch_dir_to_start)
        if epoch_start is None:
            continue

        device_name = parse_device_from_path(csv)        # NeuropixelsV2Beta or V2
        device_reader = getattr(social_ephys, device_name, None)
        if device_reader is None:
            continue

        # Optional: skip CSVs being actively written (recent mtime)

        # Read CSV; HarpSyncModel.Reader.read() extended to also return harp_start/harp_end
        df_row = device_reader.HarpSyncModel.Reader(...).read(csv)
        sync_start = io_api.to_datetime(df_row['harp_start'])

        if cls & {"experiment_name": experiment_name,
                  "epoch_start": epoch_start, "sync_start": sync_start}:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / f"{csv.stem}.joblib"
            joblib.dump(df_row['model'], model_path)
            cls.insert1({
                "experiment_name": experiment_name,
                "epoch_start": epoch_start,
                "sync_start": sync_start,
                "sync_end": io_api.to_datetime(df_row['harp_end']),
                "onix_ts_start": int(df_row['clock_start']),
                "onix_ts_end": int(df_row['clock_end']),
                "sync_model": model_path,
                "r2": float(df_row['r2']),
                "n_samples": int(df_row['n_samples']),
            })
```

**Required upstream change:** extend `HarpSyncModel.Reader.read()` in `aeon/schema/ephys.py:35-58` to return `harp_start`, `harp_end`, `n_samples` in addition to the existing `clock_start`/`clock_end`/`model`/`r2`. Backward-compatible (additive).

### 2. `EphysChunk.ingest_chunks(experiment_name)` — re-runnable, thinned

The existing classmethod sheds responsibility for sync-model fitting and persistence. It still walks AmplifierData files (probe-specific files not covered by SyncModel), but the sync work becomes a DB query.

```python
@classmethod
def ingest_chunks(cls, experiment_name):
    raw_dir, exp_key, epoch_dir_to_start, insertion_lookup = ...

    for ephys_file in sorted(raw_dir.rglob("*_AmplifierData*.bin")):
        rel_path = ephys_file.relative_to(raw_dir).as_posix()
        if cls.File & exp_key & {"file_path": rel_path}:
            continue

        probe_label, epoch_start = parse_path(ephys_file, epoch_dir_to_start)
        insertion_key = insertion_lookup.get((epoch_start, probe_label))
        if insertion_key is None:
            continue

        clock_file = ephys_file.with_name(
            ephys_file.name.replace("AmplifierData", "Clock"))
        onix_ts = np.memmap(clock_file, mode="r", dtype=np.uint64)

        # Query SyncModel by ONIX bounds (replaces in-memory cache)
        matched = (EphysSyncModel
                   & {"experiment_name": experiment_name, "epoch_start": epoch_start}
                   & f"({int(onix_ts[0])} BETWEEN onix_ts_start AND onix_ts_end) "
                     f"OR ({int(onix_ts[-1])} BETWEEN onix_ts_start AND onix_ts_end)"
                   ).to_dicts(order_by="sync_start")

        if not matched:
            logger.warning(f"No SyncModel covers {ephys_file.name}; skipping")
            continue

        chunk_start = resolve_harp(matched[0],  onix_ts[0])    # see "fast/slow path"
        chunk_end   = resolve_harp(matched[-1], onix_ts[-1])

        electrode_config_name = resolve_econfig(insertion_key)

        chunk_entry = {
            **insertion_key,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "epoch_start": epoch_start,
            "electrode_config_name": electrode_config_name,
        }
        cls.insert1(chunk_entry)
        cls.File.insert([file_row(f, chunk_entry) for f in (ephys_file, clock_file)])
        cls.SyncModel.insert([
            {**chunk_entry, "sync_start": m["sync_start"]} for m in matched
        ])
```

#### Fast vs. slow `resolve_harp`

- **Fast path:** if `onix_ts[0] == matched[0]['onix_ts_start']`, `chunk_start = matched[0]['sync_start']` directly — no model attach download. Symmetric for `chunk_end`.
- **Slow path:** ONIX times don't align exactly with SyncModel bounds → download the attached model, `joblib.load`, predict.

The fast path is the common case once SyncModel boundaries are derived from observed CSV samples (because `onix_ts_start` IS the first sample's clock value). Worth implementing.

### 3. `OnixImuChunk.populate()` — standard `dj.Imported`

Standard `dj.Imported.populate()` — runs after `EphysSyncModel.ingest()` has produced new sync rows, picks up new keys automatically. No separate filesystem walk, no override of `key_source`.

```python
IMU_COLUMNS = (
    "euler_x", "euler_y", "euler_z",
    "gravity_vector_x", "gravity_vector_y", "gravity_vector_z",
    "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
)

def make(self, key):
    # key has experiment_name, epoch_start, sync_start
    sm = (EphysSyncModel & key).fetch1()
    epoch_path = resolve_epoch_path(key)
    device_name = discover_device(epoch_path)        # NeuropixelsV2Beta or V2
    device_dir = epoch_path / device_name

    chunk_index = locate_bno055_index(device_dir, sm)  # match to sm.onix_ts_start

    stream_df_ref = {
        "experiment_name": key["experiment_name"],
        "epoch_start": str(key["epoch_start"]),
        "sync_start": str(key["sync_start"]),
        "device_name": device_name,
        "stream_group": "Bno055",
    }

    if chunk_index is None or not (device_dir / f"{device_name}_Bno055_Clock_{chunk_index}.bin").exists():
        # No-IMU rig — insert empty row, codec returns empty DF on fetch
        self.insert1({
            **key,
            "sample_count": 0,
            "timestamps": {},
            **{col: {} for col in IMU_COLUMNS},
            "stream_df": stream_df_ref,
        })
        return

    # Load + merge all Bno055 streams (ONIX-clock-indexed) — same logic the codec uses on decode
    df = load_and_merge_bno055(device_dir, device_name, chunk_index)

    # For DB summary fields only: apply sync regression to compute HARP timestamps.
    # The stored stream_df ref points to the ONIX-indexed DataFrame; sync is applied
    # only when users call OnixImuChunk.synced_df() — not on every codec fetch.
    model = joblib.load(sm["sync_model"])
    harp_index = io_api.to_datetime(model.predict(df.index.values.reshape(-1, 1)).flatten())

    self.insert1({
        **key,
        "sample_count": len(df),
        "timestamps": timestamp_summary(harp_index),                   # HARP-domain stats
        **{col: column_stats(df[col].values) for col in IMU_COLUMNS},  # sync-agnostic
        "stream_df": stream_df_ref,
    })
```

`load_and_merge_bno055` is shared between `make()` (for column stats) and the codec's `decode()` (for lazy fetch) — single source of truth for column naming + merge logic. Both produce ONIX-indexed output; HARP conversion happens in two distinct places: once in `make()` to populate the `timestamps` summary field, and once on demand in `synced_df` for user fetches.

**Column validation.** Before returning, `load_and_merge_bno055` validates that the merged DataFrame's columns match `IMU_COLUMNS` exactly (set equality), then reorders to canonical order:

```python
def load_and_merge_bno055(device_dir, device_name, chunk_index):
    # ... load Clock + 4 Bno055 binaries, prefix-rename, concat on ONIX index ...
    if set(df.columns) != set(IMU_COLUMNS):
        raise ValueError(
            f"Bno055 stream column mismatch: expected {IMU_COLUMNS}, "
            f"got {tuple(df.columns)}. Schema in aeon/schema/ephys.py "
            f"may have drifted — update OnixImuChunk + IMU_COLUMNS to match, "
            f"or revert the schema change."
        )
    return df[list(IMU_COLUMNS)]   # canonical column order
```

Because both `make()` (ingestion) and the codec's `decode()` (fetch) route through this function, validation runs at both times. Catches schema drift at ingestion (before bad data lands in DB) AND at fetch time (drift between historical data and a later schema change).

---

## Ingestion ordering

When invoked on a recurring schedule (every ~4hr) for ongoing epochs:

```
1. acquisition.Epoch.populate
2. EphysEpoch.populate                          (one-shot per new epoch — pre-existing wrinkle, see below)
3. EphysSyncModel.ingest(experiment_name)       (re-runnable, idempotent)
4. EphysChunk.ingest_chunks(experiment_name)    (re-runnable, depends on EphysSyncModel rows)
5. OnixImuChunk.populate()                      (standard DJ populate, depends on EphysSyncModel via key_source)
```

Steps 3–5 are safe to call at any cadence — each is idempotent and skips already-ingested keys.

---

## Out of scope

These are flagged for awareness but not part of this work:

- **Pydantic migration of ephys streams.** Bno055 stays in dotmap (`aeon/schema/ephys.py` → `social_ephys`). The dotmap holdout is a known item in `CLAUDE.md`; deferring to a separate effort.
- **Promoting `EphysChunk` to `dj.Imported`.** Considered, rejected for this work due to the 1↔N relationship with SyncModel that doesn't fit standard `key_source` semantics.
- **`EphysEpoch` re-population for late-arriving probes.** `EphysEpoch.make()` runs once per `acquisition.Epoch` and locks `has_ephys=False` if no probes are visible at that moment. For epochs where probes appear late, this currently mis-records. Pre-existing issue, separate ticket.
- **`read_probe_assignments` implementation.** Currently `NotImplementedError` (`utils/ephys_utils.py:126-150`). Independent track.
- **Migration of existing `EphysChunk.SyncModel` rows.** When this lands, existing data needs a one-time migration: copy attach-stored models into the new `EphysSyncModel` table, replace Part rows with link rows. Migration script TBD.

---

## Open questions

1. **In-flight CSV handling.** Should `EphysSyncModel.ingest()` skip CSVs with recent mtime to avoid ingesting partial files? Current pre-refactor `ingest_chunks` doesn't address this; could be deferred.
2. **Per-column stats helper.** `OnixImuChunk` reuses the auto-stream-table pattern of one JSON column per data column. The `column_stats()` helper from `utils/stats.py` should work as-is; verify shape during implementation.

---

## Concrete change set (preview)

| File | Change |
|------|--------|
| `aeon/schema/ephys.py` | Extend `HarpSyncModel.Reader.read()` to return `harp_start`, `harp_end`, `n_samples` |
| `aeon/dj_pipeline/ephys.py` | Add `EphysSyncModel`. Refactor `EphysChunk.SyncModel` to link Part. Thin `ingest_chunks`. Add `OnixImuChunk` (single table, no Parts). |
| `aeon/dj_pipeline/utils/codec.py` | Add `OnixStreamCodec` (registered as `<aeon_onix_stream>`) |
| `aeon/dj_pipeline/utils/ephys_utils.py` | Drop `process_ephys_file` (logic moves into `EphysSyncModel.ingest` and slimmer `ingest_chunks`); add helpers for ONIX↔SyncModel index resolution |
| `aeon/dj_pipeline/__init__.py` | Register `OnixStreamCodec` alongside `AeonStreamCodec` |
| `tests/dj_pipeline/...` | Unit tests for codec encode/decode; integration test for the three-table ingestion sequence using a golden ephys dataset |
| Migration script | One-time conversion of existing `EphysChunk.SyncModel` data to `EphysSyncModel` |
