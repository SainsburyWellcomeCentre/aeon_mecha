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
EphysSyncModel  (Manual + ingest classmethod, cron)
       │
       ├─► EphysChunk  (Manual + ingest_chunks classmethod, cron)
       │       ├─► .File         (AmplifierData + Clock files)
       │       └─► .SyncModel    (Part — link table only)
       │
       └─► OnixImuChunk  (Imported, key_source = EphysSyncModel)
               └─► .Stream       (Part — 4 rows: Euler, GravityVector, LinearAcceleration, Quaternion)
```

### Why two different population mechanisms

| Table          | Pop. type                       | Granularity                        | Cardinality vs SyncModel |
|----------------|---------------------------------|------------------------------------|--------------------------|
| EphysSyncModel | Manual + classmethod            | Per ONIX chunk (~1hr, drifts)      | self                     |
| EphysChunk     | Manual + classmethod            | HARP-aligned 1hr window            | 1↔1 typically, 1↔2 when straddling |
| OnixImuChunk   | Imported, `key_source = SyncModel` | Per ONIX chunk (mirrors SyncModel) | 1↔1                      |

`EphysChunk` keeps its existing HARP 1hr-aligned semantic and the 1↔N link to SyncModel — it can't cleanly be a `dj.Imported` keyed off SyncModel because of that 1↔2 straddling case. `OnixImuChunk` is greenfield and rides the natural ONIX cadence 1:1, so standard `dj.Imported` semantics work and no separate cron entrypoint is needed.

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

```python
@schema
class OnixImuChunk(dj.Imported):
    definition = """
    -> EphysSyncModel
    ---
    sample_count: int32
    timestamps: json                    # min/max HARP, sampling rate, dt stats; {} if no data
    """

    class Stream(dj.Part):
        definition = """
        -> master
        stream_name: enum('Euler', 'GravityVector', 'LinearAcceleration', 'Quaternion')
        ---
        stream_df: <aeon_onix_stream>   # codec reference for lazy fetch
        """

    @property
    def key_source(self):
        return EphysSyncModel
```

**No-IMU rigs:** when `*_Bno055_Clock_*.bin` files are absent, `make()` inserts master with `sample_count=0` and skips the Stream Parts. Always populates the key — no infinite retries from `populate()`.

---

## Codec: `<aeon_onix_stream>`

Sibling to the existing `<aeon_stream>` codec (`aeon/dj_pipeline/utils/codec.py`). The existing codec assumes Harp-style time-indexed binaries (`io_api.load(start=, end=)` natively trims). Bno055 binaries carry no embedded timestamps, so a separate codec handles ONIX-clock alignment + sync regression.

**Stored JSON shape:**
```json
{
  "stream_type": "Bno055Euler",
  "experiment_name": "...",
  "epoch_start": "...",
  "sync_start": "...",
  "device_name": "NeuropixelsV2Beta",
  "stream_name": "Euler"
}
```

**Decode algorithm:**
1. Resolve raw_dir from experiment.
2. Locate the ONIX device dir within the epoch (`NeuropixelsV2Beta/` or `NeuropixelsV2/`).
3. Identify the chunk index N by matching `EphysSyncModel.onix_ts_start` to the corresponding `Bno055_Clock_N.bin`.
4. Load `Bno055_Clock_N.bin` (uint64 ONIX timestamps) via `social_ephys.<device>.Bno055.Bno055Clock` reader.
5. Load `Bno055_<stream_name>_N.bin` (float32 payload) via the matching `social_ephys.<device>.Bno055.Bno055<stream_name>` reader. Sample-align by index with the clock array.
6. Download the `EphysSyncModel.sync_model` attach file, `joblib.load`, apply regression to convert ONIX → HARP datetimes per sample.
7. Return a HARP-time-indexed `pd.DataFrame`.

The codec is reusable for any future ONIX-clocked accessory (other IMUs, photodiodes, etc.).

---

## Ingestion flow

### 1. `EphysSyncModel.ingest(experiment_name)` — cron

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

### 2. `EphysChunk.ingest_chunks(experiment_name)` — cron, thinned

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

Triggered automatically by the same cron tick after `EphysSyncModel.ingest()`. No separate filesystem walk.

```python
def make(self, key):
    # key has experiment_name, epoch_start, sync_start
    sm = (EphysSyncModel & key).fetch1()
    epoch_path = resolve_epoch_path(key)
    device_name = discover_device(epoch_path)        # NeuropixelsV2Beta or V2
    device_dir = epoch_path / device_name

    chunk_index = locate_bno055_index(device_dir, sm)  # match to onix_ts_start

    clock_files = list(device_dir.glob(f"{device_name}_Bno055_Clock_*.bin"))
    if chunk_index is None or not clock_files:
        # No-IMU rig
        self.insert1({**key, "sample_count": 0, "timestamps": {}})
        return

    clock_file = device_dir / f"{device_name}_Bno055_Clock_{chunk_index}.bin"
    onix_clock = np.fromfile(clock_file, dtype=np.uint64)

    # Apply regression to get HARP per sample
    model = joblib.load(sm["sync_model"])
    harp_per_sample = model.predict(onix_clock.reshape(-1, 1)).flatten()

    self.insert1({
        **key,
        "sample_count": len(onix_clock),
        "timestamps": timestamp_summary(harp_per_sample),
    })

    self.Stream.insert([
        {
            **key,
            "stream_name": stream,
            "stream_df": {
                "stream_type": f"Bno055{stream}",
                "experiment_name": key["experiment_name"],
                "epoch_start": str(key["epoch_start"]),
                "sync_start": str(key["sync_start"]),
                "device_name": device_name,
                "stream_name": stream,
            },
        }
        for stream in ("Euler", "GravityVector", "LinearAcceleration", "Quaternion")
        if (device_dir / f"{device_name}_Bno055_{stream}_{chunk_index}.bin").exists()
    ])
```

---

## Cron ordering

```
1. acquisition.Epoch.populate                   (existing acquisition_worker)
2. EphysEpoch.populate                          (one-shot per new epoch — pre-existing wrinkle, see below)
3. EphysSyncModel.ingest(experiment_name)       (every tick, ~4hr)
4. EphysChunk.ingest_chunks(experiment_name)    (every tick, depends on EphysSyncModel)
5. OnixImuChunk.populate()                      (every tick, depends on EphysSyncModel via key_source)
```

Likely fits in the existing `streams_worker`, or warrants a new `ephys_worker` if the cadence differs from streams ingestion.

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
2. **Summary stats granularity for `OnixImuChunk`.** Per-stream summaries (mean/std/range) like the existing `<aeon_stream>` tables, or just sample count + time bounds? Lean toward minimal for v1.
3. **Worker placement.** `streams_worker` vs. new `ephys_worker`. Driven by cadence and resource isolation requirements.

---

## Concrete change set (preview)

| File | Change |
|------|--------|
| `aeon/schema/ephys.py` | Extend `HarpSyncModel.Reader.read()` to return `harp_start`, `harp_end`, `n_samples` |
| `aeon/dj_pipeline/ephys.py` | Add `EphysSyncModel`. Refactor `EphysChunk.SyncModel` to link Part. Thin `ingest_chunks`. Add `OnixImuChunk` + `OnixImuChunk.Stream`. |
| `aeon/dj_pipeline/utils/codec.py` | Add `OnixStreamCodec` (registered as `<aeon_onix_stream>`) |
| `aeon/dj_pipeline/utils/ephys_utils.py` | Drop `process_ephys_file` (logic moves into `EphysSyncModel.ingest` and slimmer `ingest_chunks`); add helpers for ONIX↔SyncModel index resolution |
| `aeon/dj_pipeline/__init__.py` | Register `OnixStreamCodec` alongside `AeonStreamCodec` |
| `aeon/dj_pipeline/populate/worker.py` | Wire `EphysSyncModel.ingest`, `EphysChunk.ingest_chunks`, `OnixImuChunk.populate` into the chosen worker |
| `tests/dj_pipeline/...` | Unit tests for codec encode/decode; integration test for the three-table cron sequence using a golden ephys dataset |
| Migration script | One-time conversion of existing `EphysChunk.SyncModel` data to `EphysSyncModel` |
