# Ephys Pipeline Architecture

Design spec for the electrophysiology pipeline tables, ingestion flow, and
clock semantics. Companion to `SPEC_EPHYS_TESTING.md` (which covers the
golden-dataset integration tests).

**Status:** Reflects the restructure from issues #583 (symmetric epoch design)
and #584 (per-epoch ElectrodeConfig resolution). Pre-restructure behavior is
no longer supported.

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
EpochConfig EpochEnd Chunk    EphysEpochConfig EphysEpochEnd EphysChunk
(Imported) (Manual) (Manual)  (Imported)       (Manual)      (Manual)
                                     │                            │
                                     ▼                            ▼
                              ProbeInsertion (Manual)      EphysSyncModel
                              + per-probe                   (Manual)
                              ElectrodeConfig                     │
                                                                  ▼
                                                            EphysBlock
                                                            EphysBlockInfo
                                                            OnixImuChunk
```

Symmetry: `Epoch → {EpochConfig, EpochEnd, Chunk}` mirrors
`EphysEpoch → {EphysEpochConfig, EphysEpochEnd, EphysChunk}`.

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
- Records the JSON basename on the `Insertion` row alongside the FK.

`ProbeInterfaceFileName: null` indicates a disabled/spoofed probe; the
ingest skips it.

Downstream consumers (`EphysChunk.ingest_chunks`, `EphysBlockInfo.make`) read
the resolved `(probe_type, electrode_config_name, config_file_name)` directly
from `Insertion` — no further `Metadata.yml` parsing at chunk- or block-time.

### Where the JSON lives

| Layout | Location |
|---|---|
| Production | `<raw-ephys-root>/<rig>/recording_configurations/<name>.json` |
| Golden test fixture | `tests/fixtures/ephys/M81_ProbeB_4Shanks_1000_to_1700_um.json` |

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

9. (Manual) ephys.EphysBlock.insert(...)          # define analysis blocks
10. ephys.EphysChunk.ingest_chunks(exp)           # per-epoch-per-probe config
                                                    resolution via Metadata.yml
11. ephys.EphysBlockInfo.populate()
12. spike_sorting.PreProcessing.populate()        # ... cascade
```

Behavior and ephys ingestion paths are independent — failures in one don't
block the other.

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

## File map (post-restructure)

| File | Role |
|---|---|
| `aeon/dj_pipeline/ephys.py` | All ephys tables: `EphysEpoch`, `EphysEpochEnd`, `EphysEpochConfig`, `EphysChunk`, `EphysSyncModel`, `EphysBlock`, `EphysBlockInfo`, `OnixImuChunk`. |
| `aeon/dj_pipeline/utils/ephys_utils.py` | `create_electrode_config`, `parse_metadata_probe_configs`, `resolve_epoch_probe_json`, `load_device_channel_map`, `discover_epoch_probes`, ProbeInsertion helpers, HARP↔ONIX time helpers. |
| `aeon/dj_pipeline/acquisition.py` | `acquisition.Epoch.ingest_epochs()` — behavior side ONLY (raw-ephys branch removed). |
| `aeon/dj_pipeline/scripts/ephys_v2_setup.py` | Manual setup script for synthetic-geometry probes (HPC-friendly). |
| `aeon/dj_pipeline/scripts/ephys_mock_ingestion.py` | Mock ingestion for `social-ephys0.1` dataset. Sets `config_file_name` placeholder. |
| `tests/fixtures/ephys/M81_ProbeB_4Shanks_1000_to_1700_um.json` | Per-epoch probeinterface JSON used by integration tests. |
