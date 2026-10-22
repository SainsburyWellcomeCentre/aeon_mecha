# SPEC: Arrow-format ephys ingestion

Status: approved 2026-10-21
Issue: [#634](https://github.com/SainsburyWellcomeCentre/aeon_mecha/issues/634)
Base: `main` + #626 (selectable preprocessing), #629 (SyncedSpikes chunk boundaries), #630 (`read_bonsai_onix_arrow`)

## Intent

Bonsai's Aeon acquisition changed how it writes ephys data. Probe, clock and IMU
streams are now Arrow IPC files, compressed on the fly, with several streams sharing
one file. `aeon_mecha` cannot ingest them: discovery globs `*_AmplifierData*.bin` and
silently finds nothing.

This work makes the ephys arm ingest **both** formats. "Works" means the ephys data
ingests correctly — sync clock, ephys epoch, ephys chunks and their associated files —
and reaches `PreProcessing`, i.e. arrow data becomes a usable SpikeInterface recording.
Everything below `PreProcessing` is format-agnostic (§6) and stays covered by
`abcGolden01`.

**Minor version, additive.** Both formats supported indefinitely; nothing is removed.

## Goals

- Ingest `golden_arrow_01` (arrow) and `abcGolden01` (binary) through the same pipeline,
  up to and including `PreProcessing`.
- Make `Metadata.json` the authority for probe identity, enablement and subject mapping,
  replacing inference.
- Keep both golden datasets in the test suite, selectable with `pytest -k`.

## Non-goals

- Behaviour-arm changes. Its `Metadata.json` is structurally unchanged (same top-level
  keys including `rig`); the new `RigConfiguration.yaml` is an additional file nothing reads.
- Removing `probe_assignments.json`. It remains the path for old-format epochs until a
  separate, deliberate deprecation decision.
- Reworking the zarr/`aeon_raw_compression` path for *binary* data. It stays as is.
  **Confirmed redundant for arrow**, which is compressed at acquisition: arrow files take
  their own branch in `make_compute` and never consult `resolve_ephys_file`, which keeps
  its `.bin` -> `.zarr` preference for binary only. No arrow file acquires a *raw*
  compression twin.

  **This is not the same artifact as `PreProcessing`'s `recording.zarr`.** That one is the
  *preprocessed* intermediate the sorter consumes, and it is written for **both** formats.
  Do not remove it for arrow on the strength of the paragraph above.
- A major version. Both formats work, so this is additive, not breaking.
- **Implementing Commutator ingestion.** Deferred: every `Turns` sample is zero in both
  golden datasets, so there is nothing to validate against. The design thinking is kept
  in §6 so it does not have to be redone.

## Anti-goals

Things that would make this a failure even if it technically ingests:

- **Silent wrong data.** Every format branch must fail loudly on an unrecognised shape.
  The current failure mode — a glob matching nothing and `ingest_chunks` returning after
  logging at info level — is exactly what we are removing, not reproducing.
- **A schema migration that destroys data.** See "DataJoint `alter()`" below.
- **Replacing inference with different inference.** If metadata and database disagree,
  raise; do not pick a winner silently.

## Constraints

- DataJoint 2.3.2. `Table.alter()` **cannot rename attributes** — see below.
- `aeon_mecha` must keep working against every historical epoch.
- CI never sees golden data (it skips), so correctness evidence comes from local runs.
- Schema changes need sign-off per `CLAUDE.md`.

---

# The data

Measured from `AEONX1/golden_arrow_01/2026-10-13T094255Z/` (full probe recorded in
`.notes/2026-10-21-golden-arrow-survey.md`).

```
Metadata.json                                  <- epoch root
Environment/Environment_EphysMessageLog_<ts>Z.csv
NeuropixelsV2/
  NeuropixelsV2_ProbeA_<0..8>.arrow            386 cols, uint16 x384 + HubClock + Clock
  NeuropixelsV2_ProbeB_<0..8>.arrow
  NeuropixelsV2_Bno055_<0..8>.arrow            17 cols
  NeuropixelsV2_Commutator_Clock_<0..8>.bin    6000 x uint64
  NeuropixelsV2_Commutator_Turns_<0..8>.bin    6000 x 4 bytes (dtype TO CONFIRM)
  NeuropixelsV2_HarpSync_<ts>Z.csv             unchanged format
  M01_Probe{A,B}_*.json                        probe config, now INSIDE NeuropixelsV2/
  <serial>_gainCalValues.csv
```

9 chunks (0..8). Chunks 0–7 are exactly 600.0 s; chunk 8 is 397.8 s. Probe batches are
30 000 rows, IMU 100 rows — **uniform except the final batch of each file**. Totals:
155 934 000 samples/probe, 519 785 IMU samples. 18 000 000 == existing `CHUNK_SAMPLES`,
so the chunk convention is unchanged.

Streams are **index-aligned but not clock-aligned**: Bno055 chunk N starts 0.03–0.09 s
after ProbeA chunk N. Overlap logic is still required. ProbeA and ProbeB are also ~1.9 ms
apart, so nothing may assume a shared sample index across probes.

---

# Design

## 1. Metadata dispatch

`Metadata.json` is a **third** layout. `parse_metadata_probe_configs` and `get_probe_id`
already branch over two by duck-typing shapes; `get_probe_id` carries a `# noqa: PLR0911`
for seven returns. Do not add a third sniffed branch.

Replace shape-sniffing with dispatch on a declared discriminator — `workflow:
"Ephys.bonsai"`; older files are identified by the shapes already handled.

**Implementation note (deviation):** this section argued for dispatch on a declared
discriminator over shape-sniffing. `get_probe_id` in practice tries
`_headstage_probe_block` first, then `NeuropixelsV2{suffix}`, then `Devices` — a third
sniffed branch, which is what this section said not to add. Kept deliberately: the three
layouts are distinguished by disjoint, unambiguous keys, and a `workflow`-string dispatch
would add a lookup table that the shapes already encode. Recorded so it reads as a
decision rather than drift.

**`DiscriminatorType` is deliberately NOT used.** Its value (`"NeuropixelsV2eHeadstage"`)
matches neither key in `DEVICE_PROBE_TYPE_MAP` (`NeuropixelsV2Beta`, `NeuropixelsV2`), and
`device_name` is not only a type — it is the filename prefix, the device subdirectory, and
a stored field in the `<aeon_onix_stream>` codec payload (`codec.py:140-190`). Re-keying
all of that on an unverified string risks writing the wrong probe geometry into
`ElectrodeConfig` with no error. Directory discovery stays. Each layout gets one resolver returning a
normalised structure; callers stop knowing about layouts.

Facts to resolve per epoch, and where each now comes from:

| Fact | Old | New (`ephys.Headstage.*`) |
|---|---|---|
| probe serial | digits in `GainCalibrationFileName` | same field, relocated |
| probe enabled | **inferred** from `ProbeInterfaceFileName: null` | explicit `ProbeX.Enable` |
| probe interface JSON | `ConfigurationA/B.ProbeInterfaceFileName` | `ProbeX.ProbeConfiguration.ProbeInterfaceFileName` |
| subject <-> probe | `probe_assignments.json` (serial-keyed, external) | `subject.Implants[].Probe` |
| insertion number | **derived** by `find_or_create_probe_insertion` | `subject.Implants[].InsertionNumber` |

Two inference sites removed. `Implants` is label-keyed, which is safe here because the
same file also defines label->serial and is regenerated per epoch by acquisition — the
staleness failure mode requires an *external* mapping, which is what `probe_assignments.json`
is.

`TargetArea` and `ImplantationDate` are **already declared and never populated** —
`TargetArea` (Lookup), `InsertionTargetArea` (Manual link) and
`ProbeInsertion.implantation_date=null` all exist at `ephys.py:79-102`, and nothing in the
codebase writes to them. So this is not a schema change; it is filling declared-but-empty
structures from data the metadata now carries. **Open — see "Decisions needed".**

**Conflict rule:** if an existing `ProbeInsertion` for that subject+probe has a different
`insertion_number` than metadata asserts, raise. Do not reconcile.

**Back-compat:** old-format epochs have no `Implants`; they keep using
`probe_assignments.json`. Both paths live.

## 2. Probe path resolution

Probe JSONs moved into `<epoch>/NeuropixelsV2/`. `resolve_epoch_probe_json`
(`ephys_utils.py:466`) searches `<raw>/recording_configurations/` then `<epoch>/`.
Add the device subdirectory as a third location. Hard `FileNotFoundError` today, so this
fails loudly already — no silent-failure risk.

**Two call sites, not one:** `EphysEpochConfig.make` (`ephys.py:299`) and
`EphysBlockInfo.make` (`ephys.py:794`, via `load_device_channel_map`). Fixing only the
first leaves `EphysBlockInfo` failing later in the cascade.

## 3. File discovery and stream selection

The hard part. `AmplifierData` is gone from arrow filenames (`NeuropixelsV2_ProbeA_7.arrow`),
breaking six sites:

```
spike_sorting.py:212    file_name LIKE '%AmplifierData%.bin'
spike_sorting.py:1178   file_name LIKE '%Clock%.bin'          <- also matches Commutator_Clock
ephys.py:546            rglob("*_AmplifierData*.bin")
ephys.py:561            re.search(r"_(Probe[A-Z])_AmplifierData", ...)
ephys.py:596            name.replace("AmplifierData", "Clock")  <- no companion file exists
ephys_utils.py:367,374  glob + regex on the same token
```

`ephys.py:596` is structural: it derives the Clock path from the amplifier filename.
In arrow, `Clock` and `HubClock` are **columns in the probe file**, so there is no
companion to derive.

`spike_sorting.py:212` feeds `PreProcessing.make_compute`, which builds the SI recording
with `se.read_binary(dtype=np.uint16, num_channels=...)` or `si.load(<zarr twin>)`
(`spike_sorting.py:288-305`). There is no arrow path into SpikeInterface today, and handing
Arrow IPC container bytes to `se.read_binary` would read them as raw uint16 — silent
garbage, the exact anti-goal above. **An arrow branch in `make_compute` is therefore
required, not optional** (see §6).

**Decision: dispatch on file extension in code; no schema change.**

`EphysChunk.File` keeps one row per physical file. Discovery globs both
`*_AmplifierData*.bin` and `*_Probe[A-Z]_*.arrow`. Consumers that need a stream fetch the
rows and dispatch on suffix: `.arrow` -> read the named column; `.bin` -> resolve the
companion file. The `LIKE` patterns are replaced with stream-explicit selection that
cannot match `Commutator_Clock`.

*Rejected alternative:* add `stream_type` / `file_format` columns to `EphysChunk.File`.
More explicit and queryable, and `ADD` is `alter()`-safe. Rejected because it is a schema
change requiring sign-off, for information already unambiguous from the filename. Revisit
if a format appears where the extension does not determine the layout.

**Not inference.** Extension determines physical format unambiguously. This is distinct
from the inference being removed in §1, which guessed *semantics* from the *absence* of a
field.

## 4. Clock reading

Two `np.memmap(..., dtype=np.uint64)` sites (`ephys.py:604`, `spike_sorting.py:1189`).
For arrow, read the `Clock` column. Per Chris, the ephys reader in #630 is optimised for
time-slicing; clock access wants column slicing, so use a small separate reader rather
than `read_bonsai_onix_arrow`.

Only first and last values are needed at `ephys.py:604`, which arrow serves from the
first and last record batch without reading the file.

**#630 dependency:** that PR's `stream_name` is missing from `_kwargs`, so a reload
silently returns a different stream. We select `Clock` vs `AmplifierData*` by name out of
one file, so this must be fixed before building on it. Its other bug (non-uniform batch
sizes) does not bite — batches here are uniform except the last — and the `float32`
dtype bug does not either, since amplifier data is `uint16`.

## 5. IMU

Bno055 goes from 5 binaries per chunk to one arrow file. `find_overlapping_bno055_chunks`
globs `*_Bno055_Clock_*.bin` and reads first/last `uint64` by byte offset; for arrow it
reads first/last `Clock` from batch metadata. It stays necessary (streams are not
clock-aligned) but will usually return one chunk.

**Column mapping, verified empirically — do not re-derive:**

| Arrow column | Maps to | Evidence |
|---|---|---|
| `EulerAngle_Yaw` | `euler_x` | range [0, 359.94] matches old `euler_x` exactly |
| `EulerAngle_Pitch` | `euler_y` | +/-90 clamp matches |
| `EulerAngle_Roll` | `euler_z` | +/-180 matches |
| `Gravity_{X,Y,Z}` | `gravity_vector_{x,y,z}` | \|Gravity\| = 9.799 ~ 9.81 |
| `Acceleration_{X,Y,Z}` | `linear_acceleration_{x,y,z}` | \|Accel\| = 4.94, not ~9.8 -> gravity not embedded |
| `Quaternion_{X,Y,Z,W}` | `quaternion_{x,y,z,w}` | \|q\| = 1.0000 |
| `Calibration` | new | BNO055 status byte (15/79/143/207 observed) |
| `Temperature` | new | constant 87 over sampled window |

**Schema change: additive only.** Add `calibration=null` and `temperature=null` to
`OnixImuChunk`. **Do not rename `euler_x/y/z`.** Record the verified semantics in column
comments instead:

```
euler_x: json     # yaw / heading, 0-360
euler_y: json     # pitch, +/-90
euler_z: json     # roll, +/-180
```

### Why no rename: DataJoint `alter()` is unsafe

DJ 2.3.2 has a rename marker (`# {old_name}`) that is **dead code**. Tested against live
MySQL 8.0:

```
definition:  euler_yaw: json   # {euler_x}
generated:   DROP `euler_x` ; ADD `euler_yaw` json NOT NULL COMMENT ":json:{euler_x}"
result:      euler_yaw = None          <- data destroyed, no error
```

Detection is `re.compile(r'COMMENT "{...}')`, but DJ prefixes every comment with its type
(`":json:{euler_x}"`), defeating the match. Confirmed for `float64`, `json` and `<blob>`:
0 `CHANGE`, 3 `DROP`. **This is an upstream DataJoint bug and should be reported.**

Renaming would therefore ship a landmine: anyone running `OnixImuChunk.alter()` silently
loses their IMU summary stats. Pure `ADD` works correctly (verified), so additive-only
needs no migration script and leaves `alter()` safe.

If a rename is ever wanted, explicit DDL works and preserves data — note the `:json:`
comment prefix is required or DJ stops recognising the type:

```sql
ALTER TABLE `<prefix>_ephys`.`_onix_imu_chunk`
  CHANGE `euler_x` `euler_yaw` json NOT NULL COMMENT ":json:heading";
```

### Recorded tradeoff: the `Column` part table (option D)

Considered and deferred. A `Column` part table keyed on `column_name` would make the
column set queryable per row, format-agnostic, and would end DDL churn — a fixed DDL
column list fundamentally cannot describe two formats with different column sets. Rejected
now because the per-column-json pattern is the house style (every generated behaviour
stream table declares it that way, `streams_maker.py:190`) and two nullable columns is a
smaller change.

**Trigger to revisit:** if a third IMU column set appears, switch to the part table rather
than adding more nullable columns.

## 6. Arrow into SpikeInterface, and the `SyncedSpikes` exception

`PreProcessing` is the format boundary **for amplifier data**. Verified by mapping every
raw-file access in `spike_sorting.py`:

| Line | Table | Reads |
|---|---|---|
| 291–301 | `PreProcessing.make_compute` | raw amplifier -> SI recording |
| 1178–1189 | **`SyncedSpikes.make`** | **raw Clock files** |
| 1818 | `load_and_verify_binary_file` | a *pipeline-produced* binary — format-agnostic |

`ephys.py` was audited the same way. Every raw-file access there falls in a path this spec
already covers: HarpSync CSVs (`EphysEpoch.ingest_epochs`, `EphysSyncModel.ingest` —
format unchanged), probe JSON (§2), the AmplifierData/Clock pair (§3/§4), and Bno055 (§5).
**No third format exception exists.**

So `SpikeSorting`, `PostProcessing`, `SortedSpikes` and `UnitMatching` all operate on
intermediates and need no arrow awareness. **`SyncedSpikes` is the exception**, and #629
widens it:

```python
amp_len = read_amplifier_recording(
    ephys_dir / f.replace("Clock", "AmplifierData"), num_channels, NP2_FS_HZ
).get_num_samples()
```

After #629 it reads **two** raw streams and depends on the `Clock` <-> `AmplifierData`
**filename correspondence**, which arrow eliminates — one file holds both. This is where
getting it wrong shifts every spike timestamp, so it needs real coverage.

### Arrow -> SpikeInterface

Add an `.arrow` branch beside `.zarr`/`.bin` in `make_compute`, using
`read_bonsai_onix_arrow` from #630. The default `stream_name="AmplifierData"` substring-
matches `AmplifierData0..383` and excludes `Clock`/`HubClock` — correct for this data.
It accepts `gain_to_uV`/`offset_to_uV` and infers channel count, so it slots in where
`se.read_binary` sits.

**#630's `_kwargs` bug is masked for the probe case** (the default stream name *is*
`"AmplifierData"`, so a reload picks the right stream) but bites any non-default stream —
i.e. the clock reader. Fix it regardless.

### Where `PreProcessing` actually sits — corrected

The original draft of this section placed `PreProcessing` above the sorting fixtures, so
"arrow coverage stops at `PreProcessing`" read as a clean boundary. **It is not one.**
`PreProcessing` is keyed on `SortingTask`, which is keyed on `ElectrodeGroup` and
`SortingParamSet` (`spike_sorting.py:148-170`) — it is *inside* the sorting cascade.

Consequences, found in review round 1:

- A fixture split that skips the sorting cascade also skips `PreProcessing`. The golden
  arrow dataset therefore never reached it, and the defect that `PreProcessing.make_fetch`
  still selected `.bin` only survived the whole branch.
- Reaching `PreProcessing` for a dataset means giving it an electrode group and a
  paramset, i.e. sorting-task configuration — not ingestion configuration.

So arrow's `PreProcessing` coverage comes from the **synthetic factory**, which can build a
SortingTask cheaply, rather than from the golden dataset, which would need a 109 GB
intermediate. The golden arrow dataset covers ingestion through `EphysChunk` and IMU.

### Coverage split

| Dataset | Covers |
|---|---|
| `golden_arrow_01` | ingestion -> `EphysChunk`, plus IMU |
| **synthetic factory (arrow variant)** | **`PreProcessing` and `SyncedSpikes` arrow paths** |
| `abcGolden01` | unchanged — full cascade through curation and unit matching |

`SyncedSpikes` already has integration coverage built on **synthetic** data, not golden:
`tests/fixtures/ephys/ephys_factories.py:187` writes `AmplifierData_N.bin` + `Clock_N.bin`
per chunk, and `TestSyncedSpikesIndexingAndExtrapolation` exercises indexing and
extrapolation off it. #629 extended that same factory with `amp_as_zarr=True` to model the
compressed variant — so the factory is already where raw-format variants are modelled.
Adding an arrow emitter is the same move.

This covers every arrow-specific code path without generating a single golden sorting.

**Explicitly given up:** no evidence that a *real* arrow recording sorts sensibly
end-to-end. Acceptable because there is no arrow-specific logic in the sorter or below —
but state it in the PR rather than imply it. A real arrow golden sorting remains additive
if wanted later; nothing above depends on it.

## 7. Commutator — DEFERRED, design recorded

**Not implemented in this effort.** Every `Turns` sample is zero across both golden
datasets (see the blocker below), so ingestion could only be validated for the degenerate
case. Deferred until data exists where the commutator actually turned.

Present in **both** datasets (24 files in `abcGolden01`) and never ingested — a
pre-existing gap, not arrow work, so deferring it costs the arrow effort nothing.

The design below is recorded so it does not have to be rederived.

Proposed table, mirroring `OnixImuChunk`:

```
-> EphysSyncModel
---
sample_count: int32
timestamps: json
turns: json          # summary stats
turns_data: <blob>   # name TBD
```

**Storage: `<blob>`, in-database.** ~216 KB per sync-window row (6000 samples/chunk,
10 Hz, 4 bytes). Reasoning is self-containment, not size: `<aeon_onix_stream>` is a
*reference* codec that rebuilds from raw files on fetch, so it depends on raw files
remaining accessible. For a stream this small, copying into the database is better.
`<blob>` over `<npy>` because `<npy>` is schema-addressed external storage needing a
configured store and a file lifecycle.

Separate table rather than columns on `OnixImuChunk`: different rates (10 Hz vs 100 Hz)
on different chunk boundaries would force resampling or NaN padding.

### BLOCKER: there is no Commutator data to validate against

Measured across **both** golden datasets:

| | Clock | Turns |
|---|---|---|
| `golden_arrow_01` | 6 000 monotonic uint64 per chunk, full 600 s span | 51 979 samples, **all zero** |
| `abcGolden01` | same | 70 303 samples, **all zero** |

The stream records correctly — Clock is real and monotonic — but the commutator never
rotated in either recording. Every Turns sample is 0.

Two consequences:

1. **The dtype cannot be determined from the data.** All-zero bytes are identical under
   `uint32`, `int32` and `float32`. It must come from the Bonsai ONIX schema or from the
   acquisition side, not from inspection.
2. **There is nothing to validate ingestion against.** Tests can only prove that a column
   of zeros round-trips. The things most likely to be wrong — dtype interpretation, sign
   convention, whether turns accumulate or reset per chunk, scaling — are all invisible
   in all-zero data.

**Resolution:** deferred. Before picking it up, ask whether a recording exists where the
commutator actually turned, and confirm the dtype from the acquisition schema rather than
from inspection.

## 8. Tests and fixtures

Add `golden_arrow_01` to `GOLDEN_DATASETS` (behaviour + ephys arms) alongside
`abcGolden01`. The behaviour fixture `golden_dataset_config` is already parametrised with
clean ids, so `pytest -k abcGolden01` / `-k goldenArrow01` already works (verified: 50
tests -> 25/25 split). The **ephys** fixture is hardcoded (`conftest.py:534`).

**Scope: `golden_arrow_01` stops at `PreProcessing`.** That dissolves the three blockers a
full-cascade parametrisation would hit:

- `golden_arrow_01` has **no `golden_test_sorting/`** on disk, so `ephys_sorting_injected`
  (`conftest.py:1000`) would `pytest.skip` everything below it anyway — and curation is
  ~23 min of the 26 min run. Not needed: §6 covers `SyncedSpikes` synthetically.
- `ephys_test_blocks` hard-asserts `len(chunks) == 12` (`conftest.py:743`). Arrow gives
  9 chunks x 2 probes = 18 rows. That assert lives in a sorting-cascade fixture arrow no
  longer reaches; it must become config-driven rather than hardcoded, but only for the
  datasets that use it.
- `ephys_sorting_setup` hardcodes `electrode_group="shank3"`, `paramset_id="400"` and a
  single `electrode_config_name` (`conftest.py:860-930`). Arrow does not reach it.

So the ephys fixture parametrisation splits in two: the ingestion/PreProcessing fixtures
take both datasets; the sorting-cascade fixtures stay on `abcGolden01`. The builder decides
whether that is a marker, a separate fixture, or a config flag on the dataset entry.

Baselines change: two probes active (old golden had ProbeB only,
`expected_probe_count: 1`), different probe configs including an LFP variant for ProbeB.

**Runtime:** much better than feared. The ~23 min of the 26 min suite is the
ephys/curation fixture, which arrow does not reach. Adding `golden_arrow_01` through
`PreProcessing` adds ingestion time only. CI is unaffected (golden data absent, tests skip).

---

# Settled during review

- **Probe enablement conflicts.** `Enable: true` with no files on disk -> raise.
  Files on disk with `Enable: false` -> skip with an info log. Matches today's golden,
  where ProbeA is present but disabled (`expected_discovered_probes: 2` vs
  `expected_probe_count: 1`).

  **Both halves implemented.** The raise needed a new `enabled_probe_labels`, because
  `probe_info` is built by iterating labels discovered from *disk* — a probe the metadata
  declares enabled whose files never copied simply never appeared. Only the
  `ephys.Headstage` layout carries an `Enable` flag, so the cross-check is skipped for
  older layouts, which express disablement as a null `ProbeInterfaceFileName`.
- **`stream_df` column set.** Return the native set per format: 15 for arrow, 13 for
  binary. Normalising would discard `Calibration`/`Temperature`.

  **Implemented with the discriminator** (`stream_format`, defaulting to `binary` for rows
  written before it existed). An execution ruling had dropped it, reasoning that
  `load_and_merge_bno055` already dispatches on which file is on disk. Review round 1
  found the case that missed: `_load_bno055_arrow` validates by exact column-set equality
  at *decode* time, so one added upstream BNO055 column would make every previously
  ingested arrow row unfetchable, retroactively. Archiving or converting a raw file has
  the same effect from the other side. This is a published *fetch contract* change.
- **Metadata filename.** Try `Metadata.json`, fall back to `Metadata.yml`.
- **#630 is an in-repo dependency**, not a package one — it vendors the reader into
  `aeon_mecha/utils/spike_sorting_utils.py`. No `pyproject` change. Its missing
  `stream_name` in `_kwargs` must be fixed first, since we select `Clock` vs
  `AmplifierData*` by name out of one file.

# Decisions taken

1. **Scope: sorting is in, as far as `PreProcessing`.** `golden_arrow_01` covers ingestion
   -> `EphysChunk` -> `PreProcessing` + IMU. `SyncedSpikes`' arrow path is covered by an
   arrow variant of the synthetic factory. `abcGolden01` keeps the full cascade. No golden
   sorting is generated for arrow; see §6 for what that gives up.
2. **Base branch:** off `main`, pulling in **#626** (preprocessing selectable per
   `SortingParamSet`), **#629** (SyncedSpikes chunk boundaries) and **#630** (arrow reader).
3. **Populate `TargetArea` / `InsertionTargetArea` / `implantation_date`** from
   `subject.Implants[]`. Auto-insert new `TargetArea` rows with `skip_duplicates=True`,
   **stripping whitespace** and **logging a warning** on a genuinely new area. Do *not*
   normalise case — `mPFC` vs `HPC` carry meaningful capitalisation. Rationale: a pure
   controlled vocabulary would block ingestion the first time a new area is targeted, which
   is wrong; unguarded auto-insert lets `"mPFC"` / `"mPFC "` / `"mpfc"` become three rows
   and silently split `InsertionTargetArea` joins. Tighten to a fixed vocabulary later once
   the real list is known.
4. **lupin / generating an arrow golden sorting: not needed.** Dropped with the scope
   decision. Recorded because it was investigated: the fixture *injects* pre-computed KS4
   output rather than running a sorter (and kilosort4 is not installed locally — lupin,
   `simple`, `spykingcircus2`, `tridesclous2` are). Had we gone that route, three costs
   applied: `test_ephys_ingestion.py:388` hard-asserts a `KSLabel` property which lupin does
   not produce; lupin's `apply_preprocessing: True` + `preprocessing_dict` defaults overlap
   #626 and risk **double preprocessing** with no error; and 384 ch x 155.9 M samples x 2
   probes would need scoping to one probe/shank.

# Left to the builder

- Order of work and slicing.
- The shape of the metadata-layout dispatch (one resolver per layout, dataclass vs dict).
- The small arrow clock reader's interface.
- How the ephys fixtures split ingestion-only vs full-cascade datasets (marker, separate
  fixture, or a config flag on the dataset entry).
- The shape of the arrow emitter in the synthetic ephys factory.

# Known loss of a correctness guard

`abcGolden01` has `Seconds = Value.HarpTime + 1`; `golden_arrow_01` has them identical.
Bonsai used to add the HARP sync second twice into the `Seconds` index column, and that
is fixed upstream.

Production was always correct — `HarpSyncModel.Reader` fits its regression on
`Value.HarpTime`, never on `Seconds`. But two tests asserted the 1 s offset *as a
requirement* and failed on the new data. They now accept either era.

**The cost:** `test_sync_start_matches_value_harptime_column` carried a negative
assertion — that the `Seconds`-derived value is absent from stored sync starts — which
proved the reader was not using the wrong column. When the two columns are identical that
assertion cannot discriminate at all; it would fail against a perfectly correct reader.
It now runs only when the columns differ.

So **on new-format acquisitions there is no longer a test that would catch a reader
switching to the `Seconds` column.** The guard survives only for older data. This is
driven by an acquisition change the spec otherwise declares out of scope, and is recorded
here rather than left in a test comment.

# Constraints recorded from the team

Not open questions — answered, and they shape the build:

- **One subject per ephys epoch is the current experimental design, not a permanent
  constraint.** So read `Implants[]` under the single `subject.Subject`, but do not bake
  the assumption into the data model — `ProbeInsertion` already expresses per-probe
  subjects and must keep doing so. If multi-subject returns before the metadata can express
  it, `probe_assignments.json` is the escape hatch, which is a further reason not to delete
  that path yet.

- **Commutator data with real rotation exists — we just do not have it locally.** So the
  deferral in §7 is an *access* blocker, not a data-existence one. When that recording is
  available, the recorded design is immediately actionable.

# Behaviour added during review rounds 2 and 3

Six rules the spec did not state. Each exists because a review round found a path that
produced wrong data without saying so, and each is now pinned by a test.

- **A chunk index present in both formats is an error, not two chunks.** This holds on
  both sides: `find_overlapping_bno055_chunks` for Bno055, `_reject_duplicate_chunk_formats`
  for amplifier chunks. An epoch written across a format change may mix formats across
  *different* chunk indices — that stays supported and tested — but the same index in both
  forms is ambiguous. Loading prefers arrow purely because the file exists, so the binary
  copy would be silently discarded while its clock had already decided which window the
  chunk belongs to.

  **The ambiguity is decided on file existence, before any overlap test.** Deciding it
  after the window filter only catches collisions where *both* copies overlap the window
  being ingested, which is the subset that happens to be visible rather than the state
  that is actually undefined.

- **The format is a property of the epoch, not of whether a window held samples.** An
  empty IMU row records the same `stream_format` and carries the same column set as its
  populated siblings, or the fetch contract changes with the data.

- **A file that exists but cannot be interpreted is named, never treated as absent.** A
  corrupt `Metadata.json` raises rather than returning `None`; `None` reaches callers as
  "no metadata file" and sends an operator after a file that is sitting there, truncated.
  `EmptyArrowChunk` is a type, not a message substring, so the one caller that may skip an
  empty chunk is not coupled to another module's wording.

- **Selection is by stream, not by extension.** `AMPLIFIER_FILE_RESTRICTION` matches
  `'%Probe%.arrow'`, not `'%.arrow'`: the Bno055 file sits in the same device directory,
  and a bare extension match would hand `PreProcessing` a 17-column IMU table as amplifier
  data. The `OR` is parenthesised because DataJoint ANDs restrictions together.

- **A typo must not resolve to a default.** This applies to the *key* as well as the
  value: an unrecognised `SI_PREPROCESSING*` key is rejected, since otherwise the paramset
  reads as if it selected something while `aeon_default` runs. `stream_format` is validated
  on write for the same reason.

- **Arrow pins what binary pinned at the call site.** `dtype=uint16` was explicit in
  `se.read_binary` and implicit in the arrow schema; the gain and offset applied downstream
  assume 16-bit unsigned samples, so the arrow reader asserts it. Likewise a `Clock` column
  containing nulls is rejected rather than upcast to `float64` + `NaN`.

## Deliberate asymmetry between the two arrow readers

`read_arrow_clock_bounds` walks past an empty record batch at either end;
`_check_uniform_batches` rejects a *leading* one. This is not an inconsistency. The bounds
reader takes only the first and last values and does no stride arithmetic. `get_traces`
addresses batch *i* at `i * batch_len`, which a leading empty batch makes wrong for every
batch after it — and `batch_len` is read from batch 0, so an empty first batch also reaches
`get_traces` as zero. A trailing empty batch is harmless to both and is accepted by both.

If an acquisition is ever found to emit a leading empty batch, the resolution is to derive
`batch_len` from the first non-empty batch, not to relax the check.
