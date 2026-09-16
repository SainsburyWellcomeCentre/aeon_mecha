# Chunk-level spike trains

status: draft · 2026-09-16 · addresses #606 · depends on PR #588

## TL;DR

Spike times sit in the database one row per unit per ephys chunk, on a grain
that does not line up with behavioural data, so every analysis re-derives the
alignment by hand. `SpikeTrains` re-chunks curated, HARP-synced spikes to the
1-hour `acquisition.Chunk` grain and stores each chunk as a pynapple `TsGroup`,
backed by a new `<pynapple@dj_store>` codec. The cost is memory: pynapple cannot
lazily load a `TsGroup`, so a week-long query pulls 2–29 GB, which the chunk
grain bounds and `fetch_span` manages but nothing eliminates.

---

## Why this table

A user asks: *what were these neurons doing while the animal was at patch 2?*

Answering that today means joining `SyncedSpikes.Unit` across chunk rows,
re-keying block-scoped unit ids to something stable, converting `datetime64[ns]`
to seconds, working out which spans of the window had ephys coverage, and then
writing the raster code. `docs/ephys_runbooks/step06_analysis_examples.py` is
that pattern written down. Every analyst repeats it, each slightly differently,
and the coverage step is the one they get wrong silently.

Three commitments shape the design:

1. **One row per behavioural chunk per probe insertion.** The same grain as
   `streams.*`, so spikes and behaviour join on `(experiment_name, chunk_start)`
   with no time arithmetic at the call site.
2. **Persistent unit identity.** `global_unit`, so a user can concatenate chunks
   across a week and follow the same neuron.
3. **One clock.** Harp seconds since 1904-01-01, float64, pipeline-wide.

The stored object is a pynapple `TsGroup`. pynapple is the standard Python
library for epoch-and-event neural data, and a `TsGroup` is a dict of per-unit
timestamp series carrying per-unit metadata and a `time_support` interval. That
is the shape a population of sorted units already has, which is why the existing
`<xarray@store>` codec does not fit: xarray holds dense gridded arrays, and
spike trains are ragged. [The codec](#the-codec) covers the choice in full.

---

## Background

`acquisition.Chunk` belongs to the behavioural rig (AEON3, `raw`, wall-clock
hour boundaries). `ephys.EphysChunk` belongs to the ephys rig (AEONX1,
`raw-ephys`, ONIX file boundaries). `SPEC_EPHYS_PIPELINE.md` establishes these as
peers whose epochs start and stop independently, so their chunk boundaries never
coincide.

Spikes therefore have to be split and regrouped across misaligned boundaries. An
`EphysChunk` can straddle two behavioural chunks, and a behavioural chunk can
span several `EphysChunk`s with gaps where the ephys rig was off.

Four operations carry spikes from probe to analysis. Sorting detects units
(`SortedSpikes`), syncing converts the ONIX clock to HARP (`SyncedSpikes`),
matching assigns persistent identity (`UnitMatching`, `GlobalUnit`). The fourth,
converting the grain, has no table — so every consumer does it by hand.

---

## Design

### Upstream: `UnitMatching.Spikes`

`UnitMatching.Spikes` is already deduplicated across overlapping blocks, already
keyed by `global_unit`, and already HARP-synced. The deduplication convention is
"earlier block owns", enforced by a unique index and a code-level check
(`SPEC_UNIT_MATCHING.md`).

Reading `SyncedSpikes.Unit` instead would re-inherit the duplicate-spike problem
that convention solves, and would leave unit identity block-scoped.

### The table

```python
@schema                                    # aeon/dj_pipeline/processed_ephys.py
class SpikeTrains(dj.Computed):
    definition = """
    # Curated, HARP-synced spike trains for one behavioural chunk, as a pynapple TsGroup
    -> acquisition.Chunk                   # experiment_name, chunk_start (BEHAVIOURAL grain)
    -> ephys.ProbeInsertion                # subject, insertion_number
    -> spike_sorting.UnitMatchingParamSet  # provenance of global_unit identity
    ---
    n_units: int32                         # units in the roster, including silent ones
    n_spikes: int64                        # total spikes across all units
    coverage_frac: float32                 # tot_length(time_support) / (chunk_end - chunk_start)
    n_sync_models: int32                   # EphysSyncModel windows spanning this chunk
    min_sync_r2: float32                   # worst regression r2 among those windows
    spikes: <pynapple@dj_store>            # TsGroup; HARP seconds since 1904-01-01
    """
```

Primary key: `(experiment_name, chunk_start, subject, insertion_number,
matching_paramset_id)`.

`chunk_start` here is the **behavioural** chunk. `ephys.EphysChunk` uses the same
attribute name for its own grain, so any query joining both must disambiguate.

`UnitMatchingParamSet` sits in `UnitMatching`'s primary key, because a block can
be matched under several paramsets. Leaving it out here would re-introduce the
ambiguity that spec removed.

### `key_source`

One entry per `(behavioural chunk, probe insertion, paramset)` where some
`EphysChunk` for that insertion overlaps the behavioural chunk's window and
`UnitMatching` has run for the covering block. The overlap join follows the
`streams_maker` pattern (`aeon/dj_pipeline/utils/streams_maker.py`), which
restricts `acquisition.Chunk` against a device's install/remove window.

Ephys falling outside every behavioural chunk is dropped. There is no
behavioural data to relate it to. `EphysBlockInfo` remains the route to it.

### What goes in the `TsGroup`

**Keys** — `global_unit`, cast to `int` (pynapple rejects non-integer keys).

**Roster** — every `global_unit` whose owning block overlaps this chunk, with an
empty `Ts` for units that fired no spikes. `UnitMatching.Spikes` writes no row
for a silent unit, so a roster built only from present rows would change chunk to
chunk and make concatenation across a span wrong by default. Empty units survive
the npz round trip because the `keys` array is stored explicitly. The roster is
stable within a block and grows across blocks as `UnitMatching` finds new units.

**Metadata** — scalar columns only, so `getby_threshold` and boolean slicing work
and the pickled metadata blob stays small:

| Column | Source |
|---|---|
| `subject`, `insertion_number` | `ephys.ProbeInsertion` |
| `electrode`, `shank`, `x_coord`, `y_coord` | `GlobalUnit → ProbeType.Electrode` |
| `unit_quality` | `SortedSpikes.Unit` |
| `snr`, `isi_violations_ratio`, `presence_ratio`, … | `SortingQuality.Metric.qc_metrics`, flattened |
| `block_start` | the owning `EphysBlock` |

`subject` and `insertion_number` ride along because **`global_unit` is unique only
within an insertion** (`SPEC_UNIT_MATCHING.md`). Two insertions in one experiment
can both have `global_unit=1`, so merging two `TsGroup`s collides silently. The
metadata makes the collision detectable; `fetch_span` re-keys on merge so it
never happens.

The raw `qc_metrics` dict stays in `SortingQuality.Metric`, where it is queryable.

### `time_support` is coverage, not chunk bounds

```python
rate = n_samples / sum(time_support interval lengths)   # pynapple/core/base_class.py
```

Setting `time_support` to the nominal chunk window when ephys covered only part
of it understates every unit's firing rate, with no warning, in an object that
looks authoritative. Cover 40 minutes of an hour and every rate is 33% low. So:

```
time_support = (union of overlapping EphysChunk windows) ∩ [chunk_start, chunk_end)
```

`IntervalSet` holds several intervals natively, so gaps inside the chunk survive,
and multi-interval supports round-trip through the npz. `coverage_frac` exposes
the same fact as a secondary attribute, so a user filters partial chunks with a
SQL restriction instead of discovering the problem in their firing rates.

Two boundary rules:

- The window is **half-open**, `[chunk_start, chunk_end)`. A spike at an exact
  boundary belongs to the later chunk. Twenty-four boundaries a day makes this
  worth pinning.
- Passing `time_support` to a pynapple constructor **drops samples outside it**,
  silently. Deriving the support from the chunks the spikes came from makes that
  impossible, but `make()` asserts the round-tripped spike count anyway.

### Time base: Harp seconds since 1904

pynapple stores `float64` seconds and has no concept of a time origin — no `t0`,
no timezone, nothing in the npz that records one. Whatever origin we choose is a
convention we have to record and defend. Harp-absolute wins on four counts:

- *Concatenation.* Objects with different origins combine silently and wrongly,
  and pynapple has nowhere to store an origin to compare.
- *Precision.* The float64 ULP at 3.87e9 s is 477 ns, 70× finer than a 30 kHz
  sample period and matching the `datetime(6)` primary keys.
- *Conversion.* `swc.aeon.io.api.to_seconds` already does it, and `1904` stays in
  the one place it lives today.
- *Compression.* Absolute magnitudes compress at 3.69× against 3.52× for
  epoch-relative, because subtracting an epoch only shifts which byte lanes stay
  constant.

`make()` asserts `t_start > 3.0e9` before insert. That check catches a whole
class of silent wrong-origin bugs, including the SpikeInterface trap where a
`SortingAnalyzer` persisted to `binary_folder` or `zarr` loses its time vector
and returns spike times starting at 0.0, 122 years adrift, with no exception.

### Alignment quality rides along

A chunk-level pynapple object is, by construction, a thing that *looks* perfectly
aligned. `SPEC_EPHYS_PIPELINE.md` warns that sub-second alignment comes from
`EphysSyncModel`'s per-chunk regression, not from epoch timestamps. Those
regressions carry `r2` and `n_samples`.

`n_sync_models` and `min_sync_r2` put that on the row, queryable without opening
a file, so a poorly-regressed window is a `WHERE` clause rather than a surprise.

---

## The codec

The table needs to store a pynapple object in a column. `<pynapple@dj_store>`
does that, mirroring `<xarray@store>` from PR #587.

### Why pynapple

Beyond the shape fit, pynapple brings the operations that follow it: epoch
handling (`IntervalSet`, `restrict`), per-unit metadata, and the standard
analyses (`count`, `value_from`, tuning curves, PETHs).

It is also where SpikeInterface points. `spikeinterface.exporters.to_pynapple_tsgroup`
ships in the version this repo already pins (0.104.2), written by the author of
open PR #610 with advice from pynapple's maintainer. Converting sorted output to
a `TsGroup` is solved upstream; this spec only has to persist the result.

The codec knows nothing about spikes. Like `XArrayNetCDFCodec`, it round-trips a
pynapple object and nothing more. Every AEON-specific rule above — the Harp
epoch, the roster, the coverage support — lives in `SpikeTrains.make()`.

### File format

`obj.save(path)` is `np.savez`, an uncompressed zip of `.npy` members. A
`TsGroup` writes:

| Key | Contents |
|---|---|
| `t` | float64, every unit's spikes concatenated and globally time-sorted |
| `index` | int, the unit id for each spike |
| `keys` | int64, the unit ids |
| `start`, `end` | float64, the `time_support` intervals |
| `type` | `"TsGroup"` |
| `_metadata` | a **pickled** dict of the per-unit metadata columns |

`nap.load_file(path)` dispatches on `type` and rebuilds the object.

### Implementation

```python
class PynappleCodec(SchemaCodec):
    """Store a pynapple object as .npz at {schema}/{table}/{pk}/{field}_<token>.npz."""

    name = "pynapple"

    def validate(self, value): ...          # Ts, Tsd, TsdFrame, TsdTensor, IntervalSet, TsGroup

    def encode(self, value, *, key=None, store_name=None) -> dict:
        schema, table, field, pk = self._extract_context(key)
        config = (key or {}).get("_config")
        path, _ = self._build_path(schema, table, field, pk, ext=".npz",
                                   store_name=store_name, config=config)
        local = self._local_path(path, store_name, config)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        value.save(local)
        return {"path": path, "store": store_name, "kind": type(value).__name__,
                "n_units": ..., "n_spikes": ..., "t_start": ..., "t_end": ...}

    def decode(self, stored, *, key=None):
        return nap.load_file(self._local_path(stored["path"], stored.get("store"),
                                              (key or {}).get("_config")))
```

**Store-only comes free.** `SchemaCodec.get_dtype` already raises `"<pynapple>
requires @ (store only)"` when the `@` modifier is missing, so not overriding it
gives the behaviour we want with an accurate error message. Issue #606 also
proposed an in-database `<pynapple>` form; dropping it removes the temp-file
buffering and keeps pickled payloads out of the shared database. `save()` and
`load_file()` are path-only, which is a problem for a blob and a non-problem
here, because the store is a local path. `_local_path` asserts
`protocol == "file"` as `XArrayNetCDFCodec` does, and `dj_store` on Ceph is the
only store in the deployment.

**The JSON record carries a summary.** `kind`, `n_units`, `n_spikes`, `t_start`,
`t_end` — the same idea as `<xarray>`'s `dims` and `data_vars`, so a user sizes a
query without opening a file. Garbage collection then works with no extra code,
because `Codec.referenced_paths` reads `path` and `store` from exactly this shape.

### Two deviations from stock pynapple

Both are measured, and both keep the file readable by a plain `nap.load_file()`.

**A faster `decode` for `TsGroup`.** `TsGroup._from_npz_reader` masks the
concatenated array once per unit, which is O(units × spikes). On 600 units and
7.9 M spikes an hour at realistic lognormal rates that costs 4.24 s, of which I/O
is 4%. One stable argsort over a narrowed `index`, plus offset slicing, gives
0.48 s — **8.9× faster, with bit-identical spike trains**.

`decode` takes that path for `TsGroup` and falls back to `nap.load_file` for
every other type. A test asserts the fast path equals the stock path, so the
coupling to pynapple's format is checked rather than assumed.

Nothing in pynapple's issue tracker mentions this, so it is unreported rather
than known and rejected. Worth offering upstream; until it lands, we carry it.

**A narrower `index` on write.** `index` holds a unit id per spike and pynapple
writes it as int64. Writing the narrowest signed integer type that holds
`max(global_unit)` cuts a realistic chunk from 126.8 MB to **79.2 MB, −38%**, for
one `.astype`. `_from_npz_reader` compares `index == key` and broadcasts across
widths, so stock pynapple still reads the file.

### Rejected

| Option | Why not |
|---|---|
| `savez_compressed` | 3.7× smaller, 55–70× slower to write (11.7 s vs 0.21 s). Available as a per-column knob, off by default. |
| zarr columns (Delta + zstd) | 20× smaller on disk, 40–60× slower to open (1.3–2.0 s vs 32 ms), and not pynapple-native. |
| memory-mapping the npz | Cannot work. See below. |

`np.load(mmap_mode=...)` silently ignores the flag on a zip archive, and
numpy#23823 — the enhancement pynapple's issue #327 was closed against — was
itself closed unmerged in June 2025. Hand-rolling it does not help: the zip local
header leaves each member byte-unaligned, and `searchsorted` over 5 M float64
costs 50,536 µs unaligned against 2.0 µs aligned.

One npz property worth keeping: `NpzFile.__getitem__` seeks to a single zip
entry, so reading `keys` and `_metadata` costs kilobytes without touching either
40 MB array. "Which units are in this chunk?" is nearly free.

---

## Known limitations

### No lazy loading. Accepted.

pynapple cannot lazily load a `TsGroup`, and structurally never will: its
`load_array=False` defers only the *values* of a `Tsd`, and a spike train is its
time index. `nap.NWBFile(lazy_loading=True)` silently ignores the flag for
`Units` tables. Demand is long-standing (issues #574, #420, #385, #379, all open
since 2023–2026); the maintainer's position is that a virtual time index "would
require a non-trivial refactor" and is a medium-term item. We take pynapple as it
is and document the cost.

The chunk grain is the mitigation. A user querying *D* hours fetches `ceil(D)+1`
chunks, of which at most two are partially wasted — under 8% at a day, under 1%
at a week. The primary key already says which chunks overlap the window, so
irrelevant ones are skipped in SQL without opening a file. That is coarse-grained
laziness, and at a one-hour grain it is most of the benefit.

**What it does not fix:** loading and concatenating a week is 2–29 GB in memory
depending on unit count and firing rate, plus 20 s to several minutes of
reconstruction. A user who does that naively will run out of memory. This is a
real limitation and the reason `fetch_span` ships with the table.

Per chunk:

| Units | Mean rate | Spikes/chunk | npz (int16 index) | 24 h | 7 d |
|---|---|---|---|---|---|
| 100 | 3 Hz | 1.1 M | 11 MB | 0.3 GB | 1.8 GB |
| 300 | 5 Hz | 5.4 M | 54 MB | 1.3 GB | 9.1 GB |
| 600 | 8 Hz | 17.3 M | 173 MB | 4.1 GB | 29 GB |

### Spikes are stored twice

`UnitMatching.Spikes` keeps its copy; this table adds another. Denormalising for
read is a deliberate trade, and Ceph absorbs it. Folding `UnitMatching.Spikes`
into this table is the obvious follow-up once `SpikeTrains` has proven itself,
and is out of scope here with four spike-sorting PRs in flight.

### Pickle in the payload

pynapple's `_metadata` is a pickled dict, so `nap.load_file` needs
`allow_pickle=True` and a malicious npz would execute code on read. The files sit
inside `dj_store` on Ceph, written only by the pipeline, and nothing else in the
deployment reads them. Keeping metadata to scalar columns keeps the pickled blob
small.

---

## Query patterns

### One chunk

```python
from aeon.dj_pipeline import processed_ephys

tg = (processed_ephys.SpikeTrains & key).fetch1("spikes")   # a pynapple TsGroup
good = tg[tg.unit_quality == "good"]
counts = good.count(0.01)                                   # TsdFrame, metadata preserved
```

`count()` carries the unit metadata through to the resulting `TsdFrame`'s
per-column metadata, so quality metrics and electrode stay attached after binning.

### A time span

```python
tg = processed_ephys.SpikeTrains.fetch_span(
    experiment_name="...", subject="...", insertion_number=1,
    start=t0, end=t1,
)
```

`fetch_span` restricts **per chunk before concatenating**, so peak memory is one
chunk rather than the whole span. It also re-keys `global_unit` when the query
covers more than one insertion, since those ids collide. Shipping this helper
matters more than the 8.9× decode: at week scale the binding constraint is
memory, not CPU.

### Joint with behaviour

```python
spikes = (processed_ephys.SpikeTrains & chunk_key).fetch1("spikes")
pose   = (processed_movement.MousePositionTracking & chunk_key).fetch1(...)
```

Same `(experiment_name, chunk_start)`. No time arithmetic.

---

## What is NOT included

- **No change to existing tables.** `SyncedSpikes`, `UnitMatching` and
  `GlobalUnit` keep their current definitions and contents.
- **No migration.** `SpikeTrains` is a new `dj.Computed` table; populating it is
  the only way data arrives.
- **No NWB.** pynapple cannot write NWB, and the NWB route (via neuroconv) is
  archival rather than analytical. A DANDI deposit would be a separate spec.
- **No in-database `<pynapple>` form.** Store-backed only.
- **No within-file laziness.**

---

## Testing

Three markers, per `SPEC_TESTING.md`: `unit` (no database), `integration`
(testcontainers MySQL), `specialized` (golden datasets).

### Unit — codec

Extend `tests/dj_pipeline/utils/test_codec_unit.py`, matching the
`TestXArrayNetCDFCodec` shape: a real `datajoint.settings.Config()`, stores set to
a `tmp_path`, codec imported inside each test body.

- `validate` accepts each of the six pynapple types and rejects others.
- `encode` writes one tokened `.npz` at a schema-addressed path containing the
  primary key, and returns the expected JSON summary.
- `decode` returns an equal object; `TsGroup` spike trains, keys, metadata and
  `time_support` all round-trip.
- The fast path equals `nap.load_file` — same keys, same spike trains, same
  metadata, including non-contiguous unit ids and units with zero spikes.
- A narrowed `index` is still readable by stock `nap.load_file`.
- A non-`file` protocol raises.

Add `"pynapple"` to the codec-registry pop list in `tests/conftest.py`'s
`mock_dj_for_unit`, alongside `"xarray"`, or the fixture double-registers.

### Unit — re-chunking

The grain conversion is arithmetic over interval lists and needs no database:

- An `EphysChunk` straddling two behavioural chunks splits at the boundary.
- A behavioural chunk spanning several `EphysChunk`s concatenates in order.
- A gap between `EphysChunk`s produces a two-interval `time_support`.
- Partial coverage yields the right `coverage_frac`.
- A spike at exactly `chunk_end` lands in the next chunk, not this one.
- A unit with no spikes appears in the roster with an empty `Ts`.

### Integration — codec round trip and GC

Extend `tests/dj_pipeline/utils/test_codec_integration.py`: a throwaway schema
with a `<pynapple@…>` column, store `location.mkdir()` before any insert.

Garbage collection gets the same treatment `<xarray@store>` got —
`schema_paths_referenced` / `orphaned` / `deleted`, dry run against real run,
`bytes_freed`, `errors == 0`, idempotency of a second `collect()`, and a re-fetch
of the surviving row asserting equality. Silent deletion of live data is the
failure that matters.

Also verify the MariaDB dict-to-JSON patch in `aeon/dj_pipeline/__init__.py`
holds for a second JSON-dtype codec.

### Specialized — golden

Populate `SpikeTrains` on the ephys golden dataset and assert:

- Total spikes across chunks equals total spikes in `UnitMatching.Spikes` over
  the same window. No spike lost, none double-counted.
- `t_start > 3.0e9` on every row.
- `coverage_frac` matches the `EphysChunk` coverage computed independently.
- The roster is identical across chunks within a block.
- Firing rates from the `TsGroup` match rates computed by hand from
  `UnitMatching.Spikes` and the coverage intervals.
- Re-running `ApplyOfficialCuration` cascades into `SpikeTrains` and repopulates.

**Re-measure the decode fast path here.** The 8.9× comes from synthetic lognormal
firing rates with no bursting, refractory structure or drift. The number goes in
the docstring only after it holds on a real Neuropixels chunk.

---

## Open questions

1. **Module placement.** `processed_ephys.py` follows the convention PR #588
   establishes, but that PR is still open. Alternative: put `SpikeTrains` in
   `spike_sorting.py` and move it later.
2. **Deferred activation.** `processed_feeder` and `processed_movement` defer
   because they depend on dynamically generated stream tables. `SpikeTrains`
   depends only on static schemas, so it does not need to — but consistency
   within the module may argue for it anyway.
3. **A second `IntervalSet` column** for per-chunk valid periods, separate from
   `time_support`? SpikeInterface has a `valid_unit_periods` extension we do not
   currently compute.
4. **Naming.** `SpikeTrains` in a `processed_ephys` schema, against
   `CuratedSpikes`, `UnitActivity`, `ChunkedSpikeTrains`.

---

## PR checklist

- [ ] `PynappleCodec` in `aeon/dj_pipeline/utils/codec.py`
- [ ] Register it in `aeon/dj_pipeline/__init__.py` before schema activation
- [ ] `pynapple` as an optional extra in `pyproject.toml`, lazy-imported in the codec
- [ ] Add `"pynapple"` to the registry pop list in `tests/conftest.py`
- [ ] Codec unit tests, including fast-path equivalence
- [ ] Codec integration tests, including the GC suite
- [ ] `SpikeTrains` in `aeon/dj_pipeline/processed_ephys.py`
- [ ] Re-chunking unit tests
- [ ] `fetch_span` helper with per-chunk restriction and cross-insertion re-keying
- [ ] Golden test on the ephys dataset, including the curation-cascade path
- [ ] Re-measure decode on a real Neuropixels chunk
- [ ] This spec
- [ ] Open PR into `main` (after explicit go-ahead)
