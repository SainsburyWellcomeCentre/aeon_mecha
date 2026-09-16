# Chunk-level spike trains

status: draft · 2026-09-16 · addresses #606 · prerequisites: PR #611, PR #588

## TL;DR

Spike times sit in the database one row per unit per ephys chunk, on a grain
that does not line up with behavioural data, so every analysis re-derives the
alignment by hand. `SpikeTrains` re-chunks curated, HARP-synced spikes to the
1-hour `acquisition.Chunk` grain and stores each chunk as a pynapple `TsGroup`,
backed by a new `<pynapple@dj_store>` codec. It buys that usability with two
accepted costs: no foreign key to the sorted data, so rows go stale and are
refreshed by an explicit step rather than a cascade; and no lazy loading, so a
week-long query reconstructs 1.5–23 GB of spike times.

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

### One row per behavioural chunk, and what it costs

A behavioural hour can be covered by several `EphysBlock`s, and
`UnitMatching.Spikes` deduplicates at `(global_unit, ephys chunk)`, so the spikes
for one hour can come from several `UnitMatching` rows. That is a many-to-one
relationship, and DataJoint has only one way to express provenance: the foreign
key. So there is a hard choice.

- Key on `UnitMatching` and the object **fragments** — two rows each holding part
  of the hour's roster, for every hour spanning a block boundary. Provenance and
  cascade come free; every downstream user pays.
- Key on the behavioural chunk and the object is **one clean row**, but there is
  no single parent to descend from, so nothing invalidates it automatically.

There is no third way. For a cascade to fire, an unbroken chain of foreign keys
must run from `UnitMatching` down to this table, and every table in that chain
carries `UnitMatching`'s key — so fragmentation propagates the whole way. A part
table listing contributing `UnitMatching` rows records provenance but does not
invalidate anything, because DataJoint cascades to children, not parents.

**This spec takes the second option.** Fragmentation is a continuous cost paid by
every user forever; staleness is a discrete cost paid by a maintainer at known
moments. The next section makes those moments detectable.

Note what survives: `-> acquisition.Chunk` and `-> ephys.ProbeInsertion` are real
foreign keys, so deleting an experiment, a chunk or an insertion still cascades.
What we give up is the leg to the *computed* ancestors — sorting, curation,
matching.

Two consequences to plan for. `populate()` ordering is no longer enforced by the
graph, so the worker configuration must sequence `SpikeTrains` after
`UnitMatching` itself. And `dj.Diagram` no longer shows where the data came from,
which on a platform where the ERD is how people learn the pipeline is a real
discoverability loss — hence `source_blocks` below, and this section.

### Upstream: `UnitMatching.Spikes`

`UnitMatching.Spikes` is already deduplicated across overlapping blocks, already
keyed by `global_unit`, and already HARP-synced. Ownership of a
`(global_unit, ephys chunk)` pair goes to the first block processed, not the
earliest in time: `UnitMatching.make()` skips a pair if any row exists, and
bidirectional seed propagation means a later block often runs first.

Reading `SyncedSpikes.Unit` instead would re-inherit the duplicate-spike problem
that convention solves, and would leave unit identity block-scoped.

### The table

```python
@schema                                  # aeon/dj_pipeline/processed_ephys.py
class SpikeTrains(dj.Computed):
    definition = """
    # Curated, HARP-synced spike trains for one behavioural chunk, as a pynapple TsGroup
    -> acquisition.Chunk                 # experiment_name, chunk_start (BEHAVIOURAL grain)
    -> ephys.ProbeInsertion              # subject, insertion_number
    ---
    n_units: int32                       # units in the roster, including silent ones
    n_spikes: int64                      # total spikes across all units
    coverage_frac: float32               # tot_length(time_support) / (chunk_end - chunk_start)
    n_partial_units: int32               # units sorted for less than the full chunk; 0 normally
    min_sync_r2: float32                 # worst HARP regression r2 over this chunk
    source_blocks: json                  # contributing EphysBlock keys + matching paramset
    spikes: <pynapple@dj_store>          # TsGroup; HARP seconds since 1904-01-01
    """
```

Primary key: `(experiment_name, chunk_start, subject, insertion_number)`.

`chunk_start` is the **behavioural** chunk, and it stays unrenamed. Because this
table has no foreign key to `EphysChunk`, there is no collision inside its own
lineage, and joins against the behavioural tables — which all key off
`acquisition.Chunk` — work directly. A user who explicitly joins `SpikeTrains`
against `EphysChunk` will get a DataJoint join-compatibility error rather than a
wrong answer, and must `proj`-rename. That is an unusual query failing loudly,
which is the right trade for the common query working silently.

`matching_paramset_id` is **not** in the key. `UnitMatching.Spikes` carries a
unique index on `(experiment_name, subject, insertion_number, global_unit,
chunk_start)` with no paramset, so a second paramset physically cannot write rows
for an already-owned triple. Putting it in the key would promise something the
upstream schema cannot deliver. It is recorded in `source_blocks` instead.

### Provenance without a foreign key

`source_blocks` is the fingerprint: the contributing `EphysBlock` keys and the
matching paramset, as stored at populate time. It replaces the lineage the
foreign key would have carried, and it is queryable.

```python
SpikeTrains.stale()          # rows whose source_blocks != the currently matched covering set
```

Staleness is **computed, never stored** — a stored boolean would itself go stale.
It covers two situations with one mechanism:

- A block covering this chunk was matched *after* the row was built. The row was
  right when computed and is now incomplete.
- Curation was re-run and the upstream rows were deleted and rebuilt.

The refresh is manual and belongs in the operator runbook:

```python
(SpikeTrains & SpikeTrains.stale()).delete()
SpikeTrains.populate()
```

This is the step that substitutes for the cascade. If it is not written down as a
routine somebody runs, it will not happen.

### `make()` contract

Compute whenever any covering block has been matched; record what contributed.
Do **not** withhold a row because coverage is incomplete — a missing row reads as
"no ephys here", which is indistinguishable from "we refused", whereas a row with
`source_blocks` recording one of two blocks is explicit and `stale()` will find
it once the second lands.

Refuse only on degenerate input: no matched coverage at all, or a round-tripped
spike count that does not match what was read.

### `key_source`

One entry per `(behavioural chunk, probe insertion)` where some `EphysChunk` for
that insertion overlaps the behavioural chunk's window and at least one covering
block has been matched. The overlap restriction follows `EphysBlockInfo.make()`,
which already resolves overlapping chunks with a SQL interval predicate.

Ephys falling outside every behavioural chunk is dropped. There is no behavioural
data to relate it to. `EphysBlockInfo` remains the route to it.

### What goes in the `TsGroup`

**Keys** — `global_unit`, cast to `int` (pynapple rejects non-integer keys).

**Roster** — every `global_unit` with spike data contributing to this chunk, plus
those a covering block found but which fired nothing, as an empty `Ts`. Empty
units survive the npz round trip because the `keys` array is stored explicitly.

Nothing is dropped for incomplete coverage. Dropping a unit would convert a
visible wrong *rate* into an invisible wrong *count*: a unit present in 23 chunks
of a day and absent from the boundary hour makes a day-long spike total silently
short, with no signal. Rates are recoverable from metadata; lost spikes are not.

**Metadata** — scalar columns only, so `getby_threshold` and boolean slicing work
and the pickled metadata blob stays small:

| Column | Source |
|---|---|
| `covered_seconds` | seconds of this chunk over which this unit was sorted |
| `subject`, `insertion_number` | `ephys.ProbeInsertion` |
| `electrode`, `shank`, `x_coord`, `y_coord` | `GlobalUnit → ProbeType.Electrode` |
| `unit_quality` | `SortedSpikes.Unit` |
| `snr`, `isi_violations_ratio`, `presence_ratio`, … | `SortingQuality.Metric.qc_metrics`, flattened |

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

That handles coverage that is missing for *every* unit. Coverage that differs
*between* units is the next section.

### One observation window per object

A `TsGroup` has **one** `time_support`, shared by every member. Members cannot
carry their own: build a group from two `Ts` with supports `[1,3]` and `[10,11]`
and pynapple takes their union and overwrites both members with it. This is a
property of the object model, not of the file format — the support is unified at
construction, in memory.

That matters when the blocks covering one chunk found different units. Take a
behavioural hour `[0, 3600)`. Unit 7 was sorted for the whole hour and fires at a
true 1 Hz. Unit 99 was only found by a block covering the second half, and also
fires at a true 1 Hz over the window it was sorted in. Measured on pynapple
0.11.4:

| group `time_support` | unit 7 | unit 99 |
|---|---|---|
| chunk window `[0, 3600]` | 1.00 Hz ✓ | **0.50 Hz ✗** |
| inferred from members | 1.00 Hz ✓ | **0.50 Hz ✗** |

Letting pynapple infer does not help: the members' supports overlap, and
`IntervalSet` merges overlapping intervals, so the union collapses to one
spanning interval. **No choice of group `time_support` makes both rates right.**

This is not a pynapple defect. A behavioural hour spanning a block boundary
genuinely has two observation windows, and the same error is available today to
anyone dividing `UnitMatching.Spikes` counts by a chunk duration. pynapple makes
the window an explicit property of the object instead of an invisible assumption
at the call site. What it does mean is that this table has to *decide* what the
window is, rather than leave it undefined.

The decision: the group's `time_support` is the chunk's overall ephys coverage,
and **`covered_seconds` in the per-unit metadata carries each unit's honest
denominator**. Two consequences:

```python
rate_true = tg.count().sum() / tg.covered_seconds    # correct for every unit, every chunk
```

- The correction is **unconditional**. `covered_seconds` is present and correct
  for every unit in every chunk, equal to the chunk coverage in the ordinary
  case, so downstream code never branches on whether this is a boundary chunk.
- **`TsGroup.rate` is unsafe on a raw chunk** and the docs must say so. It is
  right for the overwhelming majority of units, which is what makes it
  dangerous.

Seconds, not a fraction: fractions do not compose. Concatenating 24 chunks needs
a duration-weighted average, which consumers will get wrong. Seconds add.

The error is also an artifact of the chunk boundary, and it disappears under
`restrict()`, which recomputes rate over the new support. Restricting the example
to the second half returns unit 99 to 1.00 Hz. Since real analyses restrict to
bouts, visits and deliveries, the wrong number is only reachable by reading
`.rate` off a raw boundary chunk — that is, by going around `fetch_span`.

### Data-quality flags

A flag earns its place when its boring value is the overwhelming default and its
interesting value demands action. That rules out an enum — `coverage_frac < 0.99`
lets a reader set their own threshold, and two conditions can hold at once — and
it rules out counts with no actionable split, which is why there is no
`n_sync_models`.

Three altitudes, because the decisions happen at three different moments:

| Altitude | Carries | Answers |
|---|---|---|
| Row attributes | `coverage_frac`, `n_partial_units`, `min_sync_r2`, `source_blocks` | which chunks to use, without opening a file |
| `TsGroup` metadata | `covered_seconds` per unit | what each unit's rate should be divided by |
| `fetch_span` | warnings and errors | tell me now, while I am reading the data |

`n_partial_units` states the defect directly rather than by proxy.
`n_source_blocks > 1` would not work: two blocks that found the same units are
fine, and flagging them would be noise.

`fetch_span` **warns** when any contributing chunk has `n_partial_units > 0`,
naming the affected units, and **raises** when any contributing row is stale.
The asymmetry is deliberate: partial coverage is a permanent fact about the data
that a user can work around, while staleness means the row is simply out of date
and cheaply fixable. `allow_stale=True` overrides the error for anyone who
knows what they are doing.

### Time base: Harp seconds since 1904

pynapple stores `float64` seconds with no concept of an origin — no `t0`, no
timezone, nothing in the npz that records one — so the origin is a convention we
pick and hold. Harp-absolute costs nothing to produce: `swc.aeon.io.api.to_seconds`
already does it, keeping `1904` in the one place it lives. The ULP at 3.87e9 s is
477 ns, 70× finer than a 30 kHz sample and matching the `datetime(6)` keys. And
objects with different origins would combine silently and wrongly on
concatenation, which nothing can catch. Absolute magnitudes even compress
marginally better than rebased ones, so nothing argues the other way.

`make()` asserts `t_start > 3.0e9` before insert — cheap insurance against a
wrong origin, including the SpikeInterface trap where a persisted
`SortingAnalyzer` loses its time vector and returns spike times starting at 0.0.

### Alignment quality rides along

A chunk-level pynapple object is, by construction, a thing that *looks* perfectly
aligned. `SPEC_EPHYS_PIPELINE.md` warns that sub-second alignment comes from
`EphysSyncModel`'s per-chunk regression, not from epoch timestamps. Those
regressions carry `r2` and `n_samples`.

`min_sync_r2` puts that on the row, queryable without opening a file, so a
poorly-regressed window is a `WHERE` clause rather than a surprise.

`EphysSyncModel` is keyed by `EphysEpoch`, not by `ProbeInsertion`, so this is an
epoch-level fact duplicated onto each insertion's row. A behavioural chunk
spanning two ephys epochs draws from two model families, and `min_sync_r2` takes
the worst across both.

---

## The codec

`<pynapple@dj_store>` stores a pynapple object in a column, mirroring
`<xarray@store>` from PR #587. It knows nothing about spikes: it round-trips a
pynapple object and nothing more, and every AEON rule above — the Harp epoch, the
roster, the coverage support — lives in `SpikeTrains.make()`.

`spikeinterface.exporters.to_pynapple_tsgroup` already ships in the version this
repo pins (0.104.2), written by the author of open PR #610 with advice from
pynapple's maintainer. Building a `TsGroup` from sorted output is solved
upstream; this spec only persists the result.

### Implementation

`obj.save(path)` is `np.savez`, an uncompressed zip of `.npy` members — for a
`TsGroup`: `t` (float64, all spikes globally time-sorted), `index` (unit id per
spike), `keys`, `start`/`end` (the `time_support`), `type`, and `_metadata` (a
**pickled** dict). `nap.load_file` dispatches on `type`.

```python
class PynappleCodec(SchemaCodec):
    name = "pynapple"

    def validate(self, value): ...   # the six pynapple types

    def encode(self, value, *, key=None, store_name=None) -> dict:
        # _build_path(..., ext=".npz") -> _local_path -> makedirs -> value.save(local)
        return {"path": ..., "store": ..., "kind": type(value).__name__,
                "n_units": ..., "n_spikes": ..., "t_start": ..., "t_end": ...}

    def decode(self, stored, *, key=None):
        return nap.load_file(self._local_path(...))    # fast path for TsGroup, below
```

**Store-only comes free.** `SchemaCodec.get_dtype` already raises `"<pynapple>
requires @ (store only)"` when the `@` modifier is missing, so not overriding it
gives the behaviour we want with an accurate message. Dropping issue #606's
in-database form also removes temp-file buffering and keeps pickled payloads out
of the shared database. `_local_path` asserts `protocol == "file"`, as
`XArrayNetCDFCodec` does.

**The JSON record carries a summary**, so a user sizes a query without opening a
file, and garbage collection works with no extra code — `Codec.referenced_paths`
reads `path` and `store` from exactly this shape. The column targets `dj_store`,
which already holds `<filepath@dj_store>` entries pointing at externally-managed
SpikeInterface output; a GC'd codec sharing that store is worth verifying rather
than assuming.

### Two deviations from stock pynapple

Both measured, both still readable by a plain `nap.load_file()`.

**A faster `decode` for `TsGroup`.** `_from_npz_reader` masks the concatenated
array once per unit — O(units × spikes). At 600 units and 7.9 M spikes/hour on
realistic lognormal rates that is 4.24 s, of which I/O is 4%. One stable argsort
over a narrowed `index` plus offset slicing gives 0.48 s: **8.9× faster,
bit-identical**. `decode` takes that path for `TsGroup` and falls back to
`nap.load_file` otherwise, with a test pinning their equivalence. No pynapple
issue mentions this, so it is unreported rather than rejected — worth offering
upstream.

**A narrower `index` on write.** pynapple writes it as int64; the narrowest
signed type holding `max(global_unit)` cuts a realistic chunk from 126.8 MB to
**79.2 MB (−38%)** for one `.astype`. `_from_npz_reader` compares `index == key`
and broadcasts across widths, so stock pynapple still reads it.

### Rejected

| Option | Why not |
|---|---|
| `savez_compressed` | 3.7× smaller, 55–70× slower to write (11.7 s vs 0.21 s). Per-column knob, off by default. |
| zarr columns (Delta + zstd) | 20× smaller on disk, 40–60× slower to open, not pynapple-native. |
| memory-mapping the npz | Impossible — see below. |

`np.load(mmap_mode=)` silently ignores the flag on a zip, and numpy#23823, the
enhancement pynapple's #327 was closed against, was itself closed unmerged in
June 2025. Hand-rolled it still fails: the zip header leaves members
byte-unaligned, and `searchsorted` over 5 M float64 costs 50,536 µs against
2.0 µs aligned.

Worth keeping: `NpzFile.__getitem__` seeks to one zip entry, so reading `keys`
and `_metadata` costs kilobytes without touching either 40 MB array.

---

## Known limitations

### Rows go stale, and nothing fixes them automatically. Accepted.

`SpikeTrains` has no foreign key to the sorted data, so re-curation and later
matching do not invalidate it. A row can reflect a curation that was deleted
weeks ago, and it will look perfectly normal.

The mitigations are `source_blocks` + `SpikeTrains.stale()` to make it
detectable, `fetch_span` raising rather than warning on a stale row, and an
operator routine that runs the delete-and-repopulate recipe. Precedent:
`GlobalUnit` is already `dj.Manual` for the same reason, and
`SPEC_UNIT_MATCHING.md` already requires explicit orphan cleanup in
`restore_raw_sorting()`.

The residual risk is a user who reaches past `fetch_span` to `fetch1("spikes")`
and skips the check. That is why `fetch_span` is the documented entry point
rather than a convenience.

### No lazy loading. Accepted.

pynapple cannot lazily load a `TsGroup`, and structurally never will: its
`load_array=False` defers only the *values* of a `Tsd`, and a spike train is its
time index. `nap.NWBFile(lazy_loading=True)` silently ignores the flag for
`Units` tables. Demand is long-standing (issues #574, #420, #385, #379, all open
since 2023–2026); the maintainer's position is that a virtual time index "would
require a non-trivial refactor" and is a medium-term item. We take pynapple as it
is and document the cost.

The chunk grain is the mitigation. A user querying *D* hours fetches `ceil(D)+1`
chunks, of which at most two are partially wasted — 8% at a day, 1.2% at a
week. The primary key already says which chunks overlap the window, so
irrelevant ones are skipped in SQL without opening a file. That is coarse-grained
laziness, and at a one-hour grain it is most of the benefit.

**What it does not fix:** concatenating a week holds 1.5–23 GB of spike times in
memory, depending on unit count and firing rate, plus 20 s to several minutes of
reconstruction. A user who does that naively will run out of memory. This is a
real limitation and the reason `fetch_span` ships with the table.

On-disk figures are larger than in-memory ones: the npz stores a float64 time
plus an int16 unit index per spike (10 B), while a reconstructed `TsGroup` holds
only the float64 times (8 B). Decoding one chunk transiently holds both.

| Units | Mean rate | Spikes/chunk | npz/chunk | 7 d on disk | 7 d in memory |
|---|---|---|---|---|---|
| 100 | 3 Hz | 1.1 M | 11 MB | 1.8 GB | 1.5 GB |
| 300 | 5 Hz | 5.4 M | 54 MB | 9.1 GB | 7.3 GB |
| 600 | 8 Hz | 17.3 M | 173 MB | 29 GB | 23 GB |

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
    start=t0, end=t1,          # allow_stale=True to override the staleness check
)
```

`fetch_span` is the documented entry point, not a convenience wrapper. It owns
five things, three of them correctness-critical:

1. selecting the chunks overlapping `[start, end)` from the primary key alone
2. **raising** if any contributing row is stale, unless `allow_stale=True`
3. **warning** if any contributing chunk has `n_partial_units > 0`, naming them
4. restricting per chunk *before* concatenating, so peak memory is one chunk
5. re-keying `global_unit` when the span covers more than one insertion, and
   summing `covered_seconds` so the span-level object stays correct

Item 4 matters more than the 8.9× decode: at week scale the binding constraint is
memory, not CPU.

### Refreshing stale rows

```python
(processed_ephys.SpikeTrains & processed_ephys.SpikeTrains.stale()).delete()
processed_ephys.SpikeTrains.populate()
```

The manual step that stands in for the cascade. It belongs in the operator
runbook alongside the curation workflow, and it should run after any
re-curation or any new `UnitMatching` over already-covered time.

### Joint with behaviour

```python
key = {"experiment_name": exp, "chunk_start": t, "subject": s, "insertion_number": 1}
chunk_key = {k: key[k] for k in ("experiment_name", "chunk_start")}

spikes = (processed_ephys.SpikeTrains & key).fetch1("spikes")
pose   = (processed_movement.MousePositionTracking & chunk_key).fetch1(...)
```

A `SpikeTrains` key carries `subject`, `insertion_number` and
`matching_paramset_id`, which the behavioural tables do not have, so the
behavioural side takes the chunk half of the key. The point stands: the two
share `(experiment_name, chunk_start)` and neither side does time arithmetic.

---

## Out of scope

No changes to `SyncedSpikes`, `UnitMatching` or `GlobalUnit`. No NWB export —
pynapple cannot write it, and the neuroconv route is archival rather than
analytical; a DANDI deposit would be its own spec.

---

## Testing

Three markers, per `SPEC_TESTING.md`: `unit` (no database), `integration`
(testcontainers MySQL), `specialized` (golden datasets). Detailed assertions
belong in the implementation PR; these are the four that pin design decisions
and must not be quietly dropped.

**The codec round-trips, and the fast path is equivalent.** `TsGroup` spike
trains, keys, metadata and `time_support` survive; the argsort decode returns
exactly what `nap.load_file` does, including non-contiguous unit ids and units
with zero spikes; a narrowed `index` is still readable by stock pynapple. Mirror
`TestXArrayNetCDFCodec`, and add `"pynapple"` to the codec-registry pop list in
`tests/conftest.py` or the unit fixture double-registers.

**Garbage collection does not delete live data.** The `<xarray@store>` GC suite,
repeated: referenced/orphaned/deleted counts, dry run against real run,
idempotency, and a re-fetch of the survivor asserting equality.

**Re-chunking conserves spikes and coverage.** No spike lost or double-counted
against `UnitMatching.Spikes` over the same window. A spike at exactly
`chunk_end` lands in the next chunk, a gap yields a two-interval `time_support`,
and `covered_seconds` is per-unit correct and sums across concatenation.
`n_partial_units` is zero when covering blocks agree and non-zero when they
do not.

**Staleness is detected and clears.** `stale()` is empty after a clean populate
and non-empty after re-curation or a later block; `fetch_span` raises on stale
and passes with `allow_stale=True`; the delete-and-repopulate recipe returns it
to empty with the same spike counts.

Re-measure the 8.9× decode on a real Neuropixels chunk before that number goes
in a docstring — it comes from synthetic lognormal rates with no bursting,
refractory structure or drift.

---

## Open questions

1. **Should per-unit validity live upstream?** "Unit 99 was not observed by any
   sorting covering `[0, 1800)`" is a property of the `(global_unit, time)` pair,
   currently implicit in which `UnitMatching.Spikes` rows exist — so every
   consumer re-derives it, with the same bug. Raise with the `UnitMatching`
   owners before implementation.
2. **Coverage as data.** A second column holding an `IntervalSet` whose
   per-interval metadata names the units it covers would let `fetch_span` correct
   rosters automatically. Deferred from v1.
3. **Per-unit metadata is multi-valued.** `unit_quality` and `qc_metrics` are
   block-scoped, and `GlobalUnit`'s electrode is rewritten on every match. Which
   block wins, and is populate-time dependence acceptable?
4. **Population and backfill.** Who calls `populate()`, in what order (the FK
   graph no longer enforces it), and what does the backfill cost on Ceph?
5. **Block-length distribution.** One-row-per-chunk assumes boundary chunks are
   rare. Measure against `EphysBlock` on the golden dataset before implementing.
6. **Module placement and naming.** `processed_ephys.py` depends on PR #588;
   `SpikeTrains` against `CuratedSpikes` or `UnitActivity`.

---

## PR checklist

- [ ] `PynappleCodec` in `utils/codec.py`, registered before schema activation,
      `pynapple` as an optional extra, lazy-imported
- [ ] Codec tests: round trip, fast-path equivalence, GC suite
- [ ] `SpikeTrains` in `processed_ephys.py`, with `stale()`
- [ ] `fetch_span`: per-chunk restriction, re-keying, warn on partial, raise on
      stale
- [ ] Re-chunking and staleness tests; golden test on the ephys dataset
- [ ] Measure block-length distribution (open question 5)
- [ ] Raise upstream unit-validity with the `UnitMatching` owners (open question 1)
- [ ] Delete-and-repopulate added to the operator runbook
- [ ] Open PR into `main` (after explicit go-ahead)
