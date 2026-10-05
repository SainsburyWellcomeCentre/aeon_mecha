# Chunk-level spike trains

status: implemented in PR #625 · 2026-09-17 · addresses #606 · built on PR #613 (merged)

## TL;DR

Spike times sit one row per unit per ephys chunk, on a grain that does not line
up with behaviour — so every analysis re-derives the alignment by hand.
`SpikeTrains` does it once, storing each behavioural hour as a pynapple
`TsGroup` through the `<pynapple@dj_store>` codec.

Two costs come with that. The table has no foreign key to the sorted data, so
rows go stale and someone must refresh them; and pynapple cannot load lazily, so
a week-long query rebuilds 1.5–23 GB of spike times in memory.

---

## Why this table

A user asks: *what were these neurons doing while the animal was at patch 2?*

Today that means joining `SyncedSpikes.Unit` across chunk rows, re-keying
block-scoped unit ids to something stable, converting `datetime64[ns]` to
seconds, working out which parts of the window had ephys coverage at all — and
only then writing the raster code. `docs/ephys_runbooks/step06_analysis_examples.py`
is that pattern written down. Everyone repeats it a little differently, and the
coverage step is the one that goes wrong without saying so.

Three commitments shape the design:

1. **One row per behavioural chunk per probe insertion.** The same grain as
   `streams.*`, so spikes and behaviour join on `(experiment_name, chunk_start)`
   with no time arithmetic at the call site.
2. **Persistent unit identity.** `global_unit`, so a user can concatenate chunks
   across a week and follow the same neuron.
3. **One clock.** Harp seconds since 1904-01-01, float64, pipeline-wide.

Each row stores a pynapple `TsGroup` — a dict of per-unit timestamp series with
per-unit metadata and a `time_support` interval. That is the shape a population
of sorted units already has. It is also why the existing `<xarray@store>` codec
does not fit: xarray holds dense gridded arrays, and spike trains are ragged.
The `<pynapple@dj_store>` column type that stores it shipped in PR #613.

---

## Background

`acquisition.Chunk` belongs to the behavioural rig (AEON3, `raw`, wall-clock
hour boundaries). `ephys.EphysChunk` belongs to the ephys rig (AEONX1,
`raw-ephys`, ONIX file boundaries). `SPEC_EPHYS_PIPELINE.md` establishes these as
peers whose epochs start and stop independently, so their chunk boundaries never
coincide.

So spikes have to be split and regrouped across boundaries that never line up.
One `EphysChunk` can straddle two behavioural chunks, and one behavioural chunk
can span several `EphysChunk`s with gaps where the ephys rig was off.

Four steps carry spikes from probe to analysis. Sorting detects units
(`SortedSpikes`), syncing converts the ONIX clock to HARP (`SyncedSpikes`), and
matching assigns persistent identity (`UnitMatching`, `GlobalUnit`). The fourth
— changing the grain — has no table, so every consumer does it by hand.

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

**This spec takes the second option.** Fragmentation charges every user, forever.
Staleness charges one maintainer, at moments we can name, and the next section
makes those moments detectable.

Note what survives: `-> acquisition.Chunk` and `-> ephys.ProbeInsertion` are real
foreign keys, so deleting an experiment, a chunk or an insertion still cascades.
What we give up is the leg to the *computed* ancestors — sorting, curation,
matching.

Two consequences to plan for. The graph no longer enforces `populate()` order, so
the worker configuration has to run `SpikeTrains` after `UnitMatching`. And
`dj.Diagram` no longer shows where the data came from. People learn this pipeline
by reading the ERD, so that loss is real — which is why `source_blocks` exists,
and why this section does.

### Provenance without a foreign key

`source_blocks` is the fingerprint: the contributing `EphysBlock` keys and the
matching paramset, as stored at populate time. It replaces the lineage the
foreign key would have carried, and it is queryable.

```python
SpikeTrains.stale_chunks()          # rows whose source_blocks != the currently matched covering set
```

Staleness is **computed, never stored** — a stored boolean would itself go stale.
It covers two situations with one mechanism:

- A block covering this chunk was matched *after* the row was built. The row was
  right when computed and is now incomplete.
- Curation was re-run and the upstream rows were deleted and rebuilt.

The refresh is manual and belongs in the operator runbook:

```python
(SpikeTrains & SpikeTrains.stale_chunks()).delete()
SpikeTrains.populate()
```

This is the step that substitutes for the cascade. If it is not written down as a
routine somebody runs, it will not happen.

### Upstream: `UnitMatching.Spikes`

`UnitMatching.Spikes` is already deduplicated across overlapping blocks, already
keyed by `global_unit`, and already HARP-synced. A `(global_unit, ephys chunk)` pair belongs to the first block *processed*, not
the earliest in time: `UnitMatching.make()` skips a pair once any row exists, and
bidirectional seed propagation often runs a later block first.

Reading `SyncedSpikes.Unit` instead would hand back the duplicate spikes that
convention removes, and leave unit identity scoped to a block.

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
| Row attributes | `coverage_frac`, `n_partial_units`, `source_blocks` | which chunks to use, without opening a file |
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

## Known limitations

### Rows go stale, and nothing fixes them automatically. Accepted.

`SpikeTrains` has no foreign key to the sorted data, so re-curation and later
matching do not invalidate it. A row can reflect a curation that was deleted
weeks ago, and it will look perfectly normal.

Three things make it survivable. `source_blocks` and `stale_chunks()` make it
detectable; `fetch_span` raises on a stale row instead of warning; and the
operator runbook carries the delete-and-repopulate recipe. There is precedent:
`GlobalUnit` is `dj.Manual` for the same reason, and `SPEC_UNIT_MATCHING.md`
already asks for explicit orphan cleanup in `restore_raw_sorting()`.

The residual risk is a user who reaches past `fetch_span` to `fetch1("spikes")`
and skips the check. That is why `fetch_span` is the documented entry point
rather than a convenience.

### No lazy loading. Accepted.

pynapple cannot lazily load a `TsGroup`, and the reason is structural rather than
unfinished work: `load_array=False` defers the *values* of a `Tsd`, and a spike
train has no values — it *is* its time index. There is no large half to leave on
disk, so every chunk is read whole.

A mmappable format (a directory of `.npy` instead of the `.npz` zip) would allow
range-restricted reads, but the ceiling measures at ~6x on a 227 ms fetch: ~1 ms of
that is I/O and ~30 ms is pynapple object construction, which no backend removes.

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
plus an int64 unit index per spike (16 B), while a reconstructed `TsGroup` holds
only the float64 times (8 B). Decoding one chunk transiently holds both.

| Units | Mean rate | Spikes/chunk | npz/chunk | 7 d on disk | 7 d in memory |
|---|---|---|---|---|---|
| 100 | 3 Hz | 1.1 M | 17 MB | 2.9 GB | 1.5 GB |
| 300 | 5 Hz | 5.4 M | 86 MB | 14.5 GB | 7.3 GB |
| 600 | 8 Hz | 17.3 M | 276 MB | 46.4 GB | 23.2 GB |

### Sync quality is not on the row yet

Sub-second alignment comes from `EphysSyncModel`'s per-chunk regression, which
carries `r2` and `n_samples`. A `min_sync_r2` column would make a
poorly-regressed window a `WHERE` clause instead of a surprise, and the design is
straightforward: `EphysSyncModel` is keyed by `EphysEpoch` rather than
`ProbeInsertion`, so a chunk spanning two epochs draws on two model families and
the column takes the worst across both.

Not built. Until it is, a caller who cares about alignment quality queries
`EphysSyncModel` directly.

### Spikes are stored twice

`UnitMatching.Spikes` keeps its copy; this table adds another. Denormalising for read is a
deliberate trade and Ceph absorbs it. Folding `UnitMatching.Spikes` into this
table is the obvious follow-up once `SpikeTrains` has earned its place.

---

## What this buys at the call site

The whole argument for the table is this join, so here it is:

```python
key       = {"experiment_name": exp, "chunk_start": t, "subject": s, "insertion_number": 1}
chunk_key = {k: key[k] for k in ("experiment_name", "chunk_start")}

spikes = (processed_ephys.SpikeTrains & key).fetch1("spikes")      # a pynapple TsGroup
pose   = (processed_movement.MousePositionTracking & chunk_key).fetch1(...)

good = spikes[spikes.unit_quality == "good"]
counts = good.count(0.01)        # TsdFrame; unit metadata rides along
```

The two sides share `(experiment_name, chunk_start)` and neither does time
arithmetic. A `SpikeTrains` key also carries `subject` and `insertion_number`,
which the behavioural tables lack, so the behavioural side takes the chunk half.

Spans that are not one chunk go through `fetch_span`, which is the entry point
rather than a convenience: it raises on stale rows, warns on partial coverage,
sums each unit's `covered_seconds` so rates stay honest across the join, and
trims each chunk before concatenating. That last one matters more than decode
speed — at week scale the binding constraint is memory, not CPU.

---

## Out of scope

No changes to `SyncedSpikes`, `UnitMatching` or `GlobalUnit`. No NWB export —
pynapple cannot write it, and the neuroconv route is archival rather than
analytical; a DANDI deposit would be its own spec.

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
3. **Per-unit metadata is multi-valued.** `unit_quality` is block-scoped and
   `GlobalUnit`'s electrode is rewritten on every match. Which block wins is
   settled — most spikes, ties to the earliest. Whether the answer may depend on
   when `populate()` ran is not.
4. **Population and backfill.** Who calls `populate()`, in what order (the FK
   graph no longer enforces it), and what does the backfill cost on Ceph?
5. **Block-length distribution.** One row per chunk assumes boundary chunks are
   rare. Still unmeasured — it needs a production database, not the golden set.
6. ~~**Module placement and naming.**~~ Settled: `processed_ephys.py`,
   `SpikeTrains`.

---

