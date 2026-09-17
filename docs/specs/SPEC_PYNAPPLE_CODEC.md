# A `<pynapple>` codec for DataJoint

status: implemented · 2026-09-16 · addresses #606

`<xarray@store>` (PR #587) covers dense gridded data — pose tracks, continuous
traces. Nothing covers the other half of the model: ragged event and interval
data, where each entity has its own count of events at its own times. Spike
trains, foraging bouts, pellet deliveries and visit intervals are all that shape,
and all currently land in the database as bare `<blob>` arrays of datetimes that
every consumer re-assembles by hand.

`<pynapple@dj_store>` stores a pynapple object in a column and hands it back
intact. It mirrors `XArrayNetCDFCodec` closely enough that most of this spec is
"the same, with `.npz` in place of `.nc`".

The codec is domain-agnostic: it round-trips a pynapple object and knows nothing
about what the object means. `SpikeTrains` (`SPEC_SPIKE_TRAINS.md`) is the first
consumer.

---

## Design

### Store-backed only

`SchemaCodec.get_dtype` already raises `"<pynapple> requires @ (store only)"`
when the `@` modifier is missing. Not overriding it gives exactly the behaviour
we want, with an accurate error message.

Issue #606 also proposed an in-database `<pynapple>` form returning `<blob>`.
Dropping it removes the temp-file buffering that form would need — `save()` and
`load_file()` are path-only, which is a problem for a blob and a non-problem for
a file store, where the target already is a local path — and keeps pickled
payloads out of the shared database.

`_local_path` asserts `protocol == "file"`, as `XArrayNetCDFCodec` does. The only
store in the deployment is `dj_store` on Ceph, so this costs nothing today.

### File format

`obj.save(path)` is `np.savez`, an uncompressed zip of `.npy` members. For a
`TsGroup`: `t` (float64, all events globally time-sorted), `index` (the entity id
per event), `keys`, `start`/`end` (the `time_support` intervals), `type`, and
`_metadata` — a **pickled** dict of the per-entity metadata columns.
`nap.load_file` dispatches on `type` and rebuilds the object.

The pickle means decoding executes arbitrary code in principle. The files are
written only by the pipeline, into a store nothing else reads. Worth knowing, not
worth blocking on.

### Implementation

```python
class PynappleCodec(SchemaCodec):
    name = "pynapple"

    def validate(self, value): ...   # Ts, Tsd, TsdFrame, TsdTensor, IntervalSet, TsGroup

    def encode(self, value, *, key=None, store_name=None) -> dict:
        # _build_path(..., ext=".npz") -> _local_path -> makedirs -> value.save(local)
        return {"path": ..., "store": ..., "kind": type(value).__name__,
                "n_rows": ..., "t_start": ..., "t_end": ...}

    def decode(self, stored, *, key=None):
        return nap.load_file(self._local_path(...))   # fast path for TsGroup, below
```

Registered by import in `aeon/dj_pipeline/__init__.py` before any schema
activation, alongside the three existing codecs. `pynapple` is an optional extra,
lazy-imported inside `encode`/`decode`, so nobody who avoids the column takes the
dependency — which matters, because pynapple hard-requires `pynwb`, `h5py`, `neo`
and `numba` even when unused.

The returned JSON carries a summary — `kind`, `n_rows`, `t_start`, `t_end` — so a
caller sizes a query without opening a file. The keys are deliberately generic:
this codec stores pynapple objects, not spikes, and carries no domain vocabulary.
Garbage collection then works with no extra code, because
`Codec.referenced_paths` reads `path` and `store` from exactly this shape.

### One deviation from stock pynapple

**A faster `decode` for `TsGroup`.** `_from_npz_reader` masks the concatenated
array once per entity — O(entities × events). One stable argsort over a
**narrow-dtype view** of `index`, plus offset slicing, replaces that loop.
`decode` takes that path for `TsGroup` and falls back to `nap.load_file` for every
other type, with a test pinning their equivalence.

Measured on the eight golden Kilosort4 sortings (real spike trains, 64-101 units
and 0.9-2.7 M spikes each, 30 kHz), every result bit-identical:

| units | spikes | npz | stock | fast | speed-up |
|---|---|---|---|---|---|
| 64 | 2.0 M | 32 MB | 0.246 s | 0.104 s | 2.4× |
| 94 | 1.3 M | 20 MB | 0.168 s | 0.054 s | 3.1× |
| 101 | 2.7 M | 43 MB | 0.460 s | 0.132 s | 3.5× |

**The gain scales with unit count**, because the loop it replaces is
O(units × spikes) while the argsort is O(n log n). A synthetic 600-unit /
7.9 M-spike object gives 7.5× (4.22 s → 0.56 s). A real `SpikeTrains` row
aggregates four shanks into one object, so expect the upper half of that range
rather than the per-shank figures above.

The narrowing happens **in memory, for the sort only**. Nothing about the stored
file changes, so stock `nap.load_file` reads everything we write. Two traps, both
covered by tests: the argsort must be `kind="stable"` or per-entity times come
back out of order, and `bypass_check=True` alone computes `rate` from each
member's pre-unification support, so members must be built with the group support
first. Measured: `bypass_check=False` 2.8×, member-support-then-bypass 3.1×, and
7.5× once the argsort runs on a narrow dtype.

Nothing in pynapple's issue tracker mentions the mask loop, so it is unreported
rather than known and rejected. Worth offering upstream; until it lands, we carry
it.

Narrowing `index` **on disk** was considered and dropped. It buys ~38% file size
and nothing else, and would mean re-packing every `.npz` after `save()` — which
asserts knowledge of pynapple's private key layout against a `0.x` library, for
no correctness benefit.

### Rejected

| Option | Why not |
|---|---|
| `savez_compressed` | 3.7× smaller, 55–70× slower to write (11.7 s vs 0.21 s) — fatal across thousands of chunks. |
| zarr columns (Delta + zstd) | 20× smaller on disk, 40–60× slower to open, not pynapple-native. |
| memory-mapping the npz | Impossible — see below. |

`np.load(mmap_mode=)` silently ignores the flag on a zip, and numpy#23823, the
enhancement pynapple's #327 was closed against, was itself closed unmerged in
June 2025. Hand-rolled it still fails: the zip header leaves members
byte-unaligned, and `searchsorted` over 5 M float64 costs 50,536 µs against
2.0 µs aligned.

Worth keeping: `NpzFile.__getitem__` seeks to one zip entry, so reading `keys`
and `_metadata` costs kilobytes without touching either 40 MB array.

### Not included

No lazy loading. pynapple cannot lazily load a `TsGroup` and structurally never
will — `load_array=False` defers only the *values* of a `Tsd`, and an event train
is its time index. Consumers bound memory by choosing a storage grain;
`SPEC_SPIKE_TRAINS.md` covers what that means in practice.

---

## Testing

Mirror `TestXArrayNetCDFCodec` in `tests/dj_pipeline/utils/test_codec_unit.py`:
a real `datajoint.settings.Config()`, stores on a `tmp_path`, codec imported
inside each test body. Add `"pynapple"` to the codec-registry pop list in
`tests/conftest.py`'s `mock_dj_for_unit`, or the fixture double-registers.

- `validate` accepts the six pynapple types and rejects everything else.
- `encode` writes one tokened `.npz` at a schema-addressed path containing the
  primary key, and returns the expected JSON summary.
- `decode` returns an equal object: spike trains, keys, metadata and
  `time_support` all round-trip, including multi-interval supports,
  non-contiguous keys and entities with zero events.
- The fast path equals `nap.load_file` on all of the above.
- A non-`file` protocol raises.

Integration, in `test_codec_integration.py`: a throwaway schema with a
`<pynapple@…>` column and `location.mkdir()` before any insert. Repeat the
`<xarray@store>` GC suite — referenced/orphaned/deleted counts, dry run against
real run, idempotency, and a re-fetch of the survivor asserting equality, because
silent deletion of live data is the failure that matters. Also verify the MariaDB
dict-to-JSON patch in `aeon/dj_pipeline/__init__.py` holds for a second
JSON-dtype codec.

`TestPynappleCodecOnGoldenSpikes` in `tests/dj_pipeline/test_ephys_ingestion.py`
loads real Kilosort4 sortings off disk, wraps them as a `TsGroup` on Harp
seconds, and asserts bit-exactness and fast-path equivalence. It reads the
sortings directly rather than through the pipeline: the codec stores pynapple
objects and does not care where the spike times came from, so routing through
`SyncedSpikes` would couple it to a fixture rework and to PR #611 for no gain.
It skips cleanly when the golden artifacts are absent.

---

## PR checklist

- [x] `PynappleCodec` in `aeon/dj_pipeline/utils/codec.py`
- [x] Register in `aeon/dj_pipeline/__init__.py` before schema activation
- [x] `pynapple` as an optional extra in `pyproject.toml`, lazy-imported
- [x] Add `"pynapple"` to the registry pop list in `tests/conftest.py`
- [x] Unit tests, including fast-path equivalence
- [x] Integration tests, including the GC suite
- [x] Re-measure decode on real data — 2.4-3.5× on the golden sortings
- [ ] Open PR into `main` (after explicit go-ahead)
