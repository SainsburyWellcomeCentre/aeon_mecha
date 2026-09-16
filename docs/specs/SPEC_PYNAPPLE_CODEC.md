# A `<pynapple>` codec for DataJoint

status: draft · 2026-09-16 · addresses #606

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
                "n_units": ..., "n_spikes": ..., "t_start": ..., "t_end": ...}

    def decode(self, stored, *, key=None):
        return nap.load_file(self._local_path(...))   # fast path for TsGroup, below
```

Registered by import in `aeon/dj_pipeline/__init__.py` before any schema
activation, alongside the three existing codecs. `pynapple` is an optional extra,
lazy-imported inside `encode`/`decode`, so nobody who avoids the column takes the
dependency — which matters, because pynapple hard-requires `pynwb`, `h5py`, `neo`
and `numba` even when unused.

The returned JSON carries a summary, so a caller sizes a query without opening a
file. Garbage collection then works with no extra code: `Codec.referenced_paths`
reads `path` and `store` from exactly this shape.

### Two deviations from stock pynapple

Both measured, both still readable by a plain `nap.load_file()`.

**A faster `decode` for `TsGroup`.** `_from_npz_reader` masks the concatenated
array once per entity — O(entities × events). At 600 units and 7.9 M spikes an
hour on realistic lognormal rates that is 4.24 s, of which I/O is 4%. One stable
argsort over a narrowed `index` plus offset slicing gives 0.48 s: **8.9× faster,
bit-identical**. `decode` takes that path for `TsGroup` and falls back to
`nap.load_file` for every other type, with a test pinning their equivalence.

Nothing in pynapple's issue tracker mentions this, so it is unreported rather
than known and rejected. Worth offering upstream; until it lands, we carry it.

**A narrower `index` on write.** pynapple writes it as int64. The narrowest
signed type holding the largest key cuts a realistic chunk from 126.8 MB to
**79.2 MB (−38%)** for one `.astype`. `_from_npz_reader` compares `index == key`
and broadcasts across widths, so stock pynapple still reads the file.

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
- A narrowed `index` is still readable by stock `nap.load_file`.
- A non-`file` protocol raises.

Integration, in `test_codec_integration.py`: a throwaway schema with a
`<pynapple@…>` column and `location.mkdir()` before any insert. Repeat the
`<xarray@store>` GC suite — referenced/orphaned/deleted counts, dry run against
real run, idempotency, and a re-fetch of the survivor asserting equality, because
silent deletion of live data is the failure that matters. Also verify the MariaDB
dict-to-JSON patch in `aeon/dj_pipeline/__init__.py` holds for a second
JSON-dtype codec.

Re-measure the 8.9× on real data before that number goes in a docstring. It comes
from synthetic lognormal rates with no bursting, refractory structure or drift.

---

## PR checklist

- [ ] `PynappleCodec` in `aeon/dj_pipeline/utils/codec.py`
- [ ] Register in `aeon/dj_pipeline/__init__.py` before schema activation
- [ ] `pynapple` as an optional extra in `pyproject.toml`, lazy-imported
- [ ] Add `"pynapple"` to the registry pop list in `tests/conftest.py`
- [ ] Unit tests, including fast-path equivalence
- [ ] Integration tests, including the GC suite
- [ ] Re-measure decode on real data
- [ ] Open PR into `main` (after explicit go-ahead)
