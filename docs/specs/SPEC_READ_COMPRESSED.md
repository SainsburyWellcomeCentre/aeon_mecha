# Reading compressed raw ephys files

Spec for making the aeon_mecha spike-sorting load path read **compressed**
raw ephys data. The companion `aeon_raw_compression` library losslessly
compresses raw `*_AmplifierData_*.bin` Neuropixels binaries to `.zarr` under a
processed data root. This spec covers the read side: when the pipeline goes to
read an amplifier chunk, it resolves the file on disk at load time — **preferring
`.zarr`, falling back to `.bin`** — without ever changing what is stored in the
database.

**Status:** Design approved 2026-07-08. Companion to `aeon_raw_compression`
(separate repo). Read-side design decision locked with Thinh 2026-06-26
(option A: store paths as read, resolve at load time, never rewrite stored paths).

**Branch:** `es/read-compressed` (off main, after PRs #589 / #591).

---

## Background

The `aeon_raw_compression` library compresses each raw amplifier binary at
`…/data/raw/<ARENA>/<experiment>/<epoch>/<device>/<file>.bin` to a zarr archive
at the mirrored sub-path under a **processed** root,
`…/data/processed/<ARENA>/<experiment>/<epoch>/<device>/<file>.zarr`. The raw
and processed roots are siblings; the sub-path below them is identical; only the
extension changes (`.bin` → `.zarr`). The compression is a verified byte-exact
round-trip, so the decompressed traces are bit-identical to the original binary.

Compression is optional and per-user: some users will never compress. Both the
raw store (uncompressed `.bin`) and the processed store (compressed `.zarr`) can
coexist, and for any given chunk file only one of the two may be present.

aeon_mecha reads amplifier binaries in exactly one place: `PreProcessing`
(`aeon/dj_pipeline/spike_sorting.py`). `PreProcessing.make_fetch` queries the
`EphysChunk.File` rows for the block's `*_AmplifierData_*.bin` files and returns
their stored (`.bin`) file paths together with each file's `directory_type`.
`PreProcessing.make_compute` then, per file, resolves the data directory for that
`directory_type` via `acquisition.Experiment.get_data_directory(key,
directory_type=d)` (ephys uses the split `raw` / `raw-ephys` directory types),
joins it with the stored file path (`ephys_dir / f`) to get the absolute path,
and reads it with `spikeinterface.extractors.read_binary`, concatenating the
per-chunk recordings.

Today that read is `.bin`-only. This spec makes it resolve `.zarr` first.

## Design

### Principle: resolve at load time, never rewrite stored paths (option A)

The database keeps storing the raw `.bin` path exactly as ingested. Nothing in
the DB changes. At read time, a pure resolver maps the stored raw path to the
file that actually exists on disk (the `.zarr` twin if present, else the raw
`.bin`). This keeps the two stores decoupled: compressing (or later deleting)
raw files never requires touching aeon_mecha's database, and a user who never
compresses sees no behavior change.

### The resolver (pure utility)

A single pure function is added to `aeon/dj_pipeline/utils/ephys_utils.py`:

```python
def resolve_ephys_file(raw_bin_path: Path) -> Path:
    """Resolve a raw amplifier .bin path to the file that exists on disk.

    Prefers the compressed .zarr twin under the processed store, falling back
    to the raw .bin. Raises FileNotFoundError (naming both candidates) if
    neither exists.
    """
```

**Path convention (baked in, not configurable).** The raw→processed mapping is
a fixed convention, so it lives inline in the function rather than as env vars,
DataJoint config, or module constants:

1. Split the path into components (`Path.parts`).
2. Count components exactly equal to `"raw"`:
   - **exactly one** → replace it with `"processed"` and swap the suffix to
     `.zarr`; that is the zarr candidate.
   - **zero** → the path is not under a `raw` store, so there is no zarr twin;
     skip straight to the `.bin` fallback.
   - **more than one** → raise `ValueError`. The mapping would be ambiguous (we
     cannot know which `raw` is the store root), and silently guessing could
     read the wrong store. A hard error is safer.
3. If the zarr candidate exists on disk → return it.
4. Else if the raw `.bin` exists → return it.
5. Else raise `FileNotFoundError` naming both candidates.

This matches `aeon_raw_compression`'s mapping
(`(PROCESSED_ROOT / bin.relative_to(RAW_ROOT)).with_suffix(".zarr")`, defaults
`/ceph/aeon/aeon/data/raw` ↔ `/ceph/aeon/aeon/data/processed`) **without importing
that library** — aeon_mecha must work for users who never install it. The swap
assumes the raw and processed roots are siblings differing only in that one
component (the default Ceph layout); the library technically allows a non-sibling
processed root, which this resolver intentionally does not support (the read side
was locked to a hardcoded convention, no env/DB config). A comment in the
function cross-references the library, and a unit test pins the mapping.

Exact-component equality (not substring replace) means `raw-ephys` — the
DataJoint `directory_type` *label*, which never appears as a physical path
component — is correctly ignored, as is any `Raw`/`RAW` casing.

### Wiring into the load path

In `PreProcessing.make_compute`, the per-file read loop resolves each file and
branches on the resolved suffix:

```python
resolved = resolve_ephys_file(ephys_dir / f)
if resolved.suffix == ".zarr":
    si_rec = si.load(resolved)
    si_rec.set_channel_gains(gain_to_uV)
    si_rec.set_channel_offsets(offset_to_uV)
else:
    si_rec = se.read_binary(
        resolved,
        sampling_frequency=fs_hz,
        dtype=np.uint16,
        num_channels=num_channels,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
    )
si_recs.append(si_rec)
```

**Why the `.zarr` branch re-applies gains and offsets.** `read_binary`'s
`gain_to_uV`/`offset_to_uV` arguments do not transform the stored samples; they
only *attach* gain/offset metadata to the recording object (verified in
SpikeInterface `binaryrecordingextractor.py`: the constructor calls
`set_channel_gains`/`set_channel_offsets` when those args are not `None`). The
compression library compresses from a plain `read_binary` **without** gain/offset
args, so the `.zarr` on disk carries no gain/offset metadata. `si.load(zarr)`
faithfully returns that gain-less recording, so we re-attach the same gain/offset
that the `.bin` branch passes. Because the round-trip is byte-exact and the
attached metadata is then identical, the `.zarr` branch produces a recording
equivalent to the current `.bin` branch through `unsigned_to_signed`,
preprocessing, and sorting.

Files are resolved independently, so a probe whose chunk files are a mix of
compressed and uncompressed still concatenates correctly (all branches yield
recordings with matching channel count, dtype, and sampling frequency).

This is the **only** read site that changes. Other raw reads — `Clock_*.bin`,
IMU (`Bno055`), tracking, and behavior streams — are not touched by
`aeon_raw_compression` (it compresses only `*_AmplifierData_*.bin`), so they are
left exactly as-is.

### What is NOT included

- **No database changes.** No schema change, no new table or column, no
  migration, no writes to any table, and no rewriting of stored file paths.
  `EphysChunk.File.file_path` keeps storing the `.bin`-relative path forever.
  The resolver runs purely at read time; nothing it computes is persisted.
- **No changes to `aeon_raw_compression`.** This is the read side only.
- **No changes to non-amplifier read paths.**
- **No historical migration.** Existing data with no `.zarr` twin reads exactly
  as before via the `.bin` fallback.

### Backward compatibility

- Users who never compress: every `resolve_ephys_file` call finds no `.zarr`
  and returns the `.bin` — identical behavior to today.
- Existing DBs and existing sorting outputs are unaffected.
- The change is read-time only and fully reversible (no persisted state).

---

## Testing

The project uses three pytest markers (`pyproject.toml`): `unit` (no database,
synthetic data), `integration` (needs a DB — locally via testcontainers, or on
the SWC HPC against an external DB via `TEST_DB_PREFIX` because there is no
Docker there), and `specialized` (defined but currently unused). The ephys
golden suite (`tests/dj_pipeline/test_ephys_ingestion.py`) is marked
`integration` and is run on the HPC because it reads real amplifier data off
Ceph. `PreProcessing.populate()` — the wiring site — is exercised only by that
golden suite.

### Unit tests (`unit`, no DB, `tmp_path`)

Extend `tests/dj_pipeline/utils/test_ephys_utils_unit.py`. These give the bulk of
the resolver confidence cheaply and run in CI:

- **Mapping** — a raw `.bin` path under a `raw` component maps to the mirrored
  `.zarr` under `processed` (pins the convention format; drift guard).
- **Preference / errors:**
  - both `.zarr` and `.bin` present → returns `.zarr`
  - only `.bin` present → returns `.bin`
  - only `.zarr` present → returns `.zarr`
  - neither present → `FileNotFoundError` (message names both candidates)
  - more than one `"raw"` component → `ValueError`
  - zero `"raw"` components, `.bin` present → returns `.bin` (fall-through)

### Integration test (`integration`, golden, HPC)

The existing golden tests already exercise the `.bin` fallback (no `.zarr`
present today) — free regression coverage. Add one golden test that stages a
`.zarr` twin for a golden amplifier file under the processed root, runs
`PreProcessing`, and asserts it (a) reads the `.zarr` and (b) produces a
recording equivalent to the `.bin` path. This test also covers the gain/offset
re-application end-to-end on real data. It runs on the HPC with the rest of the
golden suite and skips locally (gated by `require_ephys_golden_data`).

The gain/offset equivalence is not duplicated as a standalone local test: it is
effectively verifying SpikeInterface's own load/save round-trip, and the golden
integration test covers it on real data.

---

## PR checklist

- [x] Add `resolve_ephys_file` to `aeon/dj_pipeline/utils/ephys_utils.py`
- [x] Wire it into `PreProcessing.make_compute` (resolve + suffix branch +
      gain/offset re-apply on the `.zarr` branch)
- [x] Unit tests in `tests/dj_pipeline/utils/test_ephys_utils_unit.py`
- [x] Golden integration test in `tests/dj_pipeline/test_ephys_ingestion.py`
      (`TestCompressedReadEquivalence`)
- [x] Add this spec (`docs/specs/SPEC_READ_COMPRESSED.md`)
- [x] Run unit suite locally (113 passed)
- [ ] Run golden suite on HPC (`TestCompressedReadEquivalence` + regression)
- [ ] Open PR into main (after explicit go-ahead)
