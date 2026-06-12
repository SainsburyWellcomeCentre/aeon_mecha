# Zarr Compression for Pipeline Intermediates

Spec for switching the spike sorting pipeline's intermediate file format from
uncompressed binary to zarr with Blosc-zstd compression. Covers the code
changes, test updates, and PR scope for a single implementation PR into main.

**Status:** Approved at team meeting 2026-06-11. Compression testing complete
on branch `es/compression-spec` (results in `REPORT_COMPRESSION_RESULTS.md`
on that branch). This spec defines the implementation PR.

**Branch:** `es/zarr-intermediates` (new, off main)

---

## Background

The spike sorting pipeline produces two large intermediate artifacts per
block per electrode group:

- **Preprocessed recording** — `recording.dat`, an uncompressed int16
  binary written by `PreProcessing.make_compute`. For a 30-minute block
  with 96 channels at 30 kHz, this is ~12.7 GB.

- **Sorting analyzer** — `sorting_analyzer/`, a directory of numpy arrays
  written by `PostProcessing.make_compute`. Contains waveforms, templates,
  quality metrics. Typically 0.3-0.5 GB per block.

These files live on Ceph and are the dominant consumers of pipeline disk
space. They are fully derived from the raw data and can be regenerated at
any time.

SpikeInterface supports zarr as an alternative storage format. Zarr applies
Blosc-zstd compression (lossless, built-in, no extra dependencies) to each
data chunk independently. The compression is transparent to downstream
code — `si.load()` handles both formats.

## Measured results

Tested on `es/compression-spec` branch using abcGolden01, subject
IAA-1147881, insertion 1, electrode group shank0 (96 channels). Single
30-minute block.

| File | Binary | Zarr | Reduction |
|------|--------|------|-----------|
| Recording | 12.67 GB | 5.03 GB | 60% |
| Sorting analyzer | 0.50 GB | 0.30 GB | 40% |
| **Total** | **13.17 GB** | **5.33 GB** | **60%** |

Pipeline execution time with zarr intermediates was 20% faster overall
(36.9 min vs 46.2 min), because compressed I/O reduces Ceph network
bandwidth. PreProcessing was 2.6x faster, PostProcessing 15% faster.
SpikeSorting was slightly slower (7.8 min vs 6.2 min mean) because the
Kilosort wrapper decompresses zarr to a temporary binary internally, but
this was within the normal run-to-run variation range (4.1-8.3 min).

Scaling rates per shank: ~25 GB/h binary vs ~10 GB/h zarr for recording,
~1 GB/h binary vs ~0.6 GB/h zarr for analyzer.

## Design

### Format switching mechanism

The pipeline already supports a `save_format` key in each paramset's
`params` dict (added on `es/compression-spec`). The value is either
`"binary"` or `"zarr"`. When absent, a default is used.

**This PR changes the default from `"binary"` to `"zarr"`.** Existing
paramsets that explicitly set `save_format` continue to work as before. New
paramsets and paramsets without an explicit `save_format` now produce zarr
output.

### Code changes

All changes are in `aeon/dj_pipeline/`. No changes to `aeon/schema/`,
`aeon/io/`, or any other package.

**1. Default flip (one-line change in four locations across two files)**

In `spike_sorting.py` (`PreProcessing.make_compute`,
`SpikeSorting.make_compute`, `PostProcessing.make_compute`) and
`spike_sorting_curation.py` (`ApplyCuration.make_compute`):

```python
# Before
save_format = params.get("save_format", "binary")

# After
save_format = params.get("save_format", "zarr")
```

**2. Zarr property filter**

When saving a recording to zarr, non-numeric recording properties (strings,
object arrays, structured arrays like `contact_vector`) must be stripped
because zarr v2 cannot serialize them without an explicit object codec. The
filter uses a numeric dtype allowlist:

```python
for prop in list(si_recording.get_property_keys()):
    values = si_recording.get_property(prop)
    if np.asarray(values).dtype.kind not in ("f", "i", "u", "b"):
        si_recording.delete_property(prop)
```

This keeps `gain_to_uV` (float64), `offset_to_uV` (int64), `location`
(float64), `group` (int64) and strips `contact_vector` (structured array
with string/object sub-fields). The filter runs only on the zarr path. The
binary path is unchanged.

This code already exists on `es/compression-spec` inside
`PreProcessing.make_compute`. For the implementation PR it should be
extracted to a small helper so it is not duplicated if other save points
need it.

**3. Sorting analyzer path fallback (all consumers)**

SpikeInterface's `create_sorting_analyzer(format="zarr", folder=path)`
appends `.zarr` to the folder name. So when PostProcessing passes
`output_dir / "sorting_analyzer"`, SI creates `sorting_analyzer.zarr/`.
Every location that loads a sorting analyzer by hardcoding
`output_dir / "sorting_analyzer"` needs to also check the `.zarr` path.

There are 7 such locations across two files:

- `spike_sorting.py`:
  - `PostProcessing.make_compute` (safety check, line ~540) — already
    handles both paths on `es/compression-spec`
  - `SIExport.make` (line ~629)
  - `SortedSpikes.make` (line ~719) — already has fallback on
    `es/compression-spec`
  - `Waveform.make_compute` (line ~905)
  - `SortingQuality.make_compute` (line ~996)
- `spike_sorting_curation.py`:
  - `ApplyCuration.make_compute` (line ~137)
  - `_get_analyzer_dir` helper (line ~274)

The fallback logic should be extracted to a shared helper rather than
duplicated in each location:

```python
def _resolve_analyzer_dir(output_dir: Path) -> Path:
    """Find sorting analyzer directory, checking both binary and zarr paths."""
    analyzer_dir = output_dir / "sorting_analyzer"
    if not analyzer_dir.exists():
        analyzer_dir = output_dir / "sorting_analyzer.zarr"
    return analyzer_dir
```

This helper lives in `spike_sorting.py` and is imported by
`spike_sorting_curation.py`. Each of the 7 locations replaces its
hardcoded path with a call to `_resolve_analyzer_dir(output_dir)`.

**4. PostProcessing safety check**

The existing safety check for pre-existing analyzer directories also needs
to check the `.zarr` variant:

```python
for check_dir in [analyzer_output_dir, output_dir / "sorting_analyzer.zarr"]:
    if check_dir.exists() and any(check_dir.iterdir()):
        raise FileExistsError(...)
```

This code already exists on `es/compression-spec`.

**5. File registration exclusions**

`PreProcessing.make_insert` already excludes `recording.dat` and
`recording.zarr/` contents from the DJ File part table (zarr directories
contain thousands of chunk files that should not be registered individually).
This code already exists on `es/compression-spec`.

### What is NOT included in this PR

- **`CompressionTest` table** — Testing artifact from `es/compression-spec`.
  Not production code.
- **`scripts/compression_report.py`** — Testing artifact.
- **Existing spec/plan/report documents** from `es/compression-spec` — This
  spec replaces them.
- **Raw data compression** — Separate effort, separate repo. See
  `SPEC_RAW_COMPRESSION.md`.
- **Historical data migration** — Existing binary intermediates remain
  as-is. `si.load()` handles both formats transparently when loading. New
  runs produce zarr; old data does not need to be converted.

### Backward compatibility

- `si.load()` detects format from folder contents and handles both binary
  and zarr transparently. No downstream code needs to know which format a
  given recording or analyzer uses.
- Existing paramsets with explicit `save_format: "binary"` continue to
  produce binary output.
- The `load_and_verify_binary_file` utility remains available for the binary
  path.
- No DB migration needed. The `sorting_output_dir` column stores a varchar
  path. The directory name doesn't change (it's the contents that change
  format).

---

## Test updates

### Current test state (main, as of PR #585)

The ephys integration tests in `tests/dj_pipeline/test_ephys_ingestion.py`
use the `foraging_abc_ephys_2026_05_11` golden dataset (abcGolden01 on
AEONX1, 384 recording channels, 8-channel sorting subset). The test flow:

1. `EphysEpoch.ingest_epochs` + `EphysEpochConfig.populate` +
   `EphysSyncModel.ingest` (fixture: `ephys_test_epochs`)
2. `EphysChunk.ingest_chunks` + `EphysBlockInfo.populate` (fixtures)
3. `PreProcessing.populate` — **runs live** against golden raw data
4. `SpikeSorting` — **force-injected** from pre-computed golden KS4 output
5. `PostProcessing.populate` — **runs live**
6. `SortedSpikes.populate` + `SyncedSpikes.populate` — **run live**

### What changes in tests

**No golden data changes needed.** The golden sorting output
(`golden_test_sorting/sorting_output`) contains KS4's native Kilosort
output (spike times, cluster assignments). It is format-agnostic and works
regardless of whether PreProcessing produced binary or zarr. PreProcessing
and PostProcessing run live and will naturally produce zarr output with the
new default.

**Test assertion updates:**

1. **`test_recording_binary_exists`** — Rename to
   `test_recording_zarr_exists`. Check that `recording/recording.zarr/`
   exists and is a non-empty directory. The binary-specific size checks
   (frame-size divisibility, minimum 500 MB) do not apply to compressed
   zarr. Instead, verify the recording loads correctly via `si.load()` and
   has the expected channel count and sample count range. Remove the
   `INT16_BYTES` constant.

2. **`test_sorting_analyzer_created`** — Update to check for
   `sorting_analyzer.zarr` (SI appends `.zarr` to the folder name when
   using zarr format). The `any(analyzer_dir.iterdir())` check still
   applies.

3. **Docstrings and comments** — Update references to `recording.dat` and
   `binary_folder` in test class docstrings.

### Verification on HPC

After the test changes are made, run the full test suite on the HPC against
the golden data to confirm everything passes with zarr output. This is the
final validation before the PR is ready for review.

---

## PR checklist

- [ ] Create `es/zarr-intermediates` branch off current main
- [ ] Cherry-pick or reimplement production code changes from
      `es/compression-spec` (property filter, zarr branching, path
      fallbacks, file registration exclusions)
- [ ] Change default from `"binary"` to `"zarr"` in all four
      `params.get("save_format", ...)` calls
- [ ] Add this spec document (`SPEC_ZARR_INTERMEDIATES.md`)
- [ ] Update test assertions (recording path, analyzer path)
- [ ] Run tests on HPC, verify all pass
- [ ] Open PR into main
