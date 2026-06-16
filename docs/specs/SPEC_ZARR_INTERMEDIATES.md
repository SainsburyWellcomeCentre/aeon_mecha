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

**1. Default flip (four locations across two files)**

In `spike_sorting.py` (`PreProcessing.make_compute`,
`SpikeSorting.make_compute`, `PostProcessing.make_compute`) and
`spike_sorting_curation.py` (`ApplyOfficialCuration.make`):

```python
# Before
save_format = params.get("save_format", "binary")

# After
save_format = params.get("save_format", "zarr")
```

In the three `spike_sorting.py` locations this is a one-line default change.
In `ApplyOfficialCuration.make` it is more than a flip: on main this method
had no format handling at all (it always called
`curated_analyzer.save(folder=..., overwrite=True)`), so a zarr branch was
added:

```python
save_format = params.get("save_format", "zarr")
if save_format == "zarr":
    if curated_analyzer_dir.exists():
        shutil.rmtree(curated_analyzer_dir)
    curated_analyzer.save_as(format="zarr", folder=curated_analyzer_dir)
else:
    curated_analyzer.save(folder=curated_analyzer_dir, overwrite=True)
```

The `shutil.rmtree` is needed because `SortingAnalyzer.save_as` has no
`overwrite` flag (unlike `save`), so a pre-existing curated directory from a
prior apply/revert/re-apply of the same `curation_id` must be cleared first.

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

This logic is extracted into the helper `_strip_non_numeric_properties` in
`aeon/dj_pipeline/utils/spike_sorting_utils.py` (see change 3 for why the
helpers live in a utils module) and called from `PreProcessing.make_compute`
on the zarr path only.

**3. Sorting analyzer path fallback (all consumers)**

SpikeInterface's `create_sorting_analyzer(format="zarr", folder=path)`
appends `.zarr` to the folder name. So when PostProcessing passes
`output_dir / "sorting_analyzer"`, SI creates `sorting_analyzer.zarr/`.
Every location that loads a sorting analyzer by hardcoding
`output_dir / "sorting_analyzer"` needs to also check the `.zarr` path.

The consumer locations across two files are:

- `spike_sorting.py`: `SIExport.make`, `SortedSpikes.make`, `Waveform.make`,
  `SortingQuality.make`
- `spike_sorting_curation.py`: `ApplyOfficialCuration.make` and the
  `_get_analyzer_dir_from_key` helper

(`PostProcessing.make_compute` is the creator, not a consumer; its safety
check explicitly tests both paths — see change 4.)

The fallback logic is a shared helper rather than duplicated in each
location:

```python
def _resolve_analyzer_dir(output_dir: Path) -> Path:
    """Find sorting analyzer directory, checking both binary and zarr paths."""
    analyzer_dir = output_dir / "sorting_analyzer"
    if not analyzer_dir.exists():
        analyzer_dir = output_dir / "sorting_analyzer.zarr"
    return analyzer_dir
```

This helper, together with `_strip_non_numeric_properties`, lives in
`aeon/dj_pipeline/utils/spike_sorting_utils.py`, imported by both
`spike_sorting.py` and `spike_sorting_curation.py`. The helpers are pure
(no table or schema dependency), so placing them in a utils module keeps
them importable and unit-testable without activating the spike_sorting
schema (importing `spike_sorting.py` triggers a DB connection). Each
consumer replaces its hardcoded path with a call to
`_resolve_analyzer_dir(output_dir)`.

**4. PostProcessing safety check and return path**

The safety check for pre-existing analyzer directories checks both variants:

```python
for check_dir in [analyzer_output_dir, output_dir / "sorting_analyzer.zarr"]:
    if check_dir.exists() and any(check_dir.iterdir()):
        raise FileExistsError(...)
```

`PostProcessing.make_compute` passes `output_dir / "sorting_analyzer"` to
`create_sorting_analyzer`, but with `format="zarr"` SpikeInterface writes to
`sorting_analyzer.zarr/`. So after the analyzer is built, the returned path
must be corrected to the real directory, otherwise `make_insert` iterates a
non-existent path and registers zero files in the File part table:

```python
if analyzer_format == "zarr":
    analyzer_output_dir = output_dir / "sorting_analyzer.zarr"
```

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

**New unit tests:**

A new unit test module, `tests/dj_pipeline/utils/test_spike_sorting_utils_unit.py`,
covers the two relocated helpers (see change 3). It tests
`_resolve_analyzer_dir` for the binary-preferred, zarr-fallback, neither-exists,
and both-exist cases, and `_strip_non_numeric_properties` for keeping numeric
properties (float/int/uint/bool) while stripping string properties. These run
under `-m unit` and need no database, since the helpers live in a pure utils
module.

### Verification on HPC

Validated on the SWC HPC against the golden dataset (fresh checkout, DB on
`aeon-db` with a dedicated test prefix, CPU node — the golden suite
force-injects SpikeSorting so no GPU is needed):

- Unit suite: 121 passed (includes the 6 new helper tests).
- Ephys golden integration: 29/29, including `test_recording_zarr_exists`
  and `test_sorting_analyzer_created`.

---

## PR checklist

- [x] Create `es/zarr-intermediates` branch off current main
- [x] Reimplement production code changes (property filter, zarr branching,
      path fallbacks, file registration exclusions)
- [x] Change default from `"binary"` to `"zarr"` in all four
      `params.get("save_format", ...)` calls
- [x] Relocate the pure helpers to `utils/spike_sorting_utils.py` and add
      unit tests
- [x] Add this spec document (`SPEC_ZARR_INTERMEDIATES.md`)
- [x] Update test assertions (recording path, analyzer path)
- [x] Run tests on HPC, verify all pass
- [x] Open PR into main (draft #589)
