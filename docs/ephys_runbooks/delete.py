from aeon.dj_pipeline import spike_sorting as ss
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir, scratch_recording_dir

key = {"experiment_name": "ProjectAeonOVC-2026-09-28_001", "paramset_id": "250"}   # the interrupted paramset only
root = get_sorting_root_dir()
for k in ((ss.PreProcessing & key).proj() - ss.SpikeSorting.proj()).to_dicts():   # preprocessed, not sorted
    out = root / (ss.PreProcessing & k).fetch1("sorting_output_dir")
    zarr = scratch_recording_dir(out.parent / "recording") / "recording.zarr"
    print(k["block_start"], k["electrode_group"], "| partial sort:", (out / "spike_sorting").exists(),
          "| recording.zarr:", zarr.exists(), "|", out / "spike_sorting")
print(ss.SpikeSorting.jobs.reserved & key)
print(ss.SpikeSorting.jobs.errors & key)
