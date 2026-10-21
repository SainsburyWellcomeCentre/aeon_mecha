from aeon.dj_pipeline import spike_sorting
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir
import shutil

exp = {"experiment_name": "ProjectAeonOVC-2026-09-14_001"}
failed = (spike_sorting.SpikeSorting.proj() - spike_sorting.PostProcessing.proj()) & exp
# print(len(failed))                                   # expect 4
print(len(spike_sorting.PostProcessing.jobs.errors & exp))   # should be 0
root = get_sorting_root_dir()
for rel in (spike_sorting.PreProcessing & failed.proj()).fetch("sorting_output_dir"):
    d = root / rel / "sorting_analyzer.zarr"
    print(d, d.exists())
    # shutil.rmtree(d)   # uncomment after checking the 4 paths

(spike_sorting.PostProcessing.jobs.errors & exp).delete()     # clear the 4 error rows
print(spike_sorting.PostProcessing.jobs.reserved & exp)       # should be empty