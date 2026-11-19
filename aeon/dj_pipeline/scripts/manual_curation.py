"""Apply manual (GUI) curations on SLURM.

A person makes the curation in the GUI, saves it and makes it official
(notebooks/zofia/curations/manual_curation.ipynb in foragingABC_analysis).
slurm/curations/apply_manual_curation.sh then calls slurm_apply_manual(): one block per array task.
"""

from aeon.dj_pipeline import spike_sorting, spike_sorting_curation
from aeon.dj_pipeline.scripts import bombcell_curation as bc


def manual_blocks(*restrictions):
    """Return the PostProcessing blocks whose official curation is a human (not bombcell) curation."""
    human_curated = (spike_sorting_curation.ManualCuration - bc.IS_BOMBCELL).proj()
    official = (spike_sorting_curation.OfficialCuration & human_curated).proj()
    return bc.restrict(spike_sorting.PostProcessing.proj(), restrictions) & official


def slurm_apply_manual() -> None:
    """Apply one manual official curation per array task, make SortedSpikes again, and sync the spikes.

    The list has the blocks that are already applied too, so that it does not change while the
    tasks run. Done blocks finish at once. Environment variables: as bombcell_curation.slurm_curate().
    """
    blocks = manual_blocks(*bc.env_restrictions(*bc.BLOCK_SETTINGS))
    key = bc.pick_task(blocks.keys(order_by=list(bc.BLOCK_FIELDS)))
    if key is not None:
        bc.apply_curation(key)
        bc.check_curated_block(key, (spike_sorting_curation.OfficialCuration & key).fetch1("curation_id"))
