"""Curate spike-sorted blocks with bombcell, then match units across blocks.

foragingABC_analysis runs this module on SLURM, one item per array task:
- slurm/curations/bombcell_curation.sh calls slurm_curate(): one block per task.
- slurm/curations/unit_matching.sh calls slurm_match(): one shank per task (bombcell or manual).
scripts/manual_curation.py uses the block and SLURM helpers of this module.

Rules:
- Bombcell gives only three labels: good, mua and noise (label_non_somatic=False).
- A bombcell ManualCuration has a description that starts with BOMBCELL_DESCRIPTION_PREFIX.
  Each other ManualCuration is a human (manual) curation.
- The pipeline keeps noise units. The `exclude_noise` matching parameter sets if they are matched.
"""

import os

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.curation as sc
from spikeinterface.curation.curation_format import validate_curation_dict

from aeon.dj_pipeline import spike_sorting, spike_sorting_curation

logger = spike_sorting_curation.logger

BOMBCELL_LABELS = ("good", "mua", "noise")
BOMBCELL_DESCRIPTION_PREFIX = "Bombcell automated labelling"
BOMBCELL_DESCRIPTION = (
    f"{BOMBCELL_DESCRIPTION_PREFIX} (default thresholds, label_non_somatic=False, "
    f"spikeinterface {si.__version__})"
)
IS_BOMBCELL = f"description LIKE '{BOMBCELL_DESCRIPTION_PREFIX}%'"
POPULATE_KWARGS = dict(reserve_jobs=True, suppress_errors=False, display_progress=True)
BLOCK_FIELDS = ("block_start", "experiment_name", "insertion_number", "electrode_group", "paramset_id")


# ---- Labelling ----


def label_units(sorting_analyzer: si.SortingAnalyzer) -> pd.DataFrame:
    """Label each unit good, mua or noise with the default bombcell thresholds.

    Returns a DataFrame with the column `bombcell_label`, indexed by unit_id in analyzer order.
    """
    labels = sc.bombcell_label_units(
        sorting_analyzer,
        thresholds=sc.bombcell_get_default_thresholds(),
        label_non_somatic=False,
    )
    labels = labels.loc[list(sorting_analyzer.unit_ids), ["bombcell_label"]]
    labels["bombcell_label"] = labels["bombcell_label"].str.lower()
    labels.index.name = "unit_id"

    unexpected = set(labels["bombcell_label"]) - set(BOMBCELL_LABELS)
    if unexpected:
        raise ValueError(f"bombcell returned unexpected label(s) {unexpected}; expected only {BOMBCELL_LABELS}")
    return labels


# ---- Saving as a curation ----


def build_curation_dict(labels: pd.DataFrame) -> dict:
    """Make a SpikeInterface curation dict (format_version 2): one "quality" label per unit, no edits."""
    unit_ids = [int(u) for u in labels.index]
    curation_dict = {
        "format_version": "2",
        "unit_ids": unit_ids,
        "label_definitions": {"quality": {"label_options": list(BOMBCELL_LABELS), "exclusive": True}},
        "manual_labels": [
            {"unit_id": unit_id, "labels": {"quality": [label]}}
            for unit_id, label in zip(unit_ids, labels["bombcell_label"], strict=True)
        ],
        "removed": [],
        "merges": [],
        "splits": [],
    }
    validate_curation_dict(curation_dict)
    return curation_dict


def get_bombcell_curation_id(key: dict) -> int | None:
    """Return the curation_id of the bombcell curation of a block (any spikeinterface version), or None."""
    existing = (spike_sorting_curation.ManualCuration & key & IS_BOMBCELL).to_arrays("curation_id")
    return int(max(existing)) if len(existing) else None


def save_curation(key: dict, labels: pd.DataFrame) -> int:
    """Save the bombcell labels as a new ManualCuration of the block. Return its curation_id.

    If the block already has a bombcell curation, nothing is written and its id is returned.
    """
    existing_id = get_bombcell_curation_id(key)
    if existing_id is not None:
        logger.info(f"Bombcell curation already saved (curation_id={existing_id}), skipping.")
        return existing_id
    curation_id = spike_sorting_curation.insert_curation(key, build_curation_dict(labels), BOMBCELL_DESCRIPTION)
    logger.info(f"Saved bombcell curation_id={curation_id} ({len(labels)} units)")
    return curation_id


# ---- Whole-block pipeline ----


def restrict(table, restrictions):
    for restriction in restrictions:
        table &= restriction
    return table


def bombcell_blocks(*restrictions):
    """Return the PostProcessing blocks to curate with bombcell: those without a human curation."""
    human_curated = spike_sorting_curation.ManualCuration - IS_BOMBCELL
    return restrict(spike_sorting.PostProcessing.proj(), restrictions) - human_curated.proj()


def apply_curation(key: dict) -> None:
    """Apply the official curation of a block, make SortedSpikes again, and sync the spikes."""
    spike_sorting_curation.ApplyOfficialCuration.populate(key, **POPULATE_KWARGS)
    spike_sorting.SortedSpikes.populate(key, **POPULATE_KWARGS)
    spike_sorting.SyncedSpikes.populate(key, **POPULATE_KWARGS)


def curate_block(key: dict) -> int:
    """Label a block with bombcell, make the curation official and apply it. Return its curation_id.

    Each step skips the work that is already done.
    """
    curation_id = get_bombcell_curation_id(key)
    if curation_id is None:
        # No extensions: bombcell loads the ones it needs.
        analyzer = si.load_sorting_analyzer(
            spike_sorting_curation._get_analyzer_dir_from_key(key), load_extensions=False
        )
        curation_id = save_curation(key, label_units(analyzer))
    if not spike_sorting.SortedSpikes & key:
        # make_curation_official needs the raw SortedSpikes (curation_id=-1).
        spike_sorting.SortedSpikes.populate(key, **POPULATE_KWARGS)
    spike_sorting_curation.make_curation_official(key, curation_id)
    apply_curation(key)
    return curation_id


# ---- Unit matching ----


def matching_blocks(experiment_name: str, insertion_number: int, electrode_group: str, paramset_id: str):
    """Return the blocks of one shank and sorting paramset with an applied curation and synced spikes."""
    return (
        spike_sorting.SyncedSpikes.proj()
        & spike_sorting_curation.ApplyOfficialCuration.proj()
        & {
            "experiment_name": experiment_name,
            "insertion_number": insertion_number,
            "electrode_group": electrode_group,
            "paramset_id": paramset_id,
        }
    )


def matching_paramset_description(
    experiment_name: str,
    insertion_number: int,
    electrode_group: str,
    paramset_id: str,
    curation: str = "bombcell",
    exclude_noise: bool = False,
) -> str:
    """Return the description that identifies the matching paramset of one shank.

    Do not change the format: get_or_create_matching_paramset finds existing paramsets by it.
    """
    return (
        f"{curation} | {experiment_name} insertion {insertion_number} | {electrode_group} | "
        f"sorting paramset {paramset_id} | exclude_noise={exclude_noise}"
    )


def get_or_create_matching_paramset(
    experiment_name: str,
    insertion_number: int,
    electrode_group: str,
    paramset_id: str,
    curation: str = "bombcell",
    exclude_noise: bool = False,
) -> int:
    """Return the matching_paramset_id of the shank. Make the paramset if it does not exist.

    A new paramset seeds on the first block in time and uses the default matching parameters,
    except `exclude_noise`. `curation` ("bombcell" or "manual") goes only into the description.
    Each shank needs its own paramset: UnitMatching.make() compares all overlapping blocks of a
    paramset and does not check the shank.
    """
    description = matching_paramset_description(
        experiment_name, insertion_number, electrode_group, paramset_id, curation, exclude_noise
    )
    paramset = spike_sorting.UnitMatchingParamSet & {"matching_paramset_description": description}
    existing = paramset.to_arrays("matching_paramset_id")
    if len(existing):
        return int(existing[0])

    blocks = matching_blocks(experiment_name, insertion_number, electrode_group, paramset_id)
    if not blocks:
        raise ValueError(f"No curated + synced blocks to seed matching on: {description}")
    seed_block_start = blocks.keys(order_by="block_start", limit=1)[0]["block_start"]

    existing_ids = spike_sorting.UnitMatchingParamSet.to_arrays("matching_paramset_id")
    matching_paramset_id = int(max(existing_ids)) + 1 if len(existing_ids) else 1
    spike_sorting.UnitMatchingParamSet.insert1(
        {
            "matching_paramset_id": matching_paramset_id,
            "matching_method": "spike_time_overlap",
            "seed_block_start": seed_block_start,
            "matching_paramset_description": description,
            "params": {"exclude_noise": exclude_noise},
        }
    )
    return matching_paramset_id


def run_unit_matching(block_restriction, matching_paramset_id: int) -> tuple[int, list]:
    """Populate UnitMatching until a round matches no block. Return the matched block count and the errors.

    Each populate() matches one block (the seed or the next frontier block), so it runs in a loop.
    Do not run two of these on one insertion at the same time: both take the same next GlobalUnit id.
    """
    n_matched = 0
    while True:
        result = spike_sorting.UnitMatching.populate(
            block_restriction,
            {"matching_paramset_id": matching_paramset_id},
            display_progress=True,
            suppress_errors=True,
        )
        n_matched += result["success_count"]
        if result["success_count"] == 0:
            return n_matched, result["error_list"]  # the errors of the last round stopped the matching


def match_shank(
    experiment_name: str,
    insertion_number: int,
    electrode_group: str,
    paramset_id: str,
    curation: str = "bombcell",
    exclude_noise: bool = False,
) -> list[str]:
    """Match the units of one shank across its blocks, then update GlobalUnitQuality. Return the problems."""
    shank = (experiment_name, insertion_number, electrode_group, paramset_id)
    ready = matching_blocks(*shank)
    ready_starts = ready.to_arrays("block_start")
    oldest = min(
        (spike_sorting.PostProcessing & {"experiment_name": experiment_name, "paramset_id": paramset_id}).to_arrays(
            "block_start"
        )
    )
    n_blocks = len(spike_sorting.PostProcessing & dict(zip(BLOCK_FIELDS[1:], shank, strict=True)))
    print(f"{len(ready)}/{n_blocks} block(s) are curated and synced. Matching stops at the first gap.")
    # Check before get_or_create_matching_paramset: a paramset made now would seed on a later block.
    if not len(ready_starts) or min(ready_starts) != oldest:
        return [f"The oldest block of {experiment_name} ({oldest}) is not curated and synced on this shank."]

    matching_paramset_id = get_or_create_matching_paramset(
        *shank, curation=curation, exclude_noise=exclude_noise
    )
    paramset_key = {"matching_paramset_id": matching_paramset_id}
    print(f"matching_paramset_id={matching_paramset_id}", flush=True)
    n_new, errors = run_unit_matching(ready, matching_paramset_id)
    n_done = len(spike_sorting.UnitMatching & ready & paramset_key)
    print(f"Matched {n_new} new block(s); {n_done}/{len(ready)} ready block(s) are matched.")

    # .proj(): GlobalUnit has a free-text column without lineage, and DataJoint 2.x will not join on it.
    global_units = (
        spike_sorting.GlobalUnit
        & {"experiment_name": experiment_name, "insertion_number": insertion_number}
        & paramset_key
    ).proj()
    spike_sorting.GlobalUnitQuality.refresh(global_units, display_progress=True)
    qualities = (spike_sorting.GlobalUnitQuality & global_units).to_arrays("unit_quality")
    print(f"{len(global_units)} global unit(s), majority labels: {_counts(qualities)}")

    problems = [f"{key['block_start']}: {error}" for key, error in errors]
    if n_done < len(ready):
        problems.append(f"Only {n_done}/{len(ready)} ready block(s) are matched.")
    return problems


# ---- SLURM entry points ----
# The SLURM scripts call these with `python -c`. They read the settings from environment variables.
# Without SLURM_ARRAY_TASK_ID (list mode), they print the task list and change no data.


def env_restrictions(*names: str) -> list:
    """Return DB restrictions from the environment. EXPERIMENT is an SQL LIKE pattern and is required."""
    restrictions = [f"experiment_name LIKE '{os.environ['EXPERIMENT']}'"]
    for name in names:
        value = os.environ.get(name, "")
        if not value:
            continue
        if name == "BLOCK_START_FROM":
            restrictions.append(f"block_start >= '{value}'")
        elif name == "BLOCK_START_TO":
            restrictions.append(f"block_start <= '{value}'")
        else:  # paramset_id is varchar (403, cjh_405), so the value stays a string
            restrictions.append({name.lower(): value})
    return restrictions


def _describe(item: dict) -> str:
    return "  ".join(str(item[f]) for f in BLOCK_FIELDS if f in item)


def _counts(values) -> dict:
    return {str(v): int(n) for v, n in zip(*np.unique(values, return_counts=True), strict=True)}


def pick_task(items: list[dict]) -> dict | None:
    """Return the item of this array task (task N gets item N), or None in list mode or past the end."""
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        print(f"{len(items)} task(s). Submit with --array=1-{len(items)}")
        for i, item in enumerate(items, start=1):
            print(f"  [{i:>3}] {_describe(item)}")
        return None
    if int(task) > len(items):
        print(f"Task {task}: there are only {len(items)} task(s). Nothing to do.")
        return None
    item = items[int(task) - 1]
    print(f"Task {task}/{len(items)}: {_describe(item)}", flush=True)
    return item


BLOCK_SETTINGS = ("ELECTRODE_GROUP", "PARAMSET_ID", "BLOCK_START_FROM", "BLOCK_START_TO")


def slurm_curate() -> None:
    """Curate one block per array task with bombcell. Task N curates block N, in block_start order.

    Label, save the curation, make it official, apply it, make SortedSpikes again, and sync the spikes.
    Environment variables (empty means no restriction): EXPERIMENT (required, SQL LIKE),
    ELECTRODE_GROUP, PARAMSET_ID, BLOCK_START_FROM, BLOCK_START_TO (both inclusive).
    """
    key = pick_task(bombcell_blocks(*env_restrictions(*BLOCK_SETTINGS)).keys(order_by=list(BLOCK_FIELDS)))
    if key is not None:
        check_curated_block(key, curate_block(key))


def check_curated_block(key: dict, curation_id: int) -> None:
    """Print the unit labels. Raise if SortedSpikes is not the curated one or the spikes are not synced."""
    sorted_ids = list((spike_sorting.SortedSpikes & key).to_arrays("curation_id"))
    qualities = (spike_sorting.SortedSpikes.Unit & key).to_arrays("unit_quality")
    print(f"curation_id={curation_id}, unit labels: {_counts(qualities)}")
    if sorted_ids != [curation_id]:
        raise RuntimeError(f"SortedSpikes has curation_id {sorted_ids}, expected [{curation_id}].")
    if not spike_sorting.SyncedSpikes & key:
        raise RuntimeError("SyncedSpikes is missing.")


def slurm_match() -> None:
    """Match the units of one shank and sorting paramset per array task.

    Run one task at a time (--array=...%1): GlobalUnit ids are counted per insertion,
    so two runs on one insertion take the same id.

    Environment variables (empty means no restriction):
    - EXPERIMENT (required, SQL LIKE), PARAMSET_ID, INSERTION_NUMBER, ELECTRODE_GROUP.
    - CURATION: "bombcell" (default) or "manual". It goes into the matching paramset description.
    - EXCLUDE_NOISE: "True" or "False". Default: False for bombcell, True for manual.
    """
    curation = os.environ.get("CURATION", "bombcell")
    exclude_noise = {"True": True, "False": False}[os.environ.get("EXCLUDE_NOISE") or str(curation == "manual")]
    blocks = restrict(
        spike_sorting.PostProcessing.proj(), env_restrictions("PARAMSET_ID", "INSERTION_NUMBER", "ELECTRODE_GROUP")
    )
    shank_fields = BLOCK_FIELDS[1:]
    shanks = sorted({tuple(key[f] for f in shank_fields) for key in blocks.keys()})
    shank = pick_task([dict(zip(shank_fields, s, strict=True)) for s in shanks])
    if shank is None:
        return

    print(f"curation={curation}, exclude_noise={exclude_noise}")
    problems = match_shank(**shank, curation=curation, exclude_noise=exclude_noise)
    if problems:
        raise RuntimeError("\n".join(problems))
