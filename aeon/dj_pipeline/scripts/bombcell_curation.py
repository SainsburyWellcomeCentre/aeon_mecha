"""Curate spike-sorted ephys blocks with bombcell, then match units across blocks.

This module does these steps:
1. Bombcell labels each unit of a block as good, mua or noise.
2. The labels become the ManualCuration of the block.
3. The curation becomes official, and the pipeline applies it.
4. Unit matching links the units across the blocks of one shank.

Per block:
    labels = label_units(load_analyzer(key))          # bombcell labels
    curation_id = save_curation(key, labels)          # ManualCuration and curation JSON
    curate_block(key)                                 # steps 1-3, then SortedSpikes and SyncedSpikes

Per shank and sorting paramset, after all its blocks are curated and synced:
    blocks = matching_blocks(experiment_name, insertion_number, electrode_group, paramset_id)
    matching_paramset_id = get_or_create_matching_paramset(experiment_name, insertion_number,
                                                           electrode_group, paramset_id)
    run_unit_matching(blocks, matching_paramset_id)
    spike_sorting.GlobalUnitQuality.refresh(...)      # majority label of each global unit

Rules:
- Bombcell gives only three labels: good, mua and noise.
- `label_non_somatic=False` turns off the non-somatic labels of bombcell.
- The description of a bombcell ManualCuration starts with BOMBCELL_DESCRIPTION_PREFIX.
- The pipeline keeps noise units. It does not delete them.
- The `exclude_noise` matching parameter controls if noise units take part in matching.
"""

import json
from datetime import UTC, datetime

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


# ---- Labelling ----


def load_analyzer(key: dict) -> si.SortingAnalyzer:
    """Load the raw sorting analyzer of a block.

    This function does not load the extensions. Bombcell loads the metric extensions that it needs.
    """
    analyzer_dir = spike_sorting_curation._get_analyzer_dir_from_key(key)
    return si.load_sorting_analyzer(folder=analyzer_dir, load_extensions=False)


def label_units(sorting_analyzer: si.SortingAnalyzer) -> pd.DataFrame:
    """Label each unit as good, mua or noise. Bombcell uses its default thresholds.

    Returns a DataFrame with one column, `bombcell_label`.
    The index is unit_id, in the order of `sorting_analyzer.unit_ids`.
    Raises ValueError if bombcell returns a different label.
    """
    labels = sc.bombcell_label_units(
        sorting_analyzer,
        thresholds=sc.bombcell_get_default_thresholds(),
        label_non_somatic=False,  # only good, mua and noise
    )
    labels = labels.loc[list(sorting_analyzer.unit_ids), ["bombcell_label"]]
    labels["bombcell_label"] = labels["bombcell_label"].str.lower()
    labels.index.name = "unit_id"

    unexpected = set(labels["bombcell_label"]) - set(BOMBCELL_LABELS)
    if unexpected:
        raise ValueError(
            f"bombcell returned unexpected label(s) {unexpected}; expected only {BOMBCELL_LABELS}"
        )
    return labels


# ---- Saving as a curation ----


def build_curation_dict(labels: pd.DataFrame) -> dict:
    """Make a SpikeInterface curation dict (format_version 2) from the output of label_units.

    Each unit gets one exclusive "quality" label.
    The curation does not remove, merge or split units.
    """
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
    """Return the curation_id of the bombcell curation of the block, or None.

    The search uses the description prefix.
    Thus it also finds curations from other spikeinterface versions.
    """
    bombcell_curations = spike_sorting_curation.ManualCuration & key
    bombcell_curations &= f"description LIKE '{BOMBCELL_DESCRIPTION_PREFIX}%'"
    existing = bombcell_curations.to_arrays("curation_id")
    return int(max(existing)) if len(existing) else None


def save_curation(key: dict, labels: pd.DataFrame) -> int:
    """Save the bombcell labels as a new ManualCuration of the block. Return its curation_id.

    The function writes:
    - the curation JSON at <analyzer>/spikeinterface_gui/curation_data_id<N>.json
    - one ManualCuration row
    - one ManualCuration.File row

    This is the same layout as `spike_sorting_curation.save_manual_curation()`.
    The function does not change a GUI curation that is in progress.
    If the block already has a bombcell curation, the function writes nothing.
    Then it returns the existing curation_id.
    """
    existing_id = get_bombcell_curation_id(key)
    if existing_id is not None:
        logger.info(f"Bombcell curation already saved (curation_id={existing_id}), skipping.")
        return existing_id

    curation_dict = build_curation_dict(labels)

    # The ManualCuration insert needs the full primary key (subject, probe_type, ...).
    full_key = (spike_sorting.SpikeSorting & key).fetch1("KEY")

    existing_ids = (spike_sorting_curation.ManualCuration & full_key).to_arrays("curation_id")
    curation_id = int(max(existing_ids)) + 1 if len(existing_ids) else 1

    gui_dir = spike_sorting_curation._get_analyzer_dir_from_key(full_key) / "spikeinterface_gui"
    gui_dir.mkdir(exist_ok=True)
    file_name = f"curation_data_id{curation_id}.json"
    file_path = gui_dir / file_name
    with open(file_path, "w") as f:
        json.dump(curation_dict, f, indent=4)

    # One transaction saves the master row and its File part together, or saves neither.
    with spike_sorting_curation.ManualCuration.connection.transaction:
        spike_sorting_curation.ManualCuration.insert1(
            {
                **full_key,
                "curation_id": curation_id,
                "curation_datetime": datetime.now(UTC),
                "parent_curation_id": -1,  # the labels come from the raw sorting
                "curation_method": "SpikeInterface",
                "description": BOMBCELL_DESCRIPTION,
            }
        )
        spike_sorting_curation.ManualCuration.File.insert1(
            {**full_key, "curation_id": curation_id, "file_name": file_name, "file": file_path}
        )

    logger.info(f"Saved bombcell curation_id={curation_id} ({len(labels)} units) to {file_path}")
    return curation_id


# ---- Whole-block pipeline ----


def bombcell_blocks(*restrictions):
    """Return the PostProcessing blocks to curate with bombcell.

    The blocks match all the restrictions.
    The function removes the blocks that have a human curation.
    A human curation is a ManualCuration that does not come from bombcell.
    """
    human_curated = spike_sorting_curation.ManualCuration - (
        f"description LIKE '{BOMBCELL_DESCRIPTION_PREFIX}%'"
    )
    blocks = spike_sorting.PostProcessing.proj()
    for restriction in restrictions:
        blocks &= restriction
    return blocks - human_curated.proj()


def curate_block(key: dict) -> int:
    """Curate one block with bombcell from start to end. Return its curation_id.

    The function does these steps:
    1. Label the units and save the curation.
    2. Make the curation official.
    3. Apply the curation.
    4. Make SortedSpikes again from the curated analyzer.
    5. Sync the spikes.

    The function skips each step that is already done.
    """
    curation_id = get_bombcell_curation_id(key)
    if curation_id is None:
        curation_id = save_curation(key, label_units(load_analyzer(key)))
    spike_sorting_curation.make_curation_official(key, curation_id)

    populate_kwargs = dict(reserve_jobs=True, suppress_errors=False, display_progress=True)
    spike_sorting_curation.ApplyOfficialCuration.populate(key, **populate_kwargs)
    spike_sorting.SortedSpikes.populate(key, **populate_kwargs)
    spike_sorting.SyncedSpikes.populate(key, **populate_kwargs)
    return curation_id


# ---- Unit matching ----


def matching_blocks(experiment_name: str, insertion_number: int, electrode_group: str, paramset_id: int):
    """Return the blocks of one shank and sorting paramset that are ready to match.

    A block is ready when its curation is applied and its spikes are synced.
    """
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
    paramset_id: int,
    curation: str = "bombcell",
    exclude_noise: bool = False,
) -> str:
    """Return the description that identifies the matching paramset of one shank.

    get_or_create_matching_paramset uses this description to find the paramset again.
    """
    return (
        f"{curation} | {experiment_name} insertion {insertion_number} | {electrode_group} | "
        f"sorting paramset {paramset_id} | exclude_noise={exclude_noise}"
    )


def get_or_create_matching_paramset(
    experiment_name: str,
    insertion_number: int,
    electrode_group: str,
    paramset_id: int,
    curation: str = "bombcell",
    exclude_noise: bool = False,
) -> int:
    """Return the matching_paramset_id of the shank. Make the paramset if it does not exist.

    A new paramset uses:
    - the first block in time as the seed block
    - the standard matching parameters
    - the `exclude_noise` value from the arguments

    `curation` ("bombcell" or "manual") goes only into the description.

    Each shank and sorting paramset needs its own matching paramset.
    UnitMatching.make() compares a block with all overlapping blocks of the same paramset.
    It does not check the shank.
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
    seed_block_start = blocks.keys(order_by="block_start", limit=1)[0]["block_start"]  # first block in time

    existing_ids = spike_sorting.UnitMatchingParamSet.to_arrays("matching_paramset_id")
    matching_paramset_id = int(max(existing_ids)) + 1 if len(existing_ids) else 1
    spike_sorting.UnitMatchingParamSet.insert1(
        {
            "matching_paramset_id": matching_paramset_id,
            "matching_method": "spike_time_overlap",
            "seed_block_start": seed_block_start,
            "matching_paramset_description": description,
            # The other parameters use spike_sorting._DEFAULT_MATCHING_PARAMS.
            "params": {"exclude_noise": exclude_noise},
        }
    )
    return matching_paramset_id


def run_unit_matching(block_restriction, matching_paramset_id: int) -> tuple[int, list]:
    """Populate UnitMatching until a round matches no block. Return the matched block count and the errors.

    Each populate() call matches only one block: the seed block or the next block.
    Thus the function calls populate() in a loop.

    Do not run two of these on the same insertion at the same time.
    Both runs try to use the same next GlobalUnit id, and one run fails.
    """
    n_matched, errors = 0, []
    while True:
        result = spike_sorting.UnitMatching.populate(
            block_restriction,
            {"matching_paramset_id": matching_paramset_id},
            display_progress=True,
            suppress_errors=True,
        )
        n_matched += result["success_count"]
        errors = result["error_list"]  # the errors of the last round stopped the matching
        if result["success_count"] == 0:
            return n_matched, errors
