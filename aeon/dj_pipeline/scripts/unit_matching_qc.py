"""Helper functions for QC'ing UnitMatching results.

Covers three things, used interactively from notebooks (e.g.
notebooks/zofia/spike_sorting/unit_matching.ipynb) rather than run as a CLI:

1. Matched-pair retrieval + waveform template lookup, for visually spot-checking matches.
2. Backfilling UnitMatching.BlockComparison for UnitMatching rows computed before that table
   existed (recomputes the same overlap-window comparison spike_sorting.UnitMatching.make()
   runs, but only inserts the full agreement-score grid - it does not touch GlobalUnit,
   UnitMatching.Unit, or UnitMatching.Spikes, which are already correct).
3. Orphan-unit diagnostics: for units that didn't get matched between two blocks, pull
   quality metrics, probe location, whether they even had spikes in the overlap window, and
   their best (possibly sub-threshold) candidate on the other side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import spikeinterface as si
from spikeinterface.core.template_tools import get_template_extremum_channel

from aeon.dj_pipeline import ephys, spike_sorting, spike_sorting_curation

# Fields (besides block_start/block_end) needed to resolve a curated analyzer / SortedSpikes
# row for a block.
BLOCK_KEY_FIELDS = (
    "experiment_name",
    "subject",
    "insertion_number",
    "probe_type",
    "electrode_config_name",
    "electrode_group",
    "paramset_id",
)

_analyzer_cache: dict = {}


# ---------------------------------------------------------------------------
# Matched-pair retrieval + waveform comparison support
# ---------------------------------------------------------------------------


def get_matched_pairs(restriction: dict | None = None) -> pd.DataFrame:
    """One row per global unit matched across (seed block -> later block).

    `restriction` is applied to `UnitMatching.Unit` (e.g. {"electrode_group": "shank0"}).
    Only pairs where the later block's row has a non-null `matched_spike_count` are
    included (i.e. actually matched, not newly-created global units).
    """
    df = (spike_sorting.UnitMatching.Unit & (restriction or {})).to_pandas().reset_index()
    group_cols = ["experiment_name", "subject", "insertion_number", "matching_paramset_id", "global_unit"]
    pairs = []
    for _, group in df.groupby(group_cols):
        if len(group) < 2:
            continue
        group = group.sort_values("block_start")
        seed = group.iloc[0]
        for _, child in group.iloc[1:].iterrows():
            if pd.isna(child["matched_spike_count"]):
                continue
            row = {f"{k}_a": seed[k] for k in ("block_start", "block_end", "unit")}
            row.update({f"{k}_b": child[k] for k in ("block_start", "block_end", "unit")})
            row.update({k: child[k] for k in BLOCK_KEY_FIELDS})
            row["global_unit"] = child["global_unit"]
            row["matching_paramset_id"] = child["matching_paramset_id"]
            row["matched_spike_count"] = int(child["matched_spike_count"])
            pairs.append(row)
    return pd.DataFrame(pairs)


def block_key(row, side: str) -> dict:
    """Build a block key from a `get_matched_pairs` row for side "a" or "b"."""
    key = {k: row[k] for k in BLOCK_KEY_FIELDS}
    key["block_start"] = row[f"block_start_{side}"]
    key["block_end"] = row[f"block_end_{side}"]
    return key


def select_pairs_for_review(pairs_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Spread a sample evenly across the sorted range of matched_spike_count (not random)."""
    ordered = pairs_df.sort_values("matched_spike_count").reset_index(drop=True)
    idx = np.linspace(0, len(ordered) - 1, min(n, len(ordered))).round().astype(int)
    return ordered.iloc[np.unique(idx)]


def load_analyzer(block_key_: dict):
    """Load (and cache) the curated SortingAnalyzer for a block."""
    cache_key = tuple(sorted((k, str(v)) for k, v in block_key_.items()))
    if cache_key not in _analyzer_cache:
        curation_id = (spike_sorting_curation.OfficialCuration & block_key_).fetch1("curation_id")
        file_key = {**block_key_, "curation_id": curation_id, "file_name": "curation_applied_analyzer"}
        path = str((spike_sorting_curation.ManualCuration.File & file_key).fetch1("file"))
        analyzer = si.load_sorting_analyzer(folder=path, load_extensions=False)
        _analyzer_cache[cache_key] = analyzer
    return _analyzer_cache[cache_key]


def get_unit_template(block_key_: dict, unit_id: int):
    """Return (waveform at peak channel, peak channel id, sampling_frequency)."""
    analyzer = load_analyzer(block_key_)
    if analyzer.get_extension("templates") is None:
        analyzer.load_extension("templates")
    template = analyzer.get_extension("templates").get_unit_template(unit_id)  # (n_samples, n_channels)
    peak_channel = get_template_extremum_channel(analyzer)[unit_id]
    ch_idx = list(analyzer.channel_ids).index(peak_channel)
    return template[:, ch_idx], peak_channel, analyzer.sampling_frequency


def get_unit_quality_metrics(block_key_: dict) -> pd.DataFrame:
    """Full quality_metrics table (index=unit_id) for a block's curated analyzer."""
    analyzer = load_analyzer(block_key_)
    if analyzer.get_extension("quality_metrics") is None:
        analyzer.load_extension("quality_metrics")
    return analyzer.get_extension("quality_metrics").get_data()


def get_unit_locations(block_key_: dict) -> pd.DataFrame:
    """Estimated (x, y[, z]) probe location per unit for a block's curated analyzer."""
    analyzer = load_analyzer(block_key_)
    if analyzer.get_extension("unit_locations") is None:
        analyzer.load_extension("unit_locations")
    locations = analyzer.get_extension("unit_locations").get_data()
    columns = ["x", "y", "z"][: locations.shape[1]]
    return pd.DataFrame(locations, index=analyzer.unit_ids, columns=columns)


def get_unit_manual_labels(block_key_: dict) -> pd.DataFrame:
    """Manual curation quality label + custom tags (index=unit_id) for a block.

    Read directly from the curated analyzer's `sorting` properties - these are the labels
    actually assigned by hand in the SpikeInterface GUI (`quality`: good/MUA/noise, plus one
    boolean property per tag in the CurationTag lookup). This is NOT the same thing as
    `SortedSpikes.Unit.unit_quality`, which is Kilosort's own KSLabel and doesn't reflect
    manual curation at all (it reads "good" for units manually labeled MUA/noise too).
    Curation also doesn't remove MUA/noise units - they stay in SortedSpikes.Unit.

    Missing properties (e.g. a block whose curation never touched these label_definitions)
    come back as None rather than raising.
    """
    analyzer = load_analyzer(block_key_)
    sorting = analyzer.sorting
    prop_keys = set(sorting.get_property_keys())

    data = {"unit": analyzer.unit_ids}
    data["manual_quality"] = sorting.get_property("quality") if "quality" in prop_keys else None
    for tag in spike_sorting.CurationTag.fetch("tag"):
        data[tag] = sorting.get_property(tag) if tag in prop_keys else None
    return pd.DataFrame(data).set_index("unit")


# ---------------------------------------------------------------------------
# BlockComparison backfill
# ---------------------------------------------------------------------------


def backfill_block_comparisons(key: dict) -> int:
    """Populate UnitMatching.BlockComparison for an already-computed UnitMatching row.

    Recomputes the same overlap-window spike-time comparison `UnitMatching.make()` runs
    (using that row's own UnitMatchingParamSet.params, same as make() does), against every
    previously-matched block that overlaps it, and stores the full agreement-score grid -
    skipping any comparison already present, so this is safe to call repeatedly for the same
    key. Does not touch GlobalUnit, UnitMatching.Unit, or UnitMatching.Spikes.

    Returns the number of new BlockComparison rows inserted.
    """
    insertion_key = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}
    paramset_key = {"matching_paramset_id": key["matching_paramset_id"]}
    block_start, block_end = (ephys.EphysBlock & key).fetch1("block_start", "block_end")

    raw_params = (spike_sorting.UnitMatchingParamSet & key).fetch1("params") or {}
    params = {**spike_sorting._DEFAULT_MATCHING_PARAMS, **raw_params}

    all_matched = (spike_sorting.UnitMatching & insertion_key & paramset_key).to_dicts()
    overlapping_blocks = [
        b
        for b in all_matched
        if b["block_start"] != key["block_start"]
        and b["block_start"] < block_end
        and b["block_end"] > block_start
    ]
    if not overlapping_blocks:
        return 0

    this_block_units = spike_sorting._load_block_unit_spike_trains(key)
    if not this_block_units:
        return 0

    existing_matched_starts = {
        e["matched_block_start"]
        for e in (spike_sorting.UnitMatching.BlockComparison & key).to_dicts()
    }

    new_rows = []
    for prev_block in overlapping_blocks:
        if prev_block["block_start"] in existing_matched_starts:
            continue
        prev_units = spike_sorting._load_block_unit_spike_trains(prev_block)
        if not prev_units:
            continue
        comparison = spike_sorting._compare_spike_trains_in_overlap(
            this_block_units,
            prev_units,
            (block_start, block_end),
            (prev_block["block_start"], prev_block["block_end"]),
            delta_time=params["delta_time"],
            match_score=params["match_score"],
            min_score=params["min_score"],
        )
        if comparison is None:
            continue
        new_rows.append(
            {
                **key,
                "matched_block_start": prev_block["block_start"],
                "matched_block_end": prev_block["block_end"],
                "unit_ids": np.asarray(comparison.agreement_scores.columns),
                "matched_block_unit_ids": np.asarray(comparison.agreement_scores.index),
                "agreement_scores": comparison.agreement_scores.to_numpy(),
            }
        )

    if new_rows:
        spike_sorting.UnitMatching.BlockComparison.insert(
            new_rows, ignore_extra_fields=True, allow_direct_insert=True
        )
    return len(new_rows)


# ---------------------------------------------------------------------------
# Orphan-unit diagnostics
# ---------------------------------------------------------------------------

_QC_METRIC_COLUMNS = ("presence_ratio", "isi_violations_ratio", "snr", "firing_rate", "num_spikes")


def _best_candidate(unit_id: int, is_prev_side: bool, later_block_key: dict, earlier_block_start) -> dict:
    """Best-scoring candidate for a unit, read from UnitMatching.BlockComparison's score grid.

    `later_block_key` is the block whose UnitMatching row did the comparison (always the later
    of the two blocks, since the grid is stored under "this block" = the block make() was
    processing). `is_prev_side=True` means `unit_id` belongs to the earlier (matched-against)
    block, so it's a row of the grid; `False` means it belongs to the later block, so it's a
    column. Returns the highest-scoring opposite-side unit and that score (None if no grid or
    the unit isn't in it).
    """
    none_result = {"best_candidate_unit": None, "best_candidate_score": None}
    grid = (
        spike_sorting.UnitMatching.BlockComparison
        & later_block_key
        & {"matched_block_start": earlier_block_start}
    )
    if not grid:
        return none_result
    row = grid.fetch1()
    scores = np.asarray(row["agreement_scores"])              # [matched_block_unit_ids x unit_ids]
    this_units = np.asarray(row["unit_ids"])                  # columns (later block)
    matched_units = np.asarray(row["matched_block_unit_ids"])  # rows (earlier block)
    if scores.size == 0:
        return none_result

    if is_prev_side:
        idx = np.where(matched_units == unit_id)[0]
        if len(idx) == 0:
            return none_result
        score_vec, other_units = scores[idx[0], :], this_units
    else:
        idx = np.where(this_units == unit_id)[0]
        if len(idx) == 0:
            return none_result
        score_vec, other_units = scores[:, idx[0]], matched_units

    if len(score_vec) == 0:
        return none_result
    best_i = int(np.argmax(score_vec))
    return {
        "best_candidate_unit": int(other_units[best_i]),
        "best_candidate_score": float(score_vec[best_i]),
    }


def _annotate_orphans(
    orphan_unit_ids: list[int],
    block_key_: dict,
    other_block_key_: dict,
    later_block_key: dict,
    earlier_block_start,
    is_prev_side: bool,
) -> pd.DataFrame:
    tag_options = list(spike_sorting.CurationTag.fetch("tag"))
    if not orphan_unit_ids:
        return pd.DataFrame(
            columns=[
                "unit",
                "unit_quality",
                "manual_quality",
                *tag_options,
                *_QC_METRIC_COLUMNS,
                "x",
                "y",
                "overlap_window_spike_count",
                "best_candidate_unit",
                "best_candidate_score",
            ]
        )

    quality_map = dict(
        (e["unit"], e["unit_quality"])
        for e in (spike_sorting.SortedSpikes.Unit & block_key_).proj("unit_quality").to_dicts()
    )
    qc_metrics = get_unit_quality_metrics(block_key_)
    locations = get_unit_locations(block_key_)
    manual_labels = get_unit_manual_labels(block_key_)

    this_trains = spike_sorting._load_block_unit_spike_trains(block_key_)
    overlap_start = max(block_key_["block_start"], other_block_key_["block_start"])
    overlap_end = min(block_key_["block_end"], other_block_key_["block_end"])
    overlap_start_s = overlap_start.replace(tzinfo=spike_sorting.UTC).timestamp()
    overlap_end_s = overlap_end.replace(tzinfo=spike_sorting.UTC).timestamp()

    rows = []
    for unit_id in orphan_unit_ids:
        row = {"unit": unit_id, "unit_quality": quality_map.get(unit_id)}
        if unit_id in qc_metrics.index:
            row.update({col: qc_metrics.loc[unit_id, col] for col in _QC_METRIC_COLUMNS})
        else:
            row.update(dict.fromkeys(_QC_METRIC_COLUMNS))
        if unit_id in locations.index:
            row["x"] = locations.loc[unit_id, "x"]
            row["y"] = locations.loc[unit_id, "y"]
        else:
            row["x"] = row["y"] = None
        if unit_id in manual_labels.index:
            row["manual_quality"] = manual_labels.loc[unit_id, "manual_quality"]
            for tag in tag_options:
                row[tag] = manual_labels.loc[unit_id, tag]
        else:
            row["manual_quality"] = None
            row.update(dict.fromkeys(tag_options))
        overlap_times = spike_sorting._restrict_to_overlap(
            this_trains.get(unit_id, np.array([])), overlap_start_s, overlap_end_s
        )
        row["overlap_window_spike_count"] = len(overlap_times)
        row.update(_best_candidate(unit_id, is_prev_side, later_block_key, earlier_block_start))
        rows.append(row)
    return pd.DataFrame(rows)


def _resolve_shank_blocks(shank_key: dict, matching_paramset_id: int):
    """Common setup shared by `get_orphan_units` and `get_unit_match_summary`.

    Assumes exactly two matched blocks for this (shank, paramset) - the seed block ("A",
    earlier) and the block matched against it ("B", later).

    Returns (a_rows, b_rows, block_a_key, block_b_key, matched_global_units).
    """
    paramset_key = {"matching_paramset_id": matching_paramset_id}
    unit_rows = (spike_sorting.UnitMatching.Unit & shank_key & paramset_key).to_pandas().reset_index()
    block_starts = sorted(unit_rows["block_start"].unique())
    if len(block_starts) != 2:
        raise ValueError(f"Assumes exactly 2 matched blocks; found {len(block_starts)}: {block_starts}")
    block_a_start, block_b_start = block_starts

    a_rows = unit_rows[unit_rows["block_start"] == block_a_start]
    b_rows = unit_rows[unit_rows["block_start"] == block_b_start]

    def _full_block_key(rows_df, start) -> dict:
        first = rows_df.iloc[0]
        key = {k: first[k] for k in BLOCK_KEY_FIELDS}
        key["block_start"] = start
        key["block_end"] = first["block_end"]
        return key

    block_a_key = _full_block_key(a_rows, block_a_start)
    block_b_key = _full_block_key(b_rows, block_b_start)
    matched_global_units = set(b_rows.loc[b_rows["matched_spike_count"].notna(), "global_unit"])
    return a_rows, b_rows, block_a_key, block_b_key, matched_global_units


def get_orphan_units(shank_key: dict, matching_paramset_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A-side and B-side units that didn't get matched across a shank's two blocks.

    Call `backfill_block_comparisons` for the later block's UnitMatching key first if
    BlockComparison hasn't been populated for it yet, otherwise best_candidate_* columns will
    be empty.

    Returns (a_orphans, b_orphans):
    - a_orphans: block A units whose global_unit was never carried forward into block B.
    - b_orphans: block B units that got a brand-new global_unit (matched_spike_count null).

    Each row includes: unit_quality (Kilosort's own KSLabel - NOT your manual SI GUI curation
    call), quality_metrics (presence_ratio, isi_violations_ratio, snr, firing_rate,
    num_spikes), estimated probe location (x, y), spike count restricted to just the overlap
    window (0 means it structurally couldn't have been matched), and its best candidate on
    the other side + that candidate's agreement_score (even if below the match threshold).
    """
    a_rows, b_rows, block_a_key, block_b_key, matched_global_units = _resolve_shank_blocks(
        shank_key, matching_paramset_id
    )
    a_orphan_ids = a_rows.loc[~a_rows["global_unit"].isin(matched_global_units), "unit"].tolist()
    b_orphan_ids = b_rows.loc[b_rows["matched_spike_count"].isna(), "unit"].tolist()

    a_orphans = _annotate_orphans(
        a_orphan_ids, block_a_key, block_b_key,
        later_block_key=block_b_key, earlier_block_start=block_a_key["block_start"], is_prev_side=True,
    )
    b_orphans = _annotate_orphans(
        b_orphan_ids, block_b_key, block_a_key,
        later_block_key=block_b_key, earlier_block_start=block_a_key["block_start"], is_prev_side=False,
    )
    return a_orphans, b_orphans


def get_unit_match_summary(shank_key: dict, matching_paramset_id: int) -> pd.DataFrame:
    """Per-unit best-candidate summary for *every* unit in a shank's two matched blocks.

    Same columns as `get_orphan_units`, but covers matched units too (tagged via the
    `is_matched` column) instead of only the unmatched ones - useful for comparing
    best_candidate_score (or any other per-unit metric) across the whole population, not
    just the orphans.
    """
    a_rows, b_rows, block_a_key, block_b_key, matched_global_units = _resolve_shank_blocks(
        shank_key, matching_paramset_id
    )
    a_matched_ids = set(a_rows.loc[a_rows["global_unit"].isin(matched_global_units), "unit"])
    b_matched_ids = set(b_rows.loc[b_rows["matched_spike_count"].notna(), "unit"])

    a_summary = _annotate_orphans(
        a_rows["unit"].tolist(), block_a_key, block_b_key,
        later_block_key=block_b_key, earlier_block_start=block_a_key["block_start"], is_prev_side=True,
    )
    a_summary["side"] = "A"
    a_summary["is_matched"] = a_summary["unit"].isin(a_matched_ids)

    b_summary = _annotate_orphans(
        b_rows["unit"].tolist(), block_b_key, block_a_key,
        later_block_key=block_b_key, earlier_block_start=block_a_key["block_start"], is_prev_side=False,
    )
    b_summary["side"] = "B"
    b_summary["is_matched"] = b_summary["unit"].isin(b_matched_ids)

    return pd.concat([a_summary, b_summary], ignore_index=True)
