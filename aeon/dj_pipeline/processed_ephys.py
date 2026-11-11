"""Spike trains on the same hourly grain as behaviour.

The two rigs chunk their data differently, so each row re-derives the alignment
once: one behavioural hour of curated, HARP-synced spikes as a pynapple
``TsGroup``, joinable with behaviour on ``(experiment_name, chunk_start)``.

**No foreign key to the sorted data**, on purpose: several ``UnitMatching`` rows
can cover one behavioural hour, and keying on them would fragment the object for
every user, forever. The price is that nothing invalidates a row when its sorting
changes. ``source_blocks`` records what went in, so a row that has fallen behind
can be found again.
"""

from datetime import UTC, datetime

import datajoint as dj
import numpy as np
import pandas as pd
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import acquisition, ephys, get_schema_name, spike_sorting
from aeon.dj_pipeline.utils import intervals

logger = dj.logger


# ``UnitMatching``'s own key for one sorting, less the insertion. Two of the fields
# identify a block, and a block can hold more than one sorting.
_SORTING_FIELDS = (
    "block_start",
    "block_end",
    "electrode_config_name",
    "electrode_group",
    "paramset_id",
    "matching_paramset_id",
)


def _sorting_key(row: dict) -> tuple:
    """One sorting's key, in one fixed order."""
    return tuple(row[f] for f in _SORTING_FIELDS)


schema = dj.Schema(get_schema_name("processed_ephys"))


@schema
class SpikeTrains(dj.Computed):
    definition = """
    # Curated, HARP-synced spike trains for one behavioural chunk, as a pynapple TsGroup
    -> acquisition.Chunk                 # experiment_name, chunk_start (BEHAVIOURAL grain)
    -> ephys.ProbeInsertion              # subject, insertion_number
    ---
    n_units: int32                       # units in the roster, including silent ones
    n_spikes: int64                      # total spikes across all units
    coverage_frac: float32               # covered seconds / (chunk_end - chunk_start)
    n_partial_units: int32               # units sorted for less than the full chunk; 0 normally
    source_blocks: json                  # one string per sorting that fed this row
    execution_time: datetime             # when make() started; later matching leaves this row stale
    spikes: <pynapple@dj_store>          # TsGroup; HARP seconds since 1904-01-01
    """

    @property
    def key_source(self):
        """Behavioural chunks that a matched ephys chunk overlaps.

        Half-open: an ephys chunk ending exactly at ``chunk_start`` belongs to the
        previous hour. Unmatched blocks do not count.
        """
        matched = (
            ephys.EphysChunk & (spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk).proj()
        ).proj(eph_start="chunk_start", eph_end="chunk_end")
        overlapping = (acquisition.Chunk * matched) & "eph_start < chunk_end AND eph_end > chunk_start"
        return super().key_source & overlapping

    def make(self, key: dict) -> None:
        """Build one behavioural hour's TsGroup from the blocks covering it."""
        import pynapple as nap  # optional extra; kept lazy so importing this module is cheap

        # Taken before the first read, so anything matched later is provably newer
        # than what went into this row.
        started = datetime.now(UTC)

        window = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        insertion = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}

        sortings = _covering_sortings(insertion, window)
        block_chunks = {i: v["chunks"] for i, v in sortings.items()}
        block_units = {i: v["units"] for i, v in sortings.items()}
        if not block_units:
            # populate() retries this key every run; a silent skip would hide forever.
            logger.warning(f"SpikeTrains: no matched block covers {key}, skipping")
            return

        coverage = intervals.coverage(window, [iv for chunks in block_chunks.values() for iv in chunks])
        per_unit = intervals.coverage_by_unit(window, block_chunks, block_units)
        if not per_unit:
            logger.warning(f"SpikeTrains: covering blocks found no units for {key}, skipping")
            return

        bounds = {v["bounds"] for v in sortings.values()}
        spikes_by_unit = _fetch_spikes(insertion, window, bounds)

        roster = sorted(per_unit)
        data, covered = {}, []
        for unit in roster:
            times = spikes_by_unit.get(unit, np.array([], dtype="datetime64[ns]"))
            seconds = io_api.to_seconds(pd.DatetimeIndex(times)).to_numpy()
            data[int(unit)] = nap.Ts(t=np.sort(seconds))
            covered.append(intervals.covered_seconds(per_unit[unit]))

        support = nap.IntervalSet(
            start=[io_api.to_seconds(s) for s, _ in coverage],
            end=[io_api.to_seconds(e) for _, e in coverage],
        )
        metadata = _unit_metadata(insertion, roster, covered)
        tsgroup = nap.TsGroup(data, time_support=support, metadata=metadata)

        n_spikes = int(sum(len(tsgroup[u]) for u in tsgroup.index))
        expected = int(sum(len(v) for v in spikes_by_unit.values()))
        if n_spikes != expected:
            raise ValueError(f"lost spikes assembling {key}: {n_spikes} kept of {expected}")
        if tsgroup.time_support.start[0] <= 3.0e9:
            raise ValueError(f"times are not on the HARP epoch for {key}")

        chunk_seconds = (window[1] - window[0]).total_seconds()
        covered_total = intervals.covered_seconds(coverage)
        self.insert1(
            {
                **key,
                "execution_time": started,
                "n_units": len(roster),
                "n_spikes": n_spikes,
                "coverage_frac": covered_total / chunk_seconds,
                "n_partial_units": int(sum(c < covered_total for c in covered)),
                "source_blocks": sorted("/".join(map(str, k)) for k in sortings),
                "spikes": tsgroup,
            }
        )


def _covering_sortings(insertion: dict, window: tuple) -> dict:
    """Every matched sorting overlapping ``window``, keyed by ``_sorting_key``.

    That key is ``UnitMatching``'s own; anything less silently drops a
    sorting and its spikes lose the unit they belong to. The matching parameter set
    matters most — global unit ids are handed out per insertion across all of them,
    so two paramsets describe the same neurons with different ids.

    Matched sortings only: an unmatched one credits coverage to nobody.
    """
    # DJ restrictions are SQL text, so the window goes in formatted, not as datetimes.
    lo, hi = (pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f") for t in window)
    overlap = f'block_start < "{hi}" AND block_end > "{lo}"'
    rows = (ephys.EphysBlock * spike_sorting.UnitMatching & insertion & overlap).to_dicts()

    sortings = {}
    for row in rows:
        bounds = (row["block_start"], row["block_end"])
        sorting_key = _sorting_key(row)
        block_key = {k: row[k] for k in (*insertion, "block_start", "block_end")}
        chunks = (ephys.EphysBlockInfo.Chunk * ephys.EphysChunk & block_key).to_dicts()
        sortings[sorting_key] = {
            "bounds": bounds,
            # Whole chunks, not the block window. The sorter concatenates each linked
            # chunk file entire, so the unit was observed for all of it; clipping to
            # the block would shrink the rate denominator and leave every spike from
            # the chunk in place. The block says which files to sort, not when the
            # rig was recording.
            "chunks": [(c["chunk_start"], c["chunk_end"]) for c in chunks],
            "units": {
                int(u)
                for u in np.atleast_1d((spike_sorting.UnitMatching.Unit & row).to_arrays("global_unit"))
            },
        }

    return sortings


def _fetch_spikes(insertion: dict, window: tuple, bounds: set) -> dict:
    """Spike times per unit inside ``window``, from the given blocks."""
    lo, hi = np.datetime64(window[0]), np.datetime64(window[1])
    rows = (
        spike_sorting.UnitMatching.Spikes
        & insertion
        & [{"block_start": start, "block_end": end} for start, end in bounds]
    ).to_dicts()

    by_unit: dict[int, list] = {}
    for row in rows:
        times = np.asarray(row["spike_times"], dtype="datetime64[ns]")
        kept = times[(times >= lo) & (times < hi)]  # half-open
        if len(kept):
            by_unit.setdefault(int(row["global_unit"]), []).append(kept)
    return {u: np.sort(np.concatenate(v)) for u, v in by_unit.items()}


def _unit_metadata(insertion: dict, roster: list, covered: list) -> dict:
    """Per-unit metadata columns for the TsGroup, in roster order.

    ``covered_seconds`` is each unit's rate denominator — seconds, not a fraction,
    because fractions stop adding up once chunks are concatenated over a window.
    Quality labels and ``qc_metrics`` are block-scoped and stay upstream.
    """
    electrodes = {
        int(r["global_unit"]): r
        for r in (spike_sorting.GlobalUnit * ephys.ProbeType.Electrode & insertion).to_dicts()
    }
    return {
        "covered_seconds": np.array(covered, dtype=float),
        "electrode": np.array([electrodes.get(u, {}).get("electrode", -1) for u in roster]),
        "shank": np.array([electrodes.get(u, {}).get("shank", -1) for u in roster]),
    }
