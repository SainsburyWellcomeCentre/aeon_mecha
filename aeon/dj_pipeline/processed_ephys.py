"""Analysis-facing ephys products on the behavioural chunk grain.

``SpikeTrains`` re-chunks curated, HARP-synced spike trains from the ephys rig's
``EphysChunk`` grain to ``acquisition.Chunk``, so spikes and behaviour join on
``(experiment_name, chunk_start)`` with no time arithmetic at the call site.

The table has **no foreign key to the sorted data**. That is deliberate: a
behavioural hour can be covered by several ``UnitMatching`` rows, so keying on them
would fragment the object for every user, forever. The cost is that nothing
invalidates a row automatically — ``source_blocks`` records what went in and
``SpikeTrains.stale()`` finds rows whose inputs have moved on. See
``docs/specs/SPEC_SPIKE_TRAINS.md``.
"""

import datajoint as dj
import numpy as np

from aeon.dj_pipeline import acquisition, ephys, get_schema_name, spike_sorting

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
    source_blocks: json                  # contributing EphysBlock starts + matching paramset
    spikes: <pynapple@dj_store>          # TsGroup; HARP seconds since 1904-01-01
    """

    @property
    def key_source(self):
        """Behavioural chunks that some matched ephys chunk overlaps, per insertion.

        Half-open overlap: an ephys chunk ending exactly at ``chunk_start`` belongs
        to the previous behavioural chunk. Restricted to blocks ``UnitMatching`` has
        run for, so a chunk whose ephys is sorted but unmatched stays uncomputable
        rather than producing a row with no units.

        The ephys bounds are projected onto fresh names *and* the ephys
        ``chunk_start`` is dropped, so nothing shares that attribute name with
        ``acquisition.Chunk``. Aliasing alone is not enough — the surviving
        ``chunk_start`` keeps its ephys lineage and DataJoint then refuses the
        antijoin ``populate()`` runs against this table.
        """
        matched_chunks = (
            ephys.EphysChunk.proj(eph_start="chunk_start", eph_end="chunk_end")
            & (spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk).proj()
        )
        ephys_windows = (
            dj.U("experiment_name", "subject", "insertion_number", "eph_start", "eph_end") & matched_chunks
        )
        overlap = "eph_start < chunk_end AND eph_end > chunk_start"
        return dj.U("experiment_name", "chunk_start", "subject", "insertion_number") & (
            (acquisition.Chunk * ephys_windows) & overlap
        )

    def make(self, key: dict) -> None:
        """Re-chunk one behavioural chunk's worth of spikes into a TsGroup."""
        import numpy as np
        import pandas as pd
        import pynapple as nap
        from swc.aeon.io import api as io_api

        from aeon.dj_pipeline.utils import rechunk

        window = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        insertion = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}

        block_chunks, block_units, block_starts = _covering_blocks(insertion, window)
        if not block_units:
            return

        coverage = rechunk.chunk_coverage(window, [iv for chunks in block_chunks.values() for iv in chunks])
        per_unit = rechunk.unit_coverage(window, block_chunks, block_units)
        if not per_unit:
            return

        spikes_by_unit, counts_by_unit_block = _fetch_spikes(insertion, window, block_starts)

        roster = sorted(per_unit)
        data, covered = {}, []
        for unit in roster:
            times = spikes_by_unit.get(unit, np.array([], dtype="datetime64[ns]"))
            seconds = io_api.to_seconds(pd.DatetimeIndex(times)).to_numpy()
            data[int(unit)] = nap.Ts(t=np.sort(seconds))
            covered.append(rechunk.total_seconds(per_unit[unit]))

        support = nap.IntervalSet(
            start=[io_api.to_seconds(s) for s, _ in coverage],
            end=[io_api.to_seconds(e) for _, e in coverage],
        )
        metadata = _unit_metadata(insertion, roster, counts_by_unit_block, block_starts, covered)
        tsgroup = nap.TsGroup(data, time_support=support, metadata=metadata)

        n_spikes = int(sum(len(tsgroup[u]) for u in tsgroup.index))
        expected = int(sum(len(v) for v in spikes_by_unit.values()))
        if n_spikes != expected:
            raise ValueError(f"lost spikes assembling {key}: {n_spikes} kept of {expected}")
        if tsgroup.time_support.start[0] <= 3.0e9:
            raise ValueError(f"times are not on the HARP epoch for {key}")

        chunk_seconds = (window[1] - window[0]).total_seconds()
        covered_total = rechunk.total_seconds(coverage)
        self.insert1(
            {
                **key,
                "n_units": len(roster),
                "n_spikes": n_spikes,
                "coverage_frac": covered_total / chunk_seconds,
                "n_partial_units": int(sum(c < covered_total for c in covered)),
                "source_blocks": sorted(str(b) for b in block_starts),
                "spikes": tsgroup,
            }
        )


def _covering_blocks(insertion: dict, window: tuple) -> tuple[dict, dict, dict]:
    """Blocks overlapping ``window``, the ephys chunks each covers, the units each found.

    Keyed by ``block_start``. Restricted to blocks ``UnitMatching`` has run for —
    an unmatched block contributes no unit identities, so including it would produce
    coverage with nothing to attribute it to.
    """
    overlap = f'block_start < "{window[1]}" AND block_end > "{window[0]}"'
    blocks = (ephys.EphysBlock * spike_sorting.UnitMatching & insertion & overlap).to_dicts()

    block_chunks, block_units, block_starts = {}, {}, {}
    for block in blocks:
        start = block["block_start"]
        block_key = {k: block[k] for k in (*insertion, "block_start", "block_end")}
        chunks = (ephys.EphysBlockInfo.Chunk * ephys.EphysChunk & block_key).to_dicts()
        block_chunks[start] = [(c["chunk_start"], c["chunk_end"]) for c in chunks]
        units = (spike_sorting.UnitMatching.Unit & block).to_arrays("global_unit")
        block_units[start] = {int(u) for u in np.atleast_1d(units)}
        block_starts[start] = start
    return block_chunks, block_units, block_starts


def _fetch_spikes(insertion: dict, window: tuple, block_starts: dict) -> tuple[dict, dict]:
    """Spike times per global unit, clipped to ``window``, plus per (unit, block) counts.

    The counts feed ``rechunk.owning_block``, which decides whose block-scoped
    metadata wins for a unit spanning more than one block.
    """
    import numpy as np

    lo, hi = np.datetime64(window[0]), np.datetime64(window[1])
    rows = (
        spike_sorting.UnitMatching.Spikes & insertion & [{"block_start": b} for b in block_starts]
    ).to_dicts()

    by_unit: dict[int, list] = {}
    counts: dict[int, dict] = {}
    for row in rows:
        times = np.asarray(row["spike_times"], dtype="datetime64[ns]")
        kept = times[(times >= lo) & (times < hi)]  # half-open
        if not len(kept):
            continue
        unit = int(row["global_unit"])
        by_unit.setdefault(unit, []).append(kept)
        counts.setdefault(unit, {})
        counts[unit][row["block_start"]] = counts[unit].get(row["block_start"], 0) + len(kept)
    return {u: np.sort(np.concatenate(v)) for u, v in by_unit.items()}, counts


def _unit_metadata(
    insertion: dict, roster: list, counts_by_unit_block: dict, block_starts: dict, covered: list
) -> dict:
    """Per-unit metadata columns for the TsGroup, in roster order.

    ``covered_seconds`` is the honest denominator for each unit's firing rate —
    seconds rather than a fraction, because fractions do not compose when
    ``fetch_span`` concatenates chunks.

    ``unit_quality`` is block-scoped, so a unit spanning blocks has more than one
    candidate; ``rechunk.owning_block`` picks the one holding most of its spikes.
    Flattened ``qc_metrics`` are not carried yet — they remain queryable on
    ``SortingQuality.Metric``.
    """
    import numpy as np

    from aeon.dj_pipeline.utils import rechunk

    electrodes = {
        int(r["global_unit"]): r
        for r in (spike_sorting.GlobalUnit * ephys.ProbeType.Electrode & insertion).to_dicts()
    }
    quality = {}
    for unit in roster:
        counts = counts_by_unit_block.get(unit)
        if not counts:
            quality[unit] = "n.a."
            continue
        winner = rechunk.owning_block(counts, block_starts)
        rows = (
            spike_sorting.UnitMatching.Unit * spike_sorting.SortedSpikes.Unit
            & insertion
            & {"global_unit": unit, "block_start": winner}
        ).to_dicts()
        quality[unit] = rows[0]["unit_quality"] if rows else "n.a."

    return {
        "covered_seconds": np.array(covered, dtype=float),
        "electrode": np.array([electrodes.get(u, {}).get("electrode", -1) for u in roster]),
        "shank": np.array([electrodes.get(u, {}).get("shank", -1) for u in roster]),
        "unit_quality": np.array([quality[u] for u in roster]),
    }
