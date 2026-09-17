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
                "source_blocks": sorted(f"{start}/{end}" for start, end in block_starts),
                "spikes": tsgroup,
            }
        )

    @classmethod
    def stale(cls) -> list[dict]:
        """Rows built from a set of blocks that is no longer the current one.

        This is what stands in for the cascade the table gives up by keying on the
        join target rather than the data source. It catches both directions: a block
        matched *after* a row was written, and upstream rows deleted by re-curation.

        Computed, never stored — a stored flag would itself go stale. Returns keys,
        so ``SpikeTrains() & SpikeTrains.stale()`` is the rows to delete.
        """
        stale_keys = []
        for row in cls().to_dicts():
            window = (acquisition.Chunk & row).fetch1("chunk_start", "chunk_end")
            insertion = {k: row[k] for k in ("experiment_name", "subject", "insertion_number")}
            _, block_units, _ = _covering_blocks(insertion, window)
            if sorted(f"{s}/{e}" for s, e in block_units) != list(row["source_blocks"]):
                stale_keys.append({k: row[k] for k in cls.primary_key})
        return stale_keys

    @classmethod
    def fetch_span(
        cls,
        experiment_name: str,
        subject: str,
        insertion_number: int,
        start,
        end,
        allow_stale: bool = False,
    ):
        """Spike trains over an arbitrary window, as one TsGroup.

        The documented entry point. Restricts each chunk *before* concatenating, so
        peak memory is one chunk rather than the whole span — which matters because
        pynapple reads a TsGroup whole.

        Raises on a stale contributing row and warns on partial coverage. The
        asymmetry is deliberate: stale is out of date and cheaply fixed, while
        partial coverage is a permanent fact about the data that a caller works
        around.
        """
        import warnings

        import numpy as np
        import pynapple as nap
        from swc.aeon.io import api as io_api

        insertion = {
            "experiment_name": experiment_name,
            "subject": subject,
            "insertion_number": insertion_number,
        }
        rows = (cls() & insertion & f'chunk_start >= "{start}"' & f'chunk_start < "{end}"').to_dicts()
        if not rows:
            raise ValueError(f"no SpikeTrains rows for {insertion} in [{start}, {end})")

        if not allow_stale:
            stale = {tuple(sorted(k.items())) for k in cls.stale()}
            if any(tuple(sorted({k: r[k] for k in cls.primary_key}.items())) in stale for r in rows):
                raise ValueError("span covers stale rows; delete and repopulate, or pass allow_stale=True")

        partial = [r["chunk_start"] for r in rows if r["n_partial_units"]]
        if partial:
            warnings.warn(
                f"{len(partial)} chunk(s) have units sorted for only part of the chunk; "
                f"use covered_seconds, not TsGroup.rate — first at {partial[0]}",
                stacklevel=2,
            )

        lo, hi = io_api.to_seconds(start), io_api.to_seconds(end)
        times: dict[int, list] = {}
        covered: dict[int, float] = {}
        support = []
        for row in sorted(rows, key=lambda r: r["chunk_start"]):
            tsgroup = row["spikes"]
            window = nap.IntervalSet(start=lo, end=hi)
            restricted = tsgroup.restrict(window)
            seconds = restricted.get_info("covered_seconds")
            for unit in restricted.index:
                times.setdefault(int(unit), []).append(restricted[unit].t)
                covered[int(unit)] = covered.get(int(unit), 0.0) + float(seconds[unit])
            support.extend(zip(restricted.time_support.start, restricted.time_support.end, strict=True))

        roster = sorted(times)
        data = {u: nap.Ts(t=np.sort(np.concatenate(times[u]))) for u in roster}
        merged = nap.IntervalSet(start=[s for s, _ in support], end=[e for _, e in support])
        return nap.TsGroup(
            data,
            time_support=merged,
            metadata={"covered_seconds": np.array([covered[u] for u in roster])},
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
        # EphysBlock's key is (insertion, block_start, block_end) — two blocks can
        # share a start, so keying on block_start alone silently collapses them.
        ident = (block["block_start"], block["block_end"])
        block_key = {k: block[k] for k in (*insertion, "block_start", "block_end")}
        chunks = (ephys.EphysBlockInfo.Chunk * ephys.EphysChunk & block_key).to_dicts()
        block_chunks[ident] = [(c["chunk_start"], c["chunk_end"]) for c in chunks]
        units = (spike_sorting.UnitMatching.Unit & block).to_arrays("global_unit")
        block_units[ident] = {int(u) for u in np.atleast_1d(units)}
        block_starts[ident] = block["block_start"]
    return block_chunks, block_units, block_starts


def _fetch_spikes(insertion: dict, window: tuple, block_starts: dict) -> tuple[dict, dict]:
    """Spike times per global unit, clipped to ``window``, plus per (unit, block) counts.

    The counts feed ``rechunk.owning_block``, which decides whose block-scoped
    metadata wins for a unit spanning more than one block.
    """
    import numpy as np

    lo, hi = np.datetime64(window[0]), np.datetime64(window[1])
    rows = (
        spike_sorting.UnitMatching.Spikes
        & insertion
        & [{"block_start": start, "block_end": end} for start, end in block_starts]
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
        ident = (row["block_start"], row["block_end"])
        counts[unit][ident] = counts[unit].get(ident, 0) + len(kept)
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
            & {"global_unit": unit, "block_start": winner[0], "block_end": winner[1]}
        ).to_dicts()
        quality[unit] = rows[0]["unit_quality"] if rows else "n.a."

    return {
        "covered_seconds": np.array(covered, dtype=float),
        "electrode": np.array([electrodes.get(u, {}).get("electrode", -1) for u in roster]),
        "shank": np.array([electrodes.get(u, {}).get("shank", -1) for u in roster]),
        "unit_quality": np.array([quality[u] for u in roster]),
    }
