"""Spike trains on the same hourly grain as behaviour.

The ephys rig and the behaviour rig cut their data into different chunks, so
asking "what were these neurons doing while the animal was at patch 2?" means
re-deriving the alignment by hand every time. ``SpikeTrains`` does it once: each
row holds one behavioural hour of curated, HARP-synced spikes as a pynapple
``TsGroup``, so spikes and behaviour join on ``(experiment_name, chunk_start)``
and the call site does no time arithmetic.

The table has **no foreign key to the sorted data**, on purpose. Several
``UnitMatching`` rows can cover one behavioural hour, and keying on them would
split the object into fragments for every user, forever. The price is that
nothing invalidates a row when its sorting changes: ``source_blocks`` records
what went into it, and ``stale_chunks()`` finds the rows that have fallen behind.
Full reasoning in ``docs/specs/SPEC_SPIKE_TRAINS.md``.
"""

import warnings
from datetime import datetime
from typing import TYPE_CHECKING

import datajoint as dj
import numpy as np
import pandas as pd
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import acquisition, ephys, get_schema_name, spike_sorting
from aeon.dj_pipeline.utils import rechunk

if TYPE_CHECKING:
    import pynapple as nap

logger = dj.logger


def _ts(value) -> str:
    """Format a datetime for a restriction string.

    DataJoint restrictions are SQL text, so anything a caller hands us goes
    through one known format instead of straight into the query.
    """
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")


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
        """Behavioural chunks that some matched ephys chunk overlaps.

        Overlap is half-open, so an ephys chunk ending exactly at ``chunk_start``
        counts toward the previous behavioural hour. Only blocks that
        ``UnitMatching`` has already run for count. A chunk whose ephys is sorted
        but not yet matched stays uncomputable, which beats writing a row with no
        units in it.
        """
        # Renaming the ephys bounds is load-bearing. A surviving `chunk_start`
        # carries its ephys lineage, and DataJoint then refuses the antijoin that
        # populate() runs against this table. The restriction below is a semijoin,
        # so it already comes back distinct on EphysChunk's key.
        matched = ephys.EphysChunk.proj(eph_start="chunk_start", eph_end="chunk_end") & (
            spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk
        ).proj()
        overlap = "eph_start < chunk_end AND eph_end > chunk_start"
        # A behavioural chunk can overlap several ephys chunks, so the join hands it
        # back once per match. dj.U collapses that to this table's own key.
        return dj.U(*self.primary_key) & ((acquisition.Chunk * matched) & overlap)

    def make(self, key: dict) -> None:
        """Build one behavioural hour's TsGroup from the blocks covering it."""
        import pynapple as nap  # optional extra; kept lazy so importing this module is cheap

        window = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        insertion = {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}

        block_chunks, block_units, block_starts = _covering_blocks(insertion, window)
        if not block_units:
            # populate() retries this key every run, so a silent skip would hide
            # forever. Say it once and move on.
            logger.warning(f"SpikeTrains: no matched block covers {key}, skipping")
            return

        coverage = rechunk.coverage(window, [iv for chunks in block_chunks.values() for iv in chunks])
        per_unit = rechunk.coverage_by_unit(window, block_chunks, block_units)
        if not per_unit:
            logger.warning(f"SpikeTrains: covering blocks found no units for {key}, skipping")
            return

        spikes_by_unit, counts_by_unit_block = _fetch_spikes(insertion, window, block_starts)

        roster = sorted(per_unit)
        data, covered = {}, []
        for unit in roster:
            times = spikes_by_unit.get(unit, np.array([], dtype="datetime64[ns]"))
            seconds = io_api.to_seconds(pd.DatetimeIndex(times)).to_numpy()
            data[int(unit)] = nap.Ts(t=np.sort(seconds))
            covered.append(rechunk.covered_seconds(per_unit[unit]))

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
        covered_total = rechunk.covered_seconds(coverage)
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
    def stale_chunks(cls, restriction=True) -> list[dict]:
        """Find rows whose sorting has changed since they were written.

        Nothing invalidates this table for you, so run this after re-curating, or
        after ``UnitMatching`` covers time that already has rows. Then refresh what
        it finds::

            stale = SpikeTrains.stale_chunks()
            (SpikeTrains & stale).delete()
            SpikeTrains.populate()

        It catches a block matched after the row was written, and upstream rows
        that re-curation removed. Nothing is stored — a stored flag would go stale
        itself. Each covering block costs a couple of queries, so pass a
        ``restriction`` and check one experiment or insertion when you can.
        """
        # Fetch every window in one join instead of one lookup per row.
        rows = (cls() & restriction).proj("source_blocks") * acquisition.Chunk.proj(
            "chunk_end"
        )
        stale = []
        for row in rows.to_dicts():
            insertion = {k: row[k] for k in ("experiment_name", "subject", "insertion_number")}
            window = (row["chunk_start"], row["chunk_end"])
            _, block_units, _ = _covering_blocks(insertion, window)
            if sorted(f"{s}/{e}" for s, e in block_units) != list(row["source_blocks"]):
                stale.append({k: row[k] for k in cls.primary_key})
        return stale

    @classmethod
    def fetch_span(
        cls,
        experiment_name: str,
        subject: str,
        insertion_number: int,
        start: datetime,
        end: datetime,
        allow_stale: bool = False,
    ) -> "nap.TsGroup":
        """Get one TsGroup covering any window, not one per chunk.

        Reach for this whenever the window is not exactly one behavioural hour::

            tg = SpikeTrains.fetch_span("exp-aeon3", "mouse1", 1, start=t0, end=t1)
            good = tg[tg.unit_quality == "good"]
            rate = good.count(0.01)

        Fetching the chunks and stitching them together yourself looks the same and
        is not. This sums each unit's ``covered_seconds``, so firing rates stay
        honest across the join, and it trims each chunk before concatenating, so
        peak memory is one chunk rather than the whole span.

        It raises if any row it needs is stale — pass ``allow_stale=True`` to go
        ahead anyway, or refresh with ``stale_chunks`` first. It warns if a unit was
        sorted for only part of its chunk; divide by ``covered_seconds`` in that
        case, because ``TsGroup.rate`` will use the wrong denominator.

        Size the window before you ask for it. A probe-hour is roughly 115 MB at
        Neuropixels rates, so a day is about 2.7 GB and a week about 19 GB. pynapple
        has no lazy TsGroup, and nothing here will stop you asking for more than
        fits.
        """
        import pynapple as nap

        insertion = {
            "experiment_name": experiment_name,
            "subject": subject,
            "insertion_number": insertion_number,
        }
        rows = (
            cls() & insertion & f'chunk_start >= "{_ts(start)}"' & f'chunk_start < "{_ts(end)}"'
        ).to_dicts()
        if not rows:
            raise ValueError(f"no SpikeTrains rows for {insertion} in [{start}, {end})")

        if not allow_stale:
            stale = {tuple(sorted(k.items())) for k in cls.stale_chunks()}
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
    """Blocks overlapping ``window``: the chunks each covers, the units each found.

    Only blocks ``UnitMatching`` has run for. An unmatched block brings no unit
    identities, so counting its coverage would credit time to nobody.
    """
    overlap = f'block_start < "{_ts(window[1])}" AND block_end > "{_ts(window[0])}"'
    blocks = (ephys.EphysBlock * spike_sorting.UnitMatching & insertion & overlap).to_dicts()

    block_chunks, block_units, block_starts = {}, {}, {}
    for block in blocks:
        # Two blocks can share a start, so block_start alone would silently merge
        # them. EphysBlock's key is (insertion, block_start, block_end).
        ident = (block["block_start"], block["block_end"])
        block_key = {k: block[k] for k in (*insertion, "block_start", "block_end")}
        chunks = (ephys.EphysBlockInfo.Chunk * ephys.EphysChunk & block_key).to_dicts()
        block_chunks[ident] = [(c["chunk_start"], c["chunk_end"]) for c in chunks]
        units = (spike_sorting.UnitMatching.Unit & block).to_arrays("global_unit")
        block_units[ident] = {int(u) for u in np.atleast_1d(units)}
        block_starts[ident] = block["block_start"]
    return block_chunks, block_units, block_starts


def _fetch_spikes(insertion: dict, window: tuple, block_starts: dict) -> tuple[dict, dict]:
    """Spike times per unit inside ``window``, and how many came from each block.

    The counts feed ``rechunk.owning_block``, which picks whose metadata wins when
    a unit spans two blocks and each has its own answer.
    """
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

    ``covered_seconds`` is each unit's own denominator for a firing rate. Seconds
    rather than a fraction, because fractions stop adding up once ``fetch_span``
    concatenates chunks.

    ``unit_quality`` belongs to a block, so a unit spanning two of them has two
    candidates; ``rechunk.owning_block`` picks the block holding most of its
    spikes. ``qc_metrics`` are not flattened in here yet and stay queryable on
    ``SortingQuality.Metric``.
    """
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
