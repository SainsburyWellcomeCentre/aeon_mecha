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
"""

import itertools
import warnings
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import datajoint as dj
import numpy as np
import pandas as pd
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import acquisition, ephys, get_schema_name, spike_sorting
from aeon.dj_pipeline.utils import intervals

if TYPE_CHECKING:
    import pynapple as nap

logger = dj.logger


#: The attributes that identify a probe insertion, shared by every query here.
_INSERTION = ("experiment_name", "subject", "insertion_number")


def _block_tag(ident: tuple) -> str:
    """Render one contributing sorting for ``source_blocks``.

    Carries the whole identity — bounds, electrode config, group, sorting parameter
    set, matching parameter set — so a row says what it drew on, and re-sorting
    under different settings shows up as stale. ``make`` writes these and
    ``stale_chunks`` compares against them, so the two must agree exactly; written
    once so they cannot drift apart, since a silent mismatch makes every row look
    fresh forever.
    """
    start, end, config, group, paramset, matching = ident
    return f"{start}/{end}/{config}/{group}/{paramset}/{matching}"


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
        """Behavioural chunks that a matched ephys chunk overlaps.

        Overlap is half-open: an ephys chunk ending exactly at ``chunk_start``
        belongs to the previous hour. Unmatched blocks do not count — better
        uncomputable than a row with no units in it.
        """
        # Restrict first, rename second. `proj` drops `chunk_start` from the
        # heading, so a semijoin after the rename shares only the insertion
        # attributes and degenerates to "this insertion matched something, somewhere".
        # The rename itself is still load-bearing: a surviving `chunk_start` carries
        # ephys lineage, and populate()'s antijoin is then refused.
        matched = (
            ephys.EphysChunk & (spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk).proj()
        ).proj(eph_start="chunk_start", eph_end="chunk_end")
        overlapping = (acquisition.Chunk * matched) & "eph_start < chunk_end AND eph_end > chunk_start"
        return super().key_source & overlapping

    def make(self, key: dict) -> None:
        """Build one behavioural hour's TsGroup from the blocks covering it."""
        import pynapple as nap  # optional extra; kept lazy so importing this module is cheap

        window = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        insertion = {k: key[k] for k in _INSERTION}

        sortings = _covering_sortings(insertion, window)
        block_chunks = {i: v["chunks"] for i, v in sortings.items()}
        block_units = {i: v["units"] for i, v in sortings.items()}
        if not block_units:
            # populate() retries this key every run, so a silent skip would hide
            # forever. Say it once and move on.
            logger.warning(f"SpikeTrains: no matched block covers {key}, skipping")
            return

        coverage = intervals.coverage(window, [iv for chunks in block_chunks.values() for iv in chunks])
        per_unit = intervals.coverage_by_unit(window, block_chunks, block_units)
        if not per_unit:
            logger.warning(f"SpikeTrains: covering blocks found no units for {key}, skipping")
            return

        bounds = {v["bounds"] for v in sortings.values()}
        spikes_by_unit, counts_by_unit_block = _fetch_spikes(insertion, window, bounds)

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
        metadata = _unit_metadata(insertion, roster, counts_by_unit_block, covered)
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
                "n_units": len(roster),
                "n_spikes": n_spikes,
                "coverage_frac": covered_total / chunk_seconds,
                "n_partial_units": int(sum(c < covered_total for c in covered)),
                "source_blocks": sorted(_block_tag(i) for i in sortings),
                "spikes": tsgroup,
            }
        )

    @classmethod
    def stale_chunks(cls, restriction: Any = True) -> list[dict]:
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
        rows = (cls() & restriction).proj("source_blocks") * acquisition.Chunk.proj("chunk_end")
        stale = []
        for row in rows.to_dicts():
            insertion = {k: row[k] for k in _INSERTION}
            window = (row["chunk_start"], row["chunk_end"])
            sortings = _covering_sortings(insertion, window)
            if sorted(_block_tag(i) for i in sortings) != list(row["source_blocks"]):
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

        At the edges, a chunk the window only partly covers contributes its
        ``covered_seconds`` scaled by the fraction kept. Per-unit coverage is not
        stored per interval, so that assumes a unit's coverage is spread evenly
        through the chunk's — close, not exact.

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
        # Overlap, not chunk_start: a window beginning mid-hour still needs the
        # chunk it starts inside. Half-open at both ends.
        overlapping = acquisition.Chunk & (f'chunk_start < "{_ts(end)}" AND chunk_end > "{_ts(start)}"')
        rows = (cls() & insertion & overlapping).to_dicts()
        if not rows:
            raise ValueError(f"no SpikeTrains rows for {insertion} in [{start}, {end})")

        # A list restriction is an OR in DataJoint, so pass the exact keys already
        # in hand: "any of these rows", which is what the check means.
        span_keys = [{k: r[k] for k in cls.primary_key} for r in rows]
        if not allow_stale and cls.stale_chunks(span_keys):
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
        candidates: dict[int, dict] = {}
        support = []
        window = nap.IntervalSet(start=lo, end=hi)
        for row in sorted(rows, key=lambda r: r["chunk_start"]):
            tsgroup = row["spikes"]
            # restrict() overwrites time_support with the window instead of
            # intersecting, so take the real coverage before it is lost.
            kept = tsgroup.time_support.intersect(window)
            restricted = tsgroup.restrict(window)
            seconds = restricted.get_info("covered_seconds")
            # A window that cuts a chunk keeps only part of its coverage. Scale by
            # the fraction kept: per-unit coverage is not stored per interval, so
            # this assumes a unit's coverage is spread evenly through the chunk's.
            whole = float(tsgroup.time_support.tot_length())
            frac = (float(kept.tot_length()) / whole) if whole else 0.0
            extras = [c for c in restricted.metadata_columns if c not in ("covered_seconds", "rate")]
            for unit in restricted.index:
                u = int(unit)
                times.setdefault(u, []).append(restricted[unit].t)
                covered[u] = covered.get(u, 0.0) + float(seconds[unit]) * frac
                # Chunks can disagree about a unit; resolved after the loop by the
                # same rule make() uses across sortings.
                candidates.setdefault(u, {})[row["chunk_start"]] = (
                    len(restricted[unit]),
                    {c: restricted.get_info(c)[unit] for c in extras},
                )
            support.extend(zip(kept.start, kept.end, strict=True))

        roster = sorted(times)
        data = {u: nap.Ts(t=np.sort(np.concatenate(times[u]))) for u in roster}
        # Merge touching intervals ourselves: pynapple shaves 1e-6 s off the earlier
        # end, which at HARP magnitude is the float64 limit and can drop a spike.
        spans = intervals.merge(sorted(support))
        merged = nap.IntervalSet(start=[s for s, _ in spans], end=[e for _, e in spans])
        carried = {
            u: candidates[u][_most_spikes_wins({cs: n for cs, (n, _) in candidates[u].items()})][1]
            for u in roster
        }
        extra_cols = {c for u in roster for c in carried[u]}
        metadata = {"covered_seconds": np.array([covered[u] for u in roster])}
        missing = {u for u in roster for c in extra_cols if c not in carried[u]}
        if missing:
            raise ValueError(
                f"chunks disagree on metadata columns for units {sorted(missing)}; "
                "repopulate the span so every row carries the same set"
            )
        metadata |= {c: np.array([carried[u][c] for u in roster]) for c in sorted(extra_cols)}
        return nap.TsGroup(data, time_support=merged, metadata=metadata)


def _covering_sortings(insertion: dict, window: tuple) -> dict:
    """Every matched sorting overlapping ``window``, keyed by its real identity.

    A sorting is a block, an electrode group, a sorting parameter set *and* a
    matching parameter set — ``UnitMatching``'s own key. Keying on anything less
    silently drops one of them, and the spikes it found then have no unit to
    belong to. The matching parameter set matters most: global unit ids are handed
    out per insertion across every matching paramset, so two of them describe the
    same neurons with different ids.

    Only sortings ``UnitMatching`` has run for: an unmatched one brings no unit
    identities, so counting its coverage would credit time to nobody.
    """
    overlap = f'block_start < "{_ts(window[1])}" AND block_end > "{_ts(window[0])}"'
    rows = (ephys.EphysBlock * spike_sorting.UnitMatching & insertion & overlap).to_dicts()

    sortings = {}
    for row in rows:
        bounds = (row["block_start"], row["block_end"])
        ident = (
            *bounds,
            row["electrode_config_name"],
            row["electrode_group"],
            row["paramset_id"],
            row["matching_paramset_id"],
        )
        block_key = {k: row[k] for k in (*insertion, "block_start", "block_end")}
        chunks = (ephys.EphysBlockInfo.Chunk * ephys.EphysChunk & block_key).to_dicts()
        group_key = {k: row[k] for k in ("probe_type", "electrode_config_name", "electrode_group")}
        sortings[ident] = {
            "bounds": bounds,
            # EphysBlockInfo links the chunk containing each bound whole, but the
            # sorting only ran inside the block, so credit only the overlap.
            "chunks": intervals.clip([(c["chunk_start"], c["chunk_end"]) for c in chunks], bounds),
            "units": {
                int(u)
                for u in np.atleast_1d((spike_sorting.UnitMatching.Unit & row).to_arrays("global_unit"))
            },
            "config": row["electrode_config_name"],
            "group": row["electrode_group"],
            "matching": row["matching_paramset_id"],
            "electrodes": frozenset(
                int(e)
                for e in np.atleast_1d(
                    (spike_sorting.ElectrodeGroup.Electrode & group_key).to_arrays("electrode")
                )
            ),
        }

    _assert_no_double_counting(sortings)
    _assert_one_config_per_unit(
        {i: s["config"] for i, s in sortings.items()},
        {i: s["units"] for i, s in sortings.items()},
    )
    return sortings


def _fetch_spikes(insertion: dict, window: tuple, bounds: set) -> tuple[dict, dict]:
    """Spike times per unit inside ``window``, and how many came from each block.

    The counts feed ``_most_spikes_wins``, which picks whose metadata wins when
    a unit spans two blocks and each has its own answer.
    """
    lo, hi = np.datetime64(window[0]), np.datetime64(window[1])
    rows = (
        spike_sorting.UnitMatching.Spikes
        & insertion
        & [{"block_start": start, "block_end": end} for start, end in bounds]
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
        # Same identity _covering_sortings uses, so the metadata winner resolves
        # per sorting rather than per block.
        ident = (
            row["block_start"],
            row["block_end"],
            row["electrode_config_name"],
            row["electrode_group"],
            row["paramset_id"],
            row["matching_paramset_id"],
        )
        counts[unit][ident] = counts[unit].get(ident, 0) + len(kept)
    return {u: np.sort(np.concatenate(v)) for u, v in by_unit.items()}, counts


def _assert_no_double_counting(sortings: dict) -> None:
    """Refuse a chunk whose contributing sortings would find the same neuron twice.

    ``UnitMatching`` picks its comparison partners with
    ``(self & insertion & matching_paramset) `` filtered to blocks that overlap in
    time — electrode config, group and sorting parameter set play no part. Its seed
    guard then refuses any block that overlaps nothing already matched, so every
    block under one matching parameter set is linked into one chain and global unit
    ids propagate along it. ``Spikes`` is unique on ``(global_unit, chunk_start)``,
    so whatever shares a matching parameter set cannot be counted twice. That is the
    ordinary multi-block chunk, and most of the multi-sorting ones too.

    Two matching parameter sets are a different story: ids are handed out per
    insertion across all of them, so one neuron picks up an id in each and the chunk
    counts it once per id. Same electrode group under both is provable from the
    identity — the electrodes are identical by definition. Different groups need
    ``ElectrodeGroup.Electrode``, which nothing in the pipeline populates today, so
    an unverifiable pair warns rather than passing silently.
    """
    by_group: dict[tuple, set] = defaultdict(set)
    for _start, _end, config, group, _paramset, matching in sortings:
        by_group[(config, group)].add(matching)
    for (config, group), matching_sets in sorted(by_group.items()):
        if len(matching_sets) > 1:
            raise ValueError(
                f"electrode group {group!r} of config {config!r} covers this chunk under "
                f"matching parameter sets {sorted(matching_sets)}; the same electrodes get a "
                "global unit id under each, so every neuron on them is counted twice."
            )

    unverifiable = 0
    for a, b in itertools.combinations(sorted(sortings), 2):
        if a[5] == b[5]:
            continue  # one matching paramset, so UnitMatching has already linked them
        electrodes_a, electrodes_b = sortings[a]["electrodes"], sortings[b]["electrodes"]
        if not electrodes_a or not electrodes_b:
            unverifiable += 1
            continue
        shared = electrodes_a & electrodes_b
        if shared:
            raise ValueError(
                f"two sortings covering this chunk read the same {len(shared)} electrodes "
                f"under different matching parameter sets: {a} and {b}. Neurons on those "
                "electrodes would be counted once per sorting."
            )

    if unverifiable:
        warnings.warn(
            f"{unverifiable} pair(s) of sortings cover this chunk under different matching "
            "parameter sets, but ElectrodeGroup.Electrode is empty, so electrode overlap "
            "between them could not be checked.",
            stacklevel=3,
        )


def _assert_one_config_per_unit(configs: dict, units: dict) -> None:
    """Refuse a chunk where one global unit appears under two electrode configs.

    A config change alone is fine: the units are different, coverage is already
    per-unit, and ``n_partial_units`` flags it. What is not fine is the same unit
    under both — ``GlobalUnit`` records one physical peak electrode, and the other
    config may never have recorded it.
    """
    if len(set(configs.values())) < 2:
        return
    by_config: dict[str, set] = defaultdict(set)
    for ident, name in configs.items():
        by_config[name] |= units.get(ident, set())
    shared = set.intersection(*by_config.values()) if len(by_config) > 1 else set()
    if shared:
        raise ValueError(
            f"units {sorted(shared)[:5]} appear under more than one electrode config "
            f"({sorted(by_config)}) in this chunk. Their peak electrode is recorded "
            "once and may not have been live under both."
        )
    warnings.warn(
        f"chunk spans electrode configs {sorted(by_config)}; no unit is shared, so "
        "coverage is per-config and source_blocks records which.",
        stacklevel=3,
    )


def _most_spikes_wins(counts: dict):
    """Pick which of several sources speaks for a unit that appears in more than one.

    A unit can show up in several sortings covering one chunk, and in several chunks
    covering one span. Both ask the same question and must answer it the same way,
    so both call this: most spikes wins, and a tie goes to the earliest key. Both
    kinds of key start with a time — a sorting's identity with its block's start, a
    chunk's with its own — so ordering by the key is ordering by that time, and the
    rest of the identity settles blocks that share a start rather than leaving it to
    dict order.
    """
    return min(counts, key=lambda k: (-counts[k], k))


def _unit_metadata(insertion: dict, roster: list, counts_by_unit_block: dict, covered: list) -> dict:
    """Per-unit metadata columns for the TsGroup, in roster order.

    ``covered_seconds`` is each unit's own denominator for a firing rate. Seconds
    rather than a fraction, because fractions stop adding up once ``fetch_span``
    concatenates chunks.

    ``unit_quality`` belongs to a block, so a unit spanning two of them has two
    candidates; ``_most_spikes_wins`` picks the block holding most of its
    spikes. A global unit can map to several local units in that block when a
    merge happened; the lowest local ``unit`` wins, so the answer does not depend
    on row order. ``qc_metrics`` are not flattened in here yet and stay queryable
    on ``SortingQuality.Metric``.
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
        winner = _most_spikes_wins(counts)
        rows = (
            spike_sorting.UnitMatching.Unit * spike_sorting.SortedSpikes.Unit
            & insertion
            & {"global_unit": unit, "block_start": winner[0], "block_end": winner[1]}
        ).to_dicts(order_by="unit")
        quality[unit] = rows[0]["unit_quality"] if rows else "n.a."

    return {
        "covered_seconds": np.array(covered, dtype=float),
        "electrode": np.array([electrodes.get(u, {}).get("electrode", -1) for u in roster]),
        "shank": np.array([electrodes.get(u, {}).get("shank", -1) for u in roster]),
        "unit_quality": np.array([quality[u] for u in roster]),
    }
