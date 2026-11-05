"""When each unit was actually observed, as half-open ``(start, end)`` datetimes.

Blocks start and end mid-hour and the rig stops recording, so a unit is observed only
where a block that found it was running. Those seconds are what a firing rate has to
divide by, and working them out is all this module does.

Datetimes, not the float seconds pynapple wants, and the merging happens here rather
than inside ``IntervalSet``. Two ephys chunks that meet exactly — one ends where the
next starts — are one unbroken stretch of recording, but pynapple pulls the earlier
end back by 1e-6 to keep its intervals disjoint, so they never join and a solid hour
reports gaps it does not have. Merging first keeps that out of the way: ``IntervalSet``
only ever sees intervals already separated by real gaps, and "these meet" stays an
exact datetime comparison rather than a tolerance someone has to choose.
"""

from collections import defaultdict
from collections.abc import Hashable
from datetime import datetime

Interval = tuple[datetime, datetime]


def clip(intervals: list[Interval], window: Interval) -> list[Interval]:
    """Trim ``intervals`` to what falls inside ``window``."""
    lo, hi = window
    out = []
    for start, end in intervals:
        s, e = max(start, lo), min(end, hi)
        if s < e:  # half-open: zero width means no overlap
            out.append((s, e))
    return out


def merge(intervals: list[Interval]) -> list[Interval]:
    """Join intervals that overlap or touch, keeping real gaps as gaps.

    The gaps are the point: they let a chunk report a broken ``time_support``
    instead of pretending it recorded throughout.
    """
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = out[-1]
        if start <= last_end:  # touching counts as contiguous
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


def covered_seconds(intervals: list[Interval]) -> float:
    """Total seconds covered: the denominator a firing rate should use."""
    return float(sum((end - start).total_seconds() for start, end in intervals))


def coverage(window: Interval, ephys_chunks: list[Interval]) -> list[Interval]:
    """When the ephys rig was recording during a behavioural window.

    Becomes the row's ``time_support``. Using the nominal window instead puts
    every firing rate low by the coverage fraction, in an object that looks
    authoritative.
    """
    return merge(clip(ephys_chunks, window))


def coverage_by_unit(
    window: Interval,
    block_chunks: dict[Hashable, list[Interval]],
    block_units: dict[Hashable, set[int]],
) -> dict[int, list[Interval]]:
    """``coverage``, asked per unit.

    A unit was observed wherever a block that found it was recording. Spike rows
    cannot answer this: ``UnitMatching.Spikes`` writes nothing for a unit silent in
    a chunk, so a missing row means silent *or* never sorted there, and only the
    block roster tells them apart.
    """
    per_unit: dict[int, list[Interval]] = defaultdict(list)
    for block, units in block_units.items():
        covered = clip(block_chunks.get(block, []), window)
        for unit in units:
            per_unit[unit].extend(covered)
    return {unit: merged for unit, intervals in per_unit.items() if (merged := merge(intervals))}
