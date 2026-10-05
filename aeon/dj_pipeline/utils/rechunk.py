"""Interval and roster arithmetic for re-chunking ephys data onto behavioural hours.

Boundary bugs live here, so this is plain functions over ``(start, end)`` datetime
pairs — no DataJoint, no pynapple, no I/O — and it is tested on its own.

Every window is half-open, ``[start, end)``.
"""

from collections import defaultdict
from collections.abc import Hashable
from datetime import datetime
from typing import TypeVar

Interval = tuple[datetime, datetime]
Block = TypeVar("Block", bound=Hashable)
"""Whatever identifies a block to the caller — this module never looks inside it."""


def clip(intervals: list[Interval], window: Interval) -> list[Interval]:
    """Trim ``intervals`` to what falls inside ``window``.

    An interval that only touches the window at a single instant has zero width
    and is dropped, because half-open means touching is not overlapping.
    """
    lo, hi = window
    out = []
    for start, end in intervals:
        s, e = max(start, lo), min(end, hi)
        if s < e:  # half-open: zero width means no overlap
            out.append((s, e))
    return out


def merge(intervals: list[Interval]) -> list[Interval]:
    """Join intervals that overlap or touch, and keep the gaps that remain.

    Two ephys chunks that meet exactly become one interval. A real break between
    them stays a break, which is what lets a chunk report a gapped ``time_support``
    instead of pretending it was covered throughout.
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
    """How many seconds ``intervals`` cover in total.

    Named for the ``covered_seconds`` metadata column it feeds: the denominator a
    firing rate should use, rather than the length of the chunk.
    """
    return float(sum((end - start).total_seconds() for start, end in intervals))


def coverage(window: Interval, ephys_chunks: list[Interval]) -> list[Interval]:
    """When the ephys rig was actually recording during a behavioural window.

    This becomes the row's ``time_support``. Use the nominal window instead and
    every firing rate in the chunk comes out low by the coverage fraction, in an
    object that looks authoritative.
    """
    return merge(clip(ephys_chunks, window))


def coverage_by_unit(
    window: Interval,
    block_chunks: dict[Hashable, list[Interval]],
    block_units: dict[Hashable, set[int]],
) -> dict[int, list[Interval]]:
    """The same question as ``coverage``, asked separately for each unit.

    A unit was observed wherever a block that found it was recording. So a unit
    that only one of two covering blocks found gets a shorter window than the chunk
    — the cross-block case. Get it wrong and a real neuron reports half its rate.

    Spike rows cannot answer this. ``UnitMatching.Spikes`` writes nothing for a unit
    that stayed silent in a chunk, so a missing row means silent *or* never sorted
    there, and only the block roster tells them apart.
    """
    per_unit: dict[int, list[Interval]] = defaultdict(list)
    for block, units in block_units.items():
        covered = clip(block_chunks.get(block, []), window)
        for unit in units:
            per_unit[unit].extend(covered)
    return {unit: merged for unit, intervals in per_unit.items() if (merged := merge(intervals))}


def owning_block(spike_counts_by_block: dict[Block, int], block_starts: dict[Block, datetime]) -> Block:
    """Decide which block speaks for a unit that appears in several.

    Each block has its own opinion about a unit's quality and electrode, and the
    row can only carry one. The block holding most of the unit's spikes wins; an
    exact tie goes to the earliest, so the answer never depends on dict ordering.
    """
    return max(
        spike_counts_by_block,
        key=lambda block: (spike_counts_by_block[block], -block_starts[block].timestamp()),
    )
