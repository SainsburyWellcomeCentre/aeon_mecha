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
    """Clip ``intervals`` to ``window``, dropping anything that lands empty."""
    lo, hi = window
    out = []
    for start, end in intervals:
        s, e = max(start, lo), min(end, hi)
        if s < e:  # half-open: zero width means no overlap
            out.append((s, e))
    return out


def merge(intervals: list[Interval]) -> list[Interval]:
    """Merge overlapping and touching intervals; real gaps survive as separate entries."""
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


def total_seconds(intervals: list[Interval]) -> float:
    """Total duration covered by ``intervals``, in seconds."""
    return float(sum((end - start).total_seconds() for start, end in intervals))


def chunk_coverage(window: Interval, ephys_chunks: list[Interval]) -> list[Interval]:
    """Ephys coverage of a behavioural window: the chunks, clipped and merged.

    A row's ``time_support`` comes from this. Use the nominal window instead and
    every firing rate in the chunk drops by the coverage fraction.
    """
    return merge(clip(ephys_chunks, window))


def unit_coverage(
    window: Interval,
    block_chunks: dict[Hashable, list[Interval]],
    block_units: dict[Hashable, set[int]],
) -> dict[int, list[Interval]]:
    """Per-unit coverage: where each unit was actually sorted, within ``window``.

    A unit covers the chunks of every block that found it. So a unit only some of
    the covering blocks found gets a shorter denominator than the chunk — the
    cross-block case. Get it wrong and a real neuron reports half its firing rate.

    Spike rows cannot answer this. ``UnitMatching.Spikes`` writes nothing for a unit
    that stayed silent in a chunk, so a missing row means silent *or* never sorted.
    """
    per_unit: dict[int, list[Interval]] = defaultdict(list)
    for block, units in block_units.items():
        covered = clip(block_chunks.get(block, []), window)
        for unit in units:
            per_unit[unit].extend(covered)
    return {unit: merged for unit, intervals in per_unit.items() if (merged := merge(intervals))}


def owning_block(spike_counts_by_block: dict[Block, int], block_starts: dict[Block, datetime]) -> Block:
    """Pick whose metadata wins when a unit spans several blocks.

    Most spikes wins. An exact tie goes to the earliest block, so the answer never
    depends on dict ordering.
    """
    return max(
        spike_counts_by_block,
        key=lambda block: (spike_counts_by_block[block], -block_starts[block].timestamp()),
    )
