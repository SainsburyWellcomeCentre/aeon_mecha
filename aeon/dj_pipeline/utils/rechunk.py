"""Interval and roster arithmetic for re-chunking ephys data to the behavioural grain.

Pure functions over ``(start, end)`` datetime pairs: no DataJoint, no pynapple, no
I/O. The re-chunking is where the boundary bugs live, so it is kept separable and
tested on its own.

All windows are half-open, ``[start, end)``.
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

    This is what a row's ``time_support`` is built from. Setting it to the nominal
    window instead would understate every firing rate by the coverage fraction.
    """
    return merge(clip(ephys_chunks, window))


def unit_coverage(
    window: Interval,
    block_chunks: dict[Hashable, list[Interval]],
    block_units: dict[Hashable, set[int]],
) -> dict[int, list[Interval]]:
    """Per-unit coverage: where each unit was actually sorted, within ``window``.

    A unit is covered over the chunks of every block that found it. Units found by
    only some of the blocks covering a behavioural chunk therefore get a shorter
    denominator than the chunk itself — that is the cross-block case, and getting it
    wrong reports a real neuron as firing at half its rate.

    Not derivable from spike rows: ``UnitMatching.Spikes`` writes no row for a unit
    that was silent in a chunk, so an absent row means *either* silent *or* never
    sorted there.
    """
    per_unit: dict[int, list[Interval]] = defaultdict(list)
    for block, units in block_units.items():
        covered = clip(block_chunks.get(block, []), window)
        for unit in units:
            per_unit[unit].extend(covered)
    return {unit: merged for unit, intervals in per_unit.items() if (merged := merge(intervals))}


def owning_block(spike_counts_by_block: dict[Block, int], block_starts: dict[Block, datetime]) -> Block:
    """Pick whose per-unit metadata wins when a unit spans several blocks.

    Most spikes wins; an exact tie goes to the earliest block, so the answer never
    depends on dict ordering.
    """
    return max(
        spike_counts_by_block,
        key=lambda block: (spike_counts_by_block[block], -block_starts[block].timestamp()),
    )
