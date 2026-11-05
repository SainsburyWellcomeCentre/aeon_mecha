"""Unit tests for processed_ephys helpers that need no database."""

from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(hour, minute=0):
    """A datetime on the scenario's day, for readable fixtures."""
    return dt(2026, 5, 11, hour, minute)


class TestMostSpikesWins:
    """One rule settles a unit's block-scoped metadata on both axes.

    Across the sortings covering a chunk, and across the chunks covering a longer
    window, so the two readings cannot drift apart.
    """

    @staticmethod
    def _key(hour, group="shank0"):
        return (t(hour), t(hour + 1), "cfg", group, "ks4", 1)

    def test_sorting_with_more_spikes_wins(self):
        """Metadata follows the bulk of the data."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._key(7), self._key(8)
        assert _most_spikes_wins({early: 10, late: 900}) == late

    def test_ties_break_to_the_earliest_key(self):
        """An exact tie resolves by key order, not dict order — on both key kinds."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._key(7), self._key(8)
        assert _most_spikes_wins({late: 50, early: 50}) == early
        assert _most_spikes_wins({t(9): 5, t(8): 5}) == t(8)


class TestCoverageFollowsSortedChunks:
    """Coverage is the ephys chunks that were sorted, not the block someone typed."""

    def test_a_block_ending_mid_chunk_still_credits_the_whole_chunk(self):
        """The sorter reads each linked chunk file entire, so all of it was observed.

        Clipping to the block instead would shrink the denominator while leaving every
        spike from those chunks in the row, inflating each firing rate by the ratio
        below — and stranding the spikes outside the block, which pynapple then drops.
        """
        from aeon.dj_pipeline.utils import intervals

        hour = (t(8), t(9))
        sorted_chunks = [(t(8, 30), t(8, 40)), (t(8, 40), t(8, 50)), (t(8, 50), t(9))]
        block = (t(8, 35), t(8, 55))  # starts and ends mid-chunk, as a manual block may

        covered = intervals.coverage(hour, sorted_chunks)

        assert covered == [(t(8, 30), t(9))]  # touching chunks are one stretch
        assert intervals.covered_seconds(covered) == 30 * 60
        assert intervals.covered_seconds(intervals.clip(sorted_chunks, block)) == 20 * 60
