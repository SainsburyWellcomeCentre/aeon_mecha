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
    def _ident(hour, group="shank0"):
        return (t(hour), t(hour + 1), "cfg", group, "ks4", 1)

    def test_sorting_with_more_spikes_wins(self):
        """Metadata follows the bulk of the data."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._ident(7), self._ident(8)
        assert _most_spikes_wins({early: 10, late: 900}) == late

    def test_ties_break_to_the_earliest_key(self):
        """An exact tie resolves by key order, not dict order — on both key kinds."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._ident(7), self._ident(8)
        assert _most_spikes_wins({late: 50, early: 50}) == early
        assert _most_spikes_wins({t(9): 5, t(8): 5}) == t(8)


class TestBlockChunkClipping:
    """A block is only sorted within its own bounds, whatever chunks it links."""

    def test_chunks_are_clipped_to_the_block_window(self):
        """EphysBlockInfo links the chunk holding each bound whole.

        A block covering 20 minutes of an hourly chunk would otherwise be credited
        the whole hour and report every firing rate at a third of its value.
        """
        from aeon.dj_pipeline.utils import intervals

        block = (t(8, 39), t(8, 59))
        clipped = intervals.clip([(t(8), t(9))], block)

        assert clipped == [block]
        assert intervals.covered_seconds(clipped) == 20 * 60
