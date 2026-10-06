"""Unit tests for processed_ephys helpers that need no database."""

from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(hour, minute=0):
    """A datetime on the scenario's day, for readable fixtures."""
    return dt(2026, 5, 11, hour, minute)


class TestMostSpikesWins:
    """Which block's metadata wins when a unit spans two.

    ``unit_quality`` and ``qc_metrics`` are block-scoped, so a unit appearing in two
    blocks has two candidate values and the spec leaves the choice open.
    """

    def test_block_with_more_spikes_wins(self):
        """Test that metadata follows the bulk of the data."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins as owning_block

        assert owning_block({"A": 10, "B": 900}, {"A": t(7), "B": t(8)}) == "B"

    def test_ties_break_to_the_earliest_block(self):
        """Test that an exact tie is resolved deterministically, not by dict order."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins as owning_block

        assert owning_block({"B": 50, "A": 50}, {"A": t(7), "B": t(8)}) == "A"

    def test_a_single_block_needs_no_tie_break(self):
        """Test the ordinary case, which is every chunk that sits inside one block."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins as owning_block

        assert owning_block({"A": 3}, {"A": t(7)}) == "A"

    def test_ties_fall_through_to_the_key_itself(self):
        """Test that an exact tie on both count and tiebreak is still deterministic.

        Two blocks can share a start, so the tiebreak alone can tie. Without a
        final fall-through the winner is whichever the dict yields first.
        """
        from datetime import datetime

        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        same = datetime(2026, 5, 11, 8)
        counts = {("b", same): 5, ("a", same): 5}
        starts = {("b", same): same, ("a", same): same}

        assert _most_spikes_wins(counts, starts) == ("a", same)

    def test_tiebreak_defaults_to_the_key(self):
        """Test the no-tiebreak form fetch_span uses, where the key is the order."""
        from datetime import datetime

        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = datetime(2026, 5, 11, 8), datetime(2026, 5, 11, 9)
        assert _most_spikes_wins({late: 5, early: 5}) == early
        assert _most_spikes_wins({late: 9, early: 5}) == late


class TestBlockChunkClipping:
    """A block is only sorted within its own bounds, whatever chunks it links."""

    def test_chunks_are_clipped_to_the_block_window(self):
        """Test that a block credits only the part of a chunk it actually spans.

        EphysBlockInfo links the chunk containing each block bound in full, so a
        block covering 20 minutes of an hourly chunk would otherwise be credited
        with the whole hour and report every firing rate at a third of its value.
        """
        from aeon.dj_pipeline.utils import intervals

        block = (t(8, 39), t(8, 59))
        linked_whole_chunk = [(t(8), t(9))]

        clipped = intervals.clip(linked_whole_chunk, block)

        assert clipped == [block]
        assert intervals.covered_seconds(clipped) == 20 * 60
