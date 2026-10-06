"""Unit tests for processed_ephys helpers that need no database."""

from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(hour, minute=0):
    """A datetime on the scenario's day, for readable fixtures."""
    return dt(2026, 5, 11, hour, minute)


class TestOwningBlock:
    """Which block's metadata wins when a unit spans two.

    ``unit_quality`` and ``qc_metrics`` are block-scoped, so a unit appearing in two
    blocks has two candidate values and the spec leaves the choice open.
    """

    def test_block_with_more_spikes_wins(self):
        """Test that metadata follows the bulk of the data."""
        from aeon.dj_pipeline.processed_ephys import _owning_block as owning_block

        assert owning_block({"A": 10, "B": 900}, {"A": t(7), "B": t(8)}) == "B"

    def test_ties_break_to_the_earliest_block(self):
        """Test that an exact tie is resolved deterministically, not by dict order."""
        from aeon.dj_pipeline.processed_ephys import _owning_block as owning_block

        assert owning_block({"B": 50, "A": 50}, {"A": t(7), "B": t(8)}) == "A"

    def test_a_single_block_needs_no_tie_break(self):
        """Test the ordinary case, which is every chunk that sits inside one block."""
        from aeon.dj_pipeline.processed_ephys import _owning_block as owning_block

        assert owning_block({"A": 3}, {"A": t(7)}) == "A"
