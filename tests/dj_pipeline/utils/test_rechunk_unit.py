"""Unit tests for rechunk.py — pure interval arithmetic, no database."""

from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(h, m=0, s=0):
    """A datetime on the golden day, for readable fixtures."""
    return dt(2026, 5, 11, h, m, s)


class TestIntervalArithmetic:
    """Clipping and merging, where the boundary rules live."""

    def test_clip_is_half_open_at_both_ends(self):
        """Test that an interval touching the window edge contributes nothing.

        The window is [start, end). An ephys chunk ending exactly at chunk_start
        belongs to the previous behavioural chunk, and one starting exactly at
        chunk_end belongs to the next. Getting this wrong double-counts spikes at
        24 boundaries a day.
        """
        from aeon.dj_pipeline.utils.rechunk import clip

        window = (t(8), t(9))
        assert clip([(t(7), t(8))], window) == []  # ends at the boundary
        assert clip([(t(9), t(10))], window) == []  # starts at the boundary
        assert clip([(t(7), t(8, 30))], window) == [(t(8), t(8, 30))]
        assert clip([(t(8, 30), t(10))], window) == [(t(8, 30), t(9))]
        assert clip([(t(7), t(10))], window) == [(t(8), t(9))]

    def test_merge_joins_touching_and_overlapping_but_keeps_gaps(self):
        """Test that a real gap survives merging — it becomes a second interval."""
        from aeon.dj_pipeline.utils.rechunk import merge

        assert merge([(t(8), t(8, 30)), (t(8, 30), t(9))]) == [(t(8), t(9))]
        assert merge([(t(8), t(8, 40)), (t(8, 20), t(9))]) == [(t(8), t(9))]
        assert merge([(t(8), t(8, 20)), (t(8, 40), t(9))]) == [
            (t(8), t(8, 20)),
            (t(8, 40), t(9)),
        ]
        assert merge([]) == []

    def test_total_seconds_sums_a_gapped_coverage(self):
        """Test the denominator every firing rate divides by."""
        from aeon.dj_pipeline.utils.rechunk import total_seconds

        assert total_seconds([(t(8), t(8, 20)), (t(8, 40), t(9))]) == 2400.0
        assert total_seconds([]) == 0.0


class TestCoverage:
    """Chunk-level and per-unit coverage, including the cross-block case."""

    def test_chunk_coverage_reflects_a_gap_in_ephys(self):
        """Test that a gap between ephys chunks survives into the coverage."""
        from aeon.dj_pipeline.utils.rechunk import chunk_coverage

        # rig recorded 08:00-08:20 and 08:40-09:00, off in between
        coverage = chunk_coverage((t(8), t(9)), [(t(7, 50), t(8, 20)), (t(8, 40), t(9, 10))])
        assert coverage == [(t(8), t(8, 20)), (t(8, 40), t(9))]

    def test_unit_coverage_differs_across_a_block_boundary(self):
        """Test the case the whole design turns on.

        Block A covers the first half of the hour and found unit 7. Block B covers
        the second half and found units 7 and 99. Unit 99 was never sorted over the
        first half — it was not silent there, it was not looked for. Its denominator
        must be half the hour, not the whole hour.
        """
        from aeon.dj_pipeline.utils.rechunk import unit_coverage

        coverage = unit_coverage(
            (t(8), t(9)),
            {"A": [(t(8), t(8, 30))], "B": [(t(8, 30), t(9))]},
            {"A": {7}, "B": {7, 99}},
        )

        assert coverage[7] == [(t(8), t(9))]  # both blocks, merged
        assert coverage[99] == [(t(8, 30), t(9))]  # second half only

    def test_unit_coverage_is_empty_for_a_unit_no_block_found(self):
        """Test that a unit absent from every covering block gets no coverage."""
        from aeon.dj_pipeline.utils.rechunk import unit_coverage

        assert 99 not in unit_coverage((t(8), t(9)), {"A": [(t(8), t(9))]}, {"A": {7}})


class TestOwningBlock:
    """Which block's metadata wins when a unit spans two.

    ``unit_quality`` and ``qc_metrics`` are block-scoped, so a unit appearing in two
    blocks has two candidate values and the spec leaves the choice open.
    """

    def test_block_with_more_spikes_wins(self):
        """Test that metadata follows the bulk of the data."""
        from aeon.dj_pipeline.utils.rechunk import owning_block

        assert owning_block({"A": 10, "B": 900}, {"A": t(7), "B": t(8)}) == "B"

    def test_ties_break_to_the_earliest_block(self):
        """Test that an exact tie is resolved deterministically, not by dict order."""
        from aeon.dj_pipeline.utils.rechunk import owning_block

        assert owning_block({"B": 50, "A": 50}, {"A": t(7), "B": t(8)}) == "A"

    def test_a_single_block_needs_no_tie_break(self):
        """Test the ordinary case, which is every chunk that sits inside one block."""
        from aeon.dj_pipeline.utils.rechunk import owning_block

        assert owning_block({"A": 3}, {"A": t(7)}) == "A"
