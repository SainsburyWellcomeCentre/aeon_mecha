"""Unit tests for intervals.py — pure interval arithmetic, no database."""

from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(h, m=0, s=0):
    """A datetime on the golden day, for readable fixtures."""
    return dt(2026, 5, 11, h, m, s)


class TestIntervalArithmetic:
    """Clipping and merging, where the boundary rules live."""

    def test_clip_is_half_open_at_both_ends(self):
        """An ephys chunk touching the window edge belongs to the neighbour.

        Getting this wrong double-counts spikes at 24 boundaries a day.
        """
        from aeon.dj_pipeline.utils.intervals import clip

        window = (t(8), t(9))
        assert clip([(t(7), t(8))], window) == []  # ends at the boundary
        assert clip([(t(9), t(10))], window) == []  # starts at the boundary
        assert clip([(t(7), t(8, 30))], window) == [(t(8), t(8, 30))]
        assert clip([(t(8, 30), t(10))], window) == [(t(8, 30), t(9))]
        assert clip([(t(7), t(10))], window) == [(t(8), t(9))]

    def test_merge_joins_touching_and_overlapping_but_keeps_gaps(self):
        """A real gap survives merging; a nested interval never shortens its host."""
        from aeon.dj_pipeline.utils.intervals import merge

        assert merge([(t(8), t(8, 30)), (t(8, 30), t(9))]) == [(t(8), t(9))]
        assert merge([(t(8), t(8, 40)), (t(8, 20), t(9))]) == [(t(8), t(9))]
        # Nested: one block covers the hour, another covers its middle. Taking the
        # later end rather than the greater one silently truncates the host.
        assert merge([(t(8), t(9)), (t(8, 20), t(8, 40))]) == [(t(8), t(9))]
        assert merge([(t(8), t(8, 20)), (t(8, 40), t(9))]) == [
            (t(8), t(8, 20)),
            (t(8, 40), t(9)),
        ]
        assert merge([]) == []

    def test_covered_seconds_sums_a_gapped_coverage(self):
        """The denominator every firing rate divides by: a sum, not a span."""
        from aeon.dj_pipeline.utils.intervals import covered_seconds

        assert covered_seconds([(t(8), t(8, 20)), (t(8, 40), t(9))]) == 40 * 60
        assert covered_seconds([]) == 0.0


class TestCoverage:
    """When the rig was recording, for the chunk and for each unit."""

    def test_coverage_reflects_a_gap_in_ephys(self):
        """Coverage clips to the window and keeps the gap inside it."""
        from aeon.dj_pipeline.utils.intervals import coverage

        # rig recorded 07:50-08:20 and 08:40-09:10, off in between
        covered = coverage((t(8), t(9)), [(t(7, 50), t(8, 20)), (t(8, 40), t(9, 10))])
        assert covered == [(t(8), t(8, 20)), (t(8, 40), t(9))]

    def test_coverage_by_unit_differs_across_a_block_boundary(self):
        """The case the whole design turns on.

        Block A covers the first half of the hour and found unit 7; block B covers
        the second half and found 7 and 99. Unit 99 was not silent over the first
        half, it was never looked for, so it was observed for half the hour.
        """
        from aeon.dj_pipeline.utils.intervals import coverage_by_unit

        covered = coverage_by_unit(
            (t(8), t(9)),
            {"A": [(t(7, 30), t(8, 30))], "B": [(t(8, 30), t(9, 30))]},
            {"A": {7}, "B": {7, 99}},
        )

        assert covered[7] == [(t(8), t(9))]  # both blocks, merged, clipped
        assert covered[99] == [(t(8, 30), t(9))]  # second half only

    def test_coverage_by_unit_drops_a_unit_whose_block_misses_the_window(self):
        """A block outside the window clips to nothing, so its units get no row.

        An empty list would read as "covered for zero seconds" and divide by zero
        downstream; absence is the honest answer.
        """
        from aeon.dj_pipeline.utils.intervals import coverage_by_unit

        covered = coverage_by_unit((t(8), t(9)), {"A": [(t(5), t(6))]}, {"A": {7}})
        assert covered == {}
