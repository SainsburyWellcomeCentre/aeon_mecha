"""Unit tests for the pure unit-matching helpers extracted in PR #604."""

from datetime import UTC, datetime

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestResolveMatchingParams:
    def test_merges_over_defaults(self):
        from aeon.dj_pipeline.spike_sorting import _resolve_matching_params

        defaults = {"delta_time": 0.4, "match_score": 0.5, "min_score": 0.1, "exclude_noise": True}
        assert _resolve_matching_params({}) == defaults
        assert _resolve_matching_params({"delta_time": 1.0}) == {**defaults, "delta_time": 1.0}

    def test_unknown_key_raises(self):
        """A misspelled key must raise, not silently leave the default in effect."""
        from aeon.dj_pipeline.spike_sorting import _resolve_matching_params

        with pytest.raises(ValueError, match="match_scroe"):
            _resolve_matching_params({"match_scroe": 0.9})


class TestRestrictToOverlap:
    def test_restricts_inclusively_and_rebases(self):
        """Spikes outside [start, end] drop; the rest are rebased to the window start."""
        from aeon.dj_pipeline.spike_sorting import _restrict_to_overlap

        # 5.0 and 15.0 are the bounds themselves - both ends are inclusive.
        times = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        np.testing.assert_allclose(_restrict_to_overlap(times, 5.0, 15.0), [0.0, 5.0, 10.0])
        assert len(_restrict_to_overlap(np.array([]), 0.0, 10.0)) == 0


class TestCompareSpikeTrainsInOverlap:
    """Bounds are naive datetimes; spike trains are epoch seconds."""

    B1 = (datetime(2026, 5, 11, 7, 0, 0), datetime(2026, 5, 11, 8, 0, 0))
    B2 = (datetime(2026, 5, 11, 7, 30, 0), datetime(2026, 5, 11, 8, 30, 0))
    DISJOINT = (datetime(2026, 5, 11, 9, 0, 0), datetime(2026, 5, 11, 10, 0, 0))

    @staticmethod
    def _train(bounds, offset_s, n=500, step=0.01):
        """Build n spikes at `step` intervals, starting `offset_s` after the window start."""
        start = bounds[0].replace(tzinfo=UTC).timestamp()
        return start + offset_s + np.arange(n) * step

    def test_returns_none_when_blocks_do_not_overlap(self):
        from aeon.dj_pipeline.spike_sorting import _compare_spike_trains_in_overlap

        result = _compare_spike_trains_in_overlap(
            {1: self._train(self.DISJOINT, 10)},
            {1: self._train(self.B1, 10)},
            self.DISJOINT,
            self.B1,
        )
        assert result is None

    def test_returns_none_when_one_side_has_no_spikes_in_overlap(self):
        from aeon.dj_pipeline.spike_sorting import _compare_spike_trains_in_overlap

        # Must lie inside the overlap, else both sides are empty and the helper returns
        # None via the "neither side" path instead.
        result = _compare_spike_trains_in_overlap(
            {1: self._train(self.B2, 60)},  # inside the overlap
            {1: np.array([])},  # the empty side
            self.B1,
            self.B2,
        )
        assert result is None

    def test_identical_trains_match(self):
        from aeon.dj_pipeline.spike_sorting import _compare_spike_trains_in_overlap

        # Spikes inside the 07:30-08:00 overlap of B1 and B2.
        train = self._train(self.B2, 60)
        comparison = _compare_spike_trains_in_overlap({7: train}, {3: train}, self.B1, self.B2)
        assert comparison is not None
        # agreement_scores: rows = previous units, cols = current units
        assert comparison.agreement_scores.loc[3, 7] == pytest.approx(1.0)

    def test_disjoint_trains_do_not_match(self):
        from aeon.dj_pipeline.spike_sorting import _compare_spike_trains_in_overlap

        this_train = self._train(self.B2, 60, step=0.01)
        prev_train = self._train(self.B2, 60.005, step=0.01)  # 5 ms offset, beyond delta_time
        comparison = _compare_spike_trains_in_overlap(
            {7: this_train}, {3: prev_train}, self.B1, self.B2
        )
        assert comparison is not None
        # 5 ms apart at delta_time=0.4 ms: no coincidences. `< 0.5` would accept 0.49.
        assert comparison.agreement_scores.loc[3, 7] == pytest.approx(0.0, abs=0.01)
