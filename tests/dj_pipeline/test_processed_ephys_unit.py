"""Unit tests for processed_ephys helpers that need no database."""

import warnings
from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(hour, minute=0):
    """A datetime on the scenario's day, for readable fixtures."""
    return dt(2026, 5, 11, hour, minute)


class TestMostSpikesWins:
    """One rule settles a unit's block-scoped metadata on both axes.

    Across the sortings covering a chunk, and across the chunks covering a span,
    so ``make`` and ``fetch_span`` cannot drift apart.
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


class TestSortingGuards:
    """Two sortings can cover one chunk legitimately — or not at all."""

    @staticmethod
    def _s(ident, electrodes=frozenset(), units=frozenset()):
        return {"electrodes": frozenset(electrodes), "units": set(units)}

    @staticmethod
    def _ident(hour, group="shank0", paramset="ks4", matching=1):
        return (t(hour), t(hour + 1), "cfg", group, paramset, matching)

    @pytest.mark.parametrize("second_hour", [8, 9], ids=["same block", "later block"])
    def test_one_group_matched_twice_is_refused_from_the_identity_alone(self, second_hour):
        """Unsafe without needing any electrode data.

        Global unit ids are handed out per insertion across every matching paramset,
        so the same electrodes under two of them give each neuron an id in each.
        The matching set is what makes this unsafe, so block position is irrelevant.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8, matching=1), self._ident(second_hour, matching=2)
        with pytest.raises(ValueError, match="matching parameter sets"):
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_sequential_blocks_on_one_matching_paramset_are_the_ordinary_case(self):
        """The chunk this table exists to serve passes without a murmur.

        One matching paramset links every block into a chain and ``Spikes`` is
        unique on ``(global_unit, chunk_start)``. Refusing, or even warning, would
        fire on almost every real chunk.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8), self._ident(9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_disjoint_electrode_groups_are_allowed(self):
        """shank1 and shank2 coexist: different electrodes, different neurons.

        A matching paramset per shank is the documented pattern, so over-refusing
        here would block populate on a supported configuration.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="shank1", matching=1)
        b = self._ident(8, group="shank2", matching=2)
        _assert_no_double_counting({a: self._s(a, {0, 1}), b: self._s(b, {2, 3})})

    def test_overlapping_electrode_groups_are_refused(self):
        """'all' alongside 'shank1', caught when the electrode sets are known."""
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="all", matching=1)
        b = self._ident(8, group="shank1", matching=2)
        with pytest.raises(ValueError, match="same 2 electrodes"):
            _assert_no_double_counting({a: self._s(a, {0, 1, 2, 3}), b: self._s(b, {0, 1})})

    def test_unknown_electrode_sets_warn_rather_than_pass_silently(self):
        """Nothing populates ElectrodeGroup.Electrode today.

        So the one pair that needs the overlap check has no data for it, and
        passing silently would look like a verified result.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="shank1", matching=1)
        b = self._ident(8, group="shank2", matching=2)
        with pytest.warns(UserWarning, match="could not be checked"):
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_two_configs_without_a_shared_unit_are_allowed(self):
        """A config change alone does not refuse the chunk.

        Units are per-config, coverage is per-unit, n_partial_units flags it.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_one_config_per_unit

        _assert_one_config_per_unit({"a": "cfgX", "b": "cfgY"}, {"a": {1, 2}, "b": {3, 4}})

    def test_a_unit_spanning_two_configs_is_refused(self):
        """GlobalUnit carries one peak electrode; the other config may not have it."""
        from aeon.dj_pipeline.processed_ephys import _assert_one_config_per_unit

        with pytest.raises(ValueError, match="more than one electrode config"):
            _assert_one_config_per_unit({"a": "cfgX", "b": "cfgY"}, {"a": {1, 2}, "b": {2, 3}})
