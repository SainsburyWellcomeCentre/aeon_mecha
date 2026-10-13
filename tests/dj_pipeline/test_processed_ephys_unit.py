"""Unit tests for processed_ephys helpers that need no database."""

import warnings
from datetime import datetime as dt

import pytest

pytestmark = pytest.mark.unit


def t(hour, minute=0):
    """A datetime on the scenario's day, for readable fixtures."""
    return dt(2026, 5, 11, hour, minute)


class TestMostSpikesWins:
    """Which source's metadata wins when a unit appears in more than one.

    ``unit_quality`` and ``qc_metrics`` are block-scoped, so a unit appearing in two
    sortings has two candidate values. One rule answers that on both axes - across
    the sortings covering a chunk, and across the chunks covering a span - so the
    two cannot drift apart.
    """

    @staticmethod
    def _ident(hour, group="shank0"):
        """A sorting's identity: block bounds, config, group, sorting and matching sets."""
        return (t(hour), t(hour + 1), "cfg", group, "ks4", 1)

    def test_sorting_with_more_spikes_wins(self):
        """Test that metadata follows the bulk of the data."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._ident(7), self._ident(8)
        assert _most_spikes_wins({early: 10, late: 900}) == late

    def test_ties_break_to_the_earliest_block(self):
        """Test that an exact tie is resolved deterministically, not by dict order."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = self._ident(7), self._ident(8)
        assert _most_spikes_wins({late: 50, early: 50}) == early

    def test_a_single_sorting_needs_no_tie_break(self):
        """Test the ordinary case, which is every chunk that sits inside one block."""
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        only = self._ident(7)
        assert _most_spikes_wins({only: 3}) == only

    def test_two_blocks_sharing_a_start_fall_through_to_the_rest_of_the_identity(self):
        """Test that a tie on count and start is still decided, not left to dict order.

        Two sortings can share a block start - the same block under two electrode
        groups - so ordering by start alone can tie. The rest of the identity
        carries on from there.
        """
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        a, b = self._ident(8, "shank1"), self._ident(8, "shank2")
        assert _most_spikes_wins({b: 5, a: 5}) == a

    def test_the_same_rule_orders_chunks_in_a_span(self):
        """Test the chunk-axis call in fetch_span, where the key is a chunk start.

        Both kinds of key lead with a time, so one rule orders both without being
        told which part to look at.
        """
        from aeon.dj_pipeline.processed_ephys import _most_spikes_wins

        early, late = t(8), t(9)
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


class TestSortingGuards:
    """Two sortings can cover one chunk legitimately — or not at all."""

    @staticmethod
    def _s(ident, electrodes=frozenset(), units=frozenset()):
        return {"electrodes": frozenset(electrodes), "units": set(units)}

    @staticmethod
    def _ident(hour, group="shank0", paramset="ks4", matching=1):
        return (t(hour), t(hour + 1), "cfg", group, paramset, matching)

    def test_one_group_matched_twice_is_refused_from_the_identity_alone(self):
        """Test the case provable without any electrode data.

        Global unit ids are handed out per insertion across every matching
        parameter set, so the same electrodes matched under two of them give each
        neuron an id in each family. The chunk then holds both.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8, matching=1), self._ident(8, matching=2)
        with pytest.raises(ValueError, match="matching parameter sets"):
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_the_second_matching_paramset_is_caught_across_blocks_too(self):
        """Test that it is the matching set that matters, not where the blocks sit."""
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8, matching=1), self._ident(9, matching=2)
        with pytest.raises(ValueError, match="matching parameter sets"):
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_sequential_blocks_on_one_matching_paramset_are_the_ordinary_case(self):
        """Test that the chunk this table exists to serve passes without a murmur.

        UnitMatching refuses a block that overlaps nothing already matched, so every
        block under one matching parameter set is linked into a chain and global unit
        ids propagate along it. Spikes is unique on (global_unit, chunk_start), so
        these cannot double-count. Refusing, or even warning, would fire on almost
        every real chunk.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8), self._ident(9)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_two_sorting_paramsets_under_one_matching_set_are_allowed(self):
        """Test that re-sorting a block is not the hazard it looks like.

        UnitMatching picks partners by insertion and matching parameter set alone,
        filtered to overlapping blocks - and a block overlaps itself. So the second
        sorting is compared against the first and lands on the same global units.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a, b = self._ident(8, paramset="ks4"), self._ident(8, paramset="ks4_alt")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_disjoint_electrode_groups_are_allowed(self):
        """Test that shank1 and shank2 coexist: different electrodes, different neurons.

        Separate matching parameter sets per shank is the documented pattern.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="shank1", matching=1)
        b = self._ident(8, group="shank2", matching=2)
        _assert_no_double_counting({a: self._s(a, {0, 1}), b: self._s(b, {2, 3})})

    def test_overlapping_electrode_groups_are_refused(self):
        """Test that 'all' alongside 'shank1' is caught when the sets are known."""
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="all", matching=1)
        b = self._ident(8, group="shank1", matching=2)
        with pytest.raises(ValueError, match="same 2 electrodes"):
            _assert_no_double_counting({a: self._s(a, {0, 1, 2, 3}), b: self._s(b, {0, 1})})

    def test_unknown_electrode_sets_warn_rather_than_pass_silently(self):
        """Test that an unpopulated ElectrodeGroup.Electrode is reported, not ignored.

        Nothing in the pipeline writes that part table today, so the overlap check
        has no data for the one pair that needs it. Passing silently would look
        like a verified result.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8, group="shank1", matching=1)
        b = self._ident(8, group="shank2", matching=2)
        with pytest.warns(UserWarning, match="could not be checked"):
            _assert_no_double_counting({a: self._s(a), b: self._s(b)})

    def test_a_single_sorting_needs_no_checks(self):
        """Test the ordinary case stays quiet."""
        from aeon.dj_pipeline.processed_ephys import _assert_no_double_counting

        a = self._ident(8)
        _assert_no_double_counting({a: self._s(a)})

    def test_two_configs_without_a_shared_unit_are_allowed(self):
        """Test that a config change alone does not refuse the chunk.

        Units are per-config, coverage is already per-unit, and n_partial_units
        flags it. Refusing would discard data that is handled correctly.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_one_config_per_unit

        _assert_one_config_per_unit({"a": "cfgX", "b": "cfgY"}, {"a": {1, 2}, "b": {3, 4}})

    def test_a_unit_spanning_two_configs_is_refused(self):
        """Test the unsafe case: one global unit under two electrode configs.

        GlobalUnit carries one physical peak electrode, and the other config may
        never have recorded it.
        """
        from aeon.dj_pipeline.processed_ephys import _assert_one_config_per_unit

        with pytest.raises(ValueError, match="more than one electrode config"):
            _assert_one_config_per_unit({"a": "cfgX", "b": "cfgY"}, {"a": {1, 2}, "b": {2, 3}})
