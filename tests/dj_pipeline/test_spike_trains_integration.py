"""Integration tests for processed_ephys.SpikeTrains on testcontainers MySQL.

Synthetic on purpose. The scenario in ``tests/fixtures/ephys/spike_train_factories.py``
puts a two-block chunk, an ephys gap and a boundary spike exactly where an assertion
can see them. No fixed real recording obliges.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def spike_trains_scenario(ephys_full_pipeline, tmp_path_factory):
    """Build the behaviour + ephys scenario and return what the tests assert against.

    Takes ``dj_store`` from ``ephys_full_pipeline`` rather than configuring its own.
    ``dj.config.stores`` is global and assigning it replaces the whole mapping, so two
    fixtures each pointing ``dj_store`` at their own tmpdir leaves rows written by the
    first unreadable once the second runs.
    """
    from spike_train_factories import build_scenario

    tmp_path = tmp_path_factory.mktemp("repo")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    from ephys_factories import register_synthetic_experiment

    experiment_name = "synthetic-spike-trains"
    register_synthetic_experiment(tmp_path, raw_dir, experiment_name, "2026-05-11T08-00-00")
    return build_scenario(experiment_name)


def _mine(scenario):
    """SpikeTrains restricted to the synthetic experiment.

    The golden suite populates this same table, and pytest runs it first, so an
    unrestricted query here picks up its rows.
    """
    from aeon.dj_pipeline import processed_ephys

    return processed_ephys.SpikeTrains & {"experiment_name": scenario["experiment_name"]}


class TestKeySource:
    """What is computable, and what is deliberately not."""

    def test_only_chunks_overlapping_matched_ephys_are_computable(self, spike_trains_scenario):
        """Test that the key_source is the overlap, not the cross product.

        Three behavioural chunks are registered but ephys covers only two. The third
        must not appear — there is nothing to compute it from. An empty key_source is
        not an error in DataJoint, so this asserts a positive count too.
        """
        from aeon.dj_pipeline import processed_ephys

        scoped = {"experiment_name": spike_trains_scenario["experiment_name"]}
        keys = (processed_ephys.SpikeTrains().key_source & scoped).to_dicts()
        starts = sorted({k["chunk_start"] for k in keys})

        assert starts == spike_trains_scenario["covered_chunk_starts"]
        assert spike_trains_scenario["uncovered_chunk_start"] not in starts

    def test_matching_is_per_chunk_not_per_insertion(self, spike_trains_scenario):
        """Test that one matched block does not make every chunk computable.

        Renaming chunk_start before the semijoin drops it from the heading, so the
        restriction degenerates to "this insertion has some match somewhere" and
        key_source yields chunks whose own ephys was never matched.
        """
        from aeon.dj_pipeline import acquisition, ephys, processed_ephys, spike_sorting

        scoped = {"experiment_name": spike_trains_scenario["experiment_name"]}
        for key in (processed_ephys.SpikeTrains().key_source & scoped).to_dicts():
            window = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
            # key_source guarantees a matched ephys *chunk* overlaps, not a matched
            # block: EphysBlockInfo links the chunk containing each bound whole, so
            # a chunk can extend past its block and still be legitimate here.
            matched = (
                (ephys.EphysChunk & (spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk).proj())
                & {k: key[k] for k in ("experiment_name", "subject", "insertion_number")}
                & (f'chunk_start < "{window[1]}" AND chunk_end > "{window[0]}"')
            )
            assert matched, (
                f"key_source yielded {key['chunk_start']}, which no matched block covers; "
                "the match is being applied per insertion rather than per chunk"
            )

    def test_a_block_matched_under_two_paramsets_is_refused(self, spike_trains_scenario):
        """Test that a second matching run stops the chunk rather than doubling it.

        Two matching parameter sets never compare against each other, so the same
        neurons get an id in each family. The duplicate is removed again in the same
        test: every later test here assumes one matching set per block.
        """
        from spike_train_factories import ALT_MATCHING_PARAMSET, add_duplicate_sorting

        from aeon.dj_pipeline import processed_ephys, spike_sorting

        add_duplicate_sorting(spike_trains_scenario["experiment_name"])
        key = {
            "experiment_name": spike_trains_scenario["experiment_name"],
            "chunk_start": spike_trains_scenario["covered_chunk_starts"][0],
            **{k: spike_trains_scenario[k] for k in ("subject", "insertion_number")},
        }
        try:
            with pytest.raises(ValueError, match="matching parameter sets"):
                processed_ephys.SpikeTrains().make(key)
        finally:
            alt = {"matching_paramset_id": ALT_MATCHING_PARAMSET}
            (spike_sorting.UnitMatching & alt).delete(prompt=False)
            (spike_sorting.UnitMatchingParamSet & alt).delete(prompt=False)


@pytest.fixture(scope="module")
def populated(spike_trains_scenario):
    """Populate SpikeTrains once for the scenario."""
    from aeon.dj_pipeline import processed_ephys

    scoped = {"experiment_name": spike_trains_scenario["experiment_name"]}
    processed_ephys.SpikeTrains.populate(scoped, suppress_errors=False)
    assert len(_mine(spike_trains_scenario)) == len(spike_trains_scenario["covered_chunk_starts"])
    return spike_trains_scenario


class TestMake:
    """What a populated row actually contains."""

    def test_conserves_spikes_against_the_source(self, populated):
        """Test that re-chunking loses and duplicates nothing.

        The single assertion that catches almost any boundary bug: every spike the
        scenario inserted appears exactly once across the rows.
        """
        total = sum(_mine(populated).to_arrays("n_spikes"))
        assert total == populated["expected_spike_counts"]["total"]

    def test_boundary_spike_lands_in_the_later_chunk(self, populated):
        """Test the half-open rule end to end, on a spike at exactly 09:00:00."""
        from swc.aeon.io import api as io_api

        boundary = populated["boundary_chunk_start"]
        later = (_mine(populated) & {"chunk_start": boundary}).fetch1("spikes")
        earlier = (_mine(populated) & {"chunk_start": populated["covered_chunk_starts"][0]}).fetch1(
            "spikes"
        )

        edge = io_api.to_seconds(boundary)
        assert edge in later[1].t
        assert edge not in earlier[1].t

    def test_cross_block_unit_gets_a_shorter_denominator(self, populated):
        """Test that a unit only one block found reports honest coverage.

        Unit 2 exists only in block A, which stops at 09:20, so over the 09:00 chunk
        it was sorted for 1200 s against 2400 s for units 1 and 3. Reporting the
        chunk's own coverage for it would halve its apparent firing rate.
        """
        row = (_mine(populated) & {"chunk_start": populated["boundary_chunk_start"]}).fetch1()
        covered = row["spikes"].get_info("covered_seconds")

        assert covered[populated["partial_unit"]] == 1200.0
        for unit in populated["full_units"]:
            assert covered[unit] == 2400.0
        assert row["n_partial_units"] == 1

    def test_ephys_gap_becomes_a_two_interval_support(self, populated):
        """Test that a gap survives into time_support rather than being papered over."""
        row = (_mine(populated) & {"chunk_start": populated["boundary_chunk_start"]}).fetch1()

        assert len(row["spikes"].time_support) == 2
        assert row["coverage_frac"] == pytest.approx(2400 / 3600, rel=1e-4)

    def test_roster_grows_as_blocks_discover_units(self, populated):
        """Test that the roster is stable, and that a later block's units appear.

        An unstable roster makes concatenating a span wrong by default, which is the
        motivating use case for the whole table.
        """
        first, second = (
            set((_mine(populated) & {"chunk_start": start}).fetch1("spikes").index)
            for start in populated["covered_chunk_starts"]
        )
        assert first == {1, 2}
        assert second == {1, 2, 3}

    def test_cross_block_unit_takes_the_owning_block_s_quality(self, populated):
        """Test that a cross-block unit takes its owning block's quality label.

        The two blocks disagree about unit 1 on purpose. In the 09:00 chunk block A
        owns 9 of its spikes to B's 8, so A's label wins. Wire ``owning_block`` in
        backwards and a user filtering ``unit_quality == "good"`` silently analyses
        the wrong neurons.
        """
        tsgroup = (_mine(populated) & {"chunk_start": populated["boundary_chunk_start"]}).fetch1("spikes")
        quality = tsgroup.get_info("unit_quality")

        assert quality[populated["cross_block_unit"]] == populated["owning_block_quality"]
        assert populated["owning_block_quality"] != populated["losing_block_quality"]

    def test_times_are_on_the_harp_epoch(self, populated):
        """Test that spikes land on seconds-since-1904, not seconds-since-anything-else."""
        for row in _mine(populated).to_dicts():
            assert row["spikes"].time_support.start[0] > 3.0e9


class TestStalenessAndFetchSpan:
    """The manual refresh that replaces the cascade, and the documented read path.

    Ordered last and sharing a module-scoped population: the staleness cases mutate
    upstream state, so they must not run before the tests that assert on a clean one.
    """

    def test_stale_is_detected_blocks_fetch_span_and_clears_on_the_recipe(self, populated):
        """Test the failure this design accepts, and its documented remedy.

        A row built from one of two covering blocks is right when written and wrong
        once the second lands. Nothing invalidates it, so it has to be findable.
        """
        from spike_train_factories import add_late_block

        from aeon.dj_pipeline import processed_ephys

        mine = {"experiment_name": populated["experiment_name"]}
        assert not processed_ephys.SpikeTrains.stale_chunks(mine)

        add_late_block(populated["experiment_name"])
        stale = processed_ephys.SpikeTrains.stale_chunks(mine)
        assert stale, "a block matched after the row was written must make it stale"

        # A span that excludes the stale chunk must still read. Restricting the
        # scan with a list makes it an OR in DataJoint, which would block this.
        later = populated["covered_chunk_starts"][1]
        processed_ephys.SpikeTrains.fetch_span(
            **populated["insertion_key"], start=later, end=populated["span_end"]
        )

        # While a stale row exists, fetch_span must refuse — that guard is the only
        # thing stopping someone analysing a window whose sorting has moved on.
        span = {
            **populated["insertion_key"],
            "start": populated["covered_chunk_starts"][0],
            "end": populated["span_end"],
        }
        with pytest.raises(ValueError, match="stale"):
            processed_ephys.SpikeTrains.fetch_span(**span)
        with pytest.warns(UserWarning, match="covered_seconds"):
            processed_ephys.SpikeTrains.fetch_span(**span, allow_stale=True)

        (processed_ephys.SpikeTrains() & stale).delete()
        processed_ephys.SpikeTrains.populate(mine, suppress_errors=False)
        assert not processed_ephys.SpikeTrains.stale_chunks(mine)

    def test_fetch_span_includes_the_chunk_it_starts_inside(self, populated):
        """Test that a span starting mid-chunk still returns that chunk's spikes.

        Selecting rows on chunk_start alone drops the chunk containing `start`
        entirely, so a window beginning at 08:30 silently loses 08:30-09:00.
        """
        from datetime import timedelta

        import numpy as np
        from swc.aeon.io import api as io_api

        from aeon.dj_pipeline import processed_ephys

        first = populated["covered_chunk_starts"][0]
        mid_chunk = first + timedelta(minutes=30)

        with pytest.warns(UserWarning, match="covered_seconds"):
            whole = processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"], start=first, end=populated["span_end"]
            )
        with pytest.warns(UserWarning, match="covered_seconds"):
            partial = processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"], start=mid_chunk, end=populated["span_end"]
            )

        whole_n = sum(len(whole[u]) for u in whole.index)
        partial_n = sum(len(partial[u]) for u in partial.index)
        cut = io_api.to_seconds(mid_chunk)
        before_cut = sum(int((np.asarray(whole[u].t) < cut).sum()) for u in whole.index)
        assert partial_n == whole_n - before_cut, (
            f"span from mid-chunk kept {partial_n} spikes; expected {whole_n - before_cut} "
            "(the whole span minus what precedes the cut)"
        )

    def test_fetch_span_scans_only_its_own_span(self, populated, monkeypatch):
        """Test that the staleness check is restricted to the span being read.

        An unrestricted scan walks every row in the table — all experiments, all
        insertions — at a couple of queries each, to read one hour.
        """
        from aeon.dj_pipeline import processed_ephys

        seen = {}
        original = processed_ephys.SpikeTrains.stale_chunks.__func__

        def spy(cls, restriction=True):
            seen["restriction"] = restriction
            return original(cls, restriction)

        monkeypatch.setattr(processed_ephys.SpikeTrains, "stale_chunks", classmethod(spy))
        with pytest.warns(UserWarning, match="covered_seconds"):
            processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"],
                start=populated["covered_chunk_starts"][0],
                end=populated["span_end"],
            )

        assert seen.get("restriction") is not True, "fetch_span scanned the whole table for staleness"

    def test_fetch_span_carries_unit_metadata(self, populated):
        """Test that the span object supports the filtering its docstring shows.

        fetch_span's own example is `tg[tg.unit_quality == "good"]`; dropping the
        metadata makes that a KeyError, and quality, electrode and shank are the
        columns an analysis actually filters on.
        """
        from aeon.dj_pipeline import processed_ephys

        with pytest.warns(UserWarning, match="covered_seconds"):
            tsgroup = processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"],
                start=populated["covered_chunk_starts"][0],
                end=populated["span_end"],
            )

        assert {"covered_seconds", "unit_quality", "electrode", "shank"} <= set(tsgroup.metadata_columns)
        # Unit 1 is the only cross-block unit, so it is the only one whose label
        # depends on the rule. Assert on it, or an inverted rule still passes.
        assert (
            tsgroup.get_info("unit_quality")[populated["cross_block_unit"]]
            == (populated["owning_block_quality"])
        )
        assert populated["owning_block_quality"] != populated["losing_block_quality"]

    def test_covered_seconds_never_exceeds_the_span(self, populated):
        """Test that a unit is not reported as observed for longer than the window.

        A chunk the window only partly covers contributes its whole stored
        covered_seconds, so a one-minute span inside an hour reports 3600 s and
        every rate in it comes out 60x low.
        """
        from datetime import timedelta

        from aeon.dj_pipeline import processed_ephys

        start = populated["covered_chunk_starts"][0] + timedelta(minutes=30)
        end = start + timedelta(minutes=1)
        tsgroup = processed_ephys.SpikeTrains.fetch_span(**populated["insertion_key"], start=start, end=end)

        spanned = (end - start).total_seconds()
        covered = tsgroup.get_info("covered_seconds")
        worst = max(float(covered[u]) for u in tsgroup.index)
        assert worst <= spanned, f"a unit reports {worst}s of coverage in a {spanned}s window"

    def test_fetch_span_keeps_the_ephys_gap(self, populated):
        """Test that a gap inside the span survives into the returned time_support.

        The 09:00 chunk has an ephys gap at 09:20-09:40. pynapple's restrict()
        replaces time_support with the restriction window rather than intersecting
        it, so rebuilding the span support from restricted.time_support reports a
        solid hour and every firing rate in it comes out low.
        """
        from aeon.dj_pipeline import processed_ephys

        with pytest.warns(UserWarning, match="covered_seconds"):
            tsgroup = processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"],
                start=populated["covered_chunk_starts"][0],
                end=populated["span_end"],
            )

        total = float(tsgroup.time_support.tot_length())
        spanned = (populated["span_end"] - populated["covered_chunk_starts"][0]).total_seconds()
        assert total < spanned, (
            f"time_support covers {total}s of a {spanned}s span; the 09:20-09:40 ephys gap was erased"
        )
        assert len(tsgroup.time_support) >= 2, "the gap should split the support"

    def test_fetch_span_concatenates_and_sums_covered_seconds(self, populated):
        """Test that a span returns one object and that the denominator composes.

        covered_seconds is stored in seconds rather than as a fraction precisely so
        this addition is correct; a fraction would need duration weighting that every
        caller would get wrong.
        """
        from aeon.dj_pipeline import processed_ephys

        with pytest.warns(UserWarning, match="covered_seconds"):
            tsgroup = processed_ephys.SpikeTrains.fetch_span(
                **populated["insertion_key"],
                start=populated["covered_chunk_starts"][0],
                end=populated["span_end"],
            )

        assert set(tsgroup.index) >= {1, 2, 3}
        covered = tsgroup.get_info("covered_seconds")
        assert covered[1] > covered[2]  # unit 2 is the one block A alone found
