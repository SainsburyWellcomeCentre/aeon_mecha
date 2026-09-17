"""Integration tests for processed_ephys.SpikeTrains on testcontainers MySQL.

Synthetic rather than golden, deliberately: the scenario in
``tests/fixtures/ephys/spike_train_factories.py`` puts a two-block chunk, an ephys
gap and a boundary spike where assertions can see them, which no fixed real
recording does.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def spike_trains_scenario(dj_config_integration, tmp_path_factory):
    """Build the behaviour + ephys scenario and return what the tests assert against."""
    import datajoint as dj
    from spike_train_factories import build_scenario

    store_dir = tmp_path_factory.mktemp("dj_store")
    dj.config.stores = {
        "dj_store": {"protocol": "file", "location": str(store_dir), "stage": str(store_dir)}
    }

    tmp_path = tmp_path_factory.mktemp("repo")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    from ephys_factories import register_synthetic_experiment

    experiment_name = "synthetic-spike-trains"
    register_synthetic_experiment(tmp_path, raw_dir, experiment_name, "2026-05-11T08-00-00")
    return build_scenario(experiment_name)


class TestKeySource:
    """What is computable, and what is deliberately not."""

    def test_only_chunks_overlapping_matched_ephys_are_computable(self, spike_trains_scenario):
        """Test that the key_source is the overlap, not the cross product.

        Three behavioural chunks are registered but ephys covers only two. The third
        must not appear — there is nothing to compute it from. An empty key_source is
        not an error in DataJoint, so this asserts a positive count too.
        """
        from aeon.dj_pipeline import processed_ephys

        keys = processed_ephys.SpikeTrains().key_source.to_dicts()
        starts = sorted({k["chunk_start"] for k in keys})

        assert starts == spike_trains_scenario["covered_chunk_starts"]
        assert spike_trains_scenario["uncovered_chunk_start"] not in starts


@pytest.fixture(scope="module")
def populated(spike_trains_scenario):
    """Populate SpikeTrains once for the scenario."""
    from aeon.dj_pipeline import processed_ephys

    processed_ephys.SpikeTrains.populate(suppress_errors=False)
    assert len(processed_ephys.SpikeTrains()) == len(spike_trains_scenario["covered_chunk_starts"])
    return spike_trains_scenario


class TestMake:
    """What a populated row actually contains."""

    def test_conserves_spikes_against_the_source(self, populated):
        """Test that re-chunking loses and duplicates nothing.

        The single assertion that catches almost any boundary bug: every spike the
        scenario inserted appears exactly once across the rows.
        """
        from aeon.dj_pipeline import processed_ephys

        total = sum(processed_ephys.SpikeTrains().to_arrays("n_spikes"))
        assert total == populated["expected_spike_counts"]["total"]

    def test_boundary_spike_lands_in_the_later_chunk(self, populated):
        """Test the half-open rule end to end, on a spike at exactly 09:00:00."""
        from swc.aeon.io import api as io_api

        from aeon.dj_pipeline import processed_ephys

        boundary = populated["boundary_chunk_start"]
        later = (processed_ephys.SpikeTrains() & {"chunk_start": boundary}).fetch1("spikes")
        earlier = (
            processed_ephys.SpikeTrains() & {"chunk_start": populated["covered_chunk_starts"][0]}
        ).fetch1("spikes")

        edge = io_api.to_seconds(boundary)
        assert edge in later[1].t
        assert edge not in earlier[1].t

    def test_cross_block_unit_gets_a_shorter_denominator(self, populated):
        """Test that a unit only one block found reports honest coverage.

        Unit 2 exists only in block A, which stops at 09:20, so over the 09:00 chunk
        it was sorted for 1200 s against 2400 s for units 1 and 3. Reporting the
        chunk's own coverage for it would halve its apparent firing rate.
        """
        from aeon.dj_pipeline import processed_ephys

        row = (processed_ephys.SpikeTrains() & {"chunk_start": populated["boundary_chunk_start"]}).fetch1()
        covered = row["spikes"].get_info("covered_seconds")

        assert covered[populated["partial_unit"]] == 1200.0
        for unit in populated["full_units"]:
            assert covered[unit] == 2400.0
        assert row["n_partial_units"] == 1

    def test_ephys_gap_becomes_a_two_interval_support(self, populated):
        """Test that a gap survives into time_support rather than being papered over."""
        from aeon.dj_pipeline import processed_ephys

        row = (processed_ephys.SpikeTrains() & {"chunk_start": populated["boundary_chunk_start"]}).fetch1()

        assert len(row["spikes"].time_support) == 2
        assert row["coverage_frac"] == pytest.approx(2400 / 3600, rel=1e-4)

    def test_roster_grows_as_blocks_discover_units(self, populated):
        """Test that the roster is stable, and that a later block's units appear.

        An unstable roster makes concatenating a span wrong by default, which is the
        motivating use case for the whole table.
        """
        from aeon.dj_pipeline import processed_ephys

        first, second = (
            set((processed_ephys.SpikeTrains() & {"chunk_start": start}).fetch1("spikes").index)
            for start in populated["covered_chunk_starts"]
        )
        assert first == {1, 2}
        assert second == {1, 2, 3}

    def test_times_are_on_the_harp_epoch(self, populated):
        """Test that spikes land on seconds-since-1904, not seconds-since-anything-else."""
        from aeon.dj_pipeline import processed_ephys

        for row in processed_ephys.SpikeTrains().to_dicts():
            assert row["spikes"].time_support.start[0] > 3.0e9


class TestStalenessAndFetchSpan:
    """The manual refresh that replaces the cascade, and the documented read path.

    Ordered last and sharing a module-scoped population: the staleness cases mutate
    upstream state, so they must not run before the tests that assert on a clean one.
    """

    def test_stale_detects_a_later_block_and_the_recipe_clears_it(self, populated):
        """Test the failure this design accepts, and its documented remedy.

        A row built from one of two covering blocks is right when written and wrong
        once the second lands. Nothing invalidates it, so it has to be findable.
        """
        from spike_train_factories import add_late_block

        from aeon.dj_pipeline import processed_ephys

        assert not processed_ephys.SpikeTrains.stale()

        add_late_block(populated["experiment_name"])
        stale = processed_ephys.SpikeTrains.stale()
        assert stale, "a block matched after the row was written must make it stale"

        (processed_ephys.SpikeTrains() & stale).delete()
        processed_ephys.SpikeTrains.populate(suppress_errors=False)
        assert not processed_ephys.SpikeTrains.stale()

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
