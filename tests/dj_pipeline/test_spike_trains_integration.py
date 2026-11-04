"""Integration tests for processed_ephys.SpikeTrains on testcontainers MySQL.

Synthetic on purpose. The scenario in ``tests/fixtures/ephys/spike_train_factories.py``
puts a two-block chunk, an ephys gap and a boundary spike exactly where an assertion
can see them. No fixed real recording obliges.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def spike_trains_scenario(ephys_full_pipeline, tmp_path_factory):
    """Build the behaviour + ephys scenario.

    Takes ``dj_store`` from ``ephys_full_pipeline``: ``dj.config.stores`` is global
    and assigning it replaces the whole mapping, so a second fixture pointing
    ``dj_store`` elsewhere makes the first's rows unreadable.
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

    The golden suite populates the same table and runs first, so an unrestricted
    query here picks up its rows.
    """
    from aeon.dj_pipeline import processed_ephys

    return processed_ephys.SpikeTrains & {"experiment_name": scenario["experiment_name"]}


class TestKeySource:
    """What is computable, and what is deliberately not."""

    def test_only_chunks_overlapping_matched_ephys_are_computable(self, spike_trains_scenario):
        """The key_source is the overlap, not the cross product.

        Three behavioural chunks; ephys covers all three but the 10:00 one was never
        matched. Excluding it is therefore a statement about matching, not about
        ephys being absent. An empty key_source is not an error in DataJoint, so
        assert the positive count too.
        """
        from aeon.dj_pipeline import ephys, processed_ephys

        scoped = {"experiment_name": spike_trains_scenario["experiment_name"]}
        keys = (processed_ephys.SpikeTrains().key_source & scoped).to_dicts()
        starts = sorted({k["chunk_start"] for k in keys})

        assert starts == spike_trains_scenario["covered_chunk_starts"]
        assert spike_trains_scenario["uncovered_chunk_start"] not in starts
        assert ephys.EphysChunk & {
            **spike_trains_scenario["insertion_key"],
            "chunk_start": spike_trains_scenario["unmatched_ephys_start"],
        }, "the unmatched ephys chunk is missing, so the exclusion above proves nothing"

    def test_matching_is_per_chunk_not_per_insertion(self, spike_trains_scenario):
        """One matched block must not make every chunk computable.

        Renaming chunk_start before the semijoin drops it from the heading, and the
        restriction degenerates to "this insertion matched something, somewhere".
        """
        from aeon.dj_pipeline import acquisition, ephys, processed_ephys, spike_sorting

        scoped = {"experiment_name": spike_trains_scenario["experiment_name"]}
        computable = (processed_ephys.SpikeTrains().key_source & scoped).to_dicts()
        assert computable, "nothing computable, so the loop below would pass vacuously"
        for key in computable:
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
        """Every inserted spike appears exactly once across the rows.

        The single assertion that catches almost any boundary bug.
        """
        total = sum(_mine(populated).to_arrays("n_spikes"))
        assert total == populated["expected_spike_counts"]["total"]

    def test_boundary_spike_lands_in_the_later_chunk(self, populated):
        """The half-open rule end to end, on a spike at exactly 09:00:00."""
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
        """A unit only one block found reports honest coverage.

        Unit 2 is in block A alone, which stops at 09:20: 1200 s against 2400 s for
        units 1 and 3. The chunk's own coverage would halve its apparent rate.
        """
        row = (_mine(populated) & {"chunk_start": populated["boundary_chunk_start"]}).fetch1()
        covered = row["spikes"].get_info("covered_seconds")

        assert covered[populated["partial_unit"]] == 1200.0
        for unit in populated["full_units"]:
            assert covered[unit] == 2400.0
        assert row["n_partial_units"] == 1
        # The 09:20-09:40 ephys gap, which the chunk's own coverage would paper over.
        assert row["coverage_frac"] == pytest.approx(2400 / 3600, rel=1e-4)

    def test_roster_grows_as_blocks_discover_units(self, populated):
        """The roster is stable, and a later block's units appear.

        An unstable roster makes concatenating a span wrong by default.
        """
        first, second = (
            set((_mine(populated) & {"chunk_start": start}).fetch1("spikes").index)
            for start in populated["covered_chunk_starts"]
        )
        assert first == {1, 2}
        assert second == {1, 2, 3}

    def test_cross_block_unit_takes_the_owning_block_s_quality(self, populated):
        """A cross-block unit takes its owning block's quality label.

        The blocks disagree about unit 1 on purpose; block A owns 9 of its spikes to
        B's 8. Wire the rule in backwards and someone filtering ``unit_quality ==
        "good"`` silently analyses the wrong neurons.
        """
        tsgroup = (_mine(populated) & {"chunk_start": populated["boundary_chunk_start"]}).fetch1("spikes")
        quality = tsgroup.get_info("unit_quality")

        assert quality[populated["cross_block_unit"]] == populated["owning_block_quality"]
        assert populated["owning_block_quality"] != populated["losing_block_quality"]
