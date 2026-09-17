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

        keys = processed_ephys.SpikeTrains.key_source.to_dicts()
        starts = sorted({k["chunk_start"] for k in keys})

        assert starts == spike_trains_scenario["covered_chunk_starts"]
        assert spike_trains_scenario["uncovered_chunk_start"] not in starts
