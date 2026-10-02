"""UnitMatching / GlobalUnit on the golden dataset: two overlapping blocks, one shank.

Structural invariants from docs/specs/SPEC_UNIT_MATCHING.md are asserted hard. Which units
actually matched comes from compare_two_sorters on real data and will drift with any
spikeinterface bump, so that is asserted as a floor rather than an exact count.

Marked `specialized` rather than `integration`: the full chain behind these fixtures
(PostProcessing over ~2M spikes x 96 channels for two blocks) measured ~18 minutes.
"""

import pytest

pytestmark = pytest.mark.specialized


class TestUnitMatchingStructure:
    def test_both_blocks_matched(self, ephys_unit_matching_populated, ctx):
        rows = ctx.spike_sorting.UnitMatching & {"experiment_name": ctx.cfg["experiment_name"]}
        assert len(rows) == 2

    def test_spikes_ownership_invariant(self, ephys_unit_matching_populated, ctx):
        """Each (global_unit, chunk_start) appears at most once across all blocks.

        Blocks 1 and 2 share chunks 4-6, so without the ownership rule the overlap would
        produce duplicate Spikes rows and inflate downstream firing rates. The spec calls
        this *the* invariant and backs it with a unique index.
        """
        pairs = (
            ctx.spike_sorting.UnitMatching.Spikes & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        seen = [(r["global_unit"], r["chunk_start"]) for r in pairs]
        assert len(seen) == len(set(seen)), "duplicate (global_unit, chunk_start) in Spikes"

    def test_global_unit_ids_contiguous_from_one(self, ephys_unit_matching_populated, ctx):
        ids = sorted(
            int(i)
            for i in (
                ctx.spike_sorting.GlobalUnit & {"experiment_name": ctx.cfg["experiment_name"]}
            ).to_arrays("global_unit")
        )
        assert ids == list(range(1, len(ids) + 1))

    def test_noise_units_excluded(self, ephys_unit_matching_populated, ctx):
        """Units labelled noise stay in SortedSpikes but get no global identity."""
        for block_start, noise_units in ephys_unit_matching_populated["noise_units"].items():
            matched = (
                ctx.spike_sorting.UnitMatching.Unit
                & {"experiment_name": ctx.cfg["experiment_name"], "block_start": block_start}
            ).to_arrays("unit")
            matched_ids = {int(u) for u in matched}
            for unit in noise_units:
                assert unit not in matched_ids


class TestUnitMatchingGuards:
    def test_non_seed_first_block_raises(self, ephys_curation_applied, ctx):
        """make() must refuse a first block that is not the seed."""
        blocks = ephys_curation_applied["blocks"]
        ctx.spike_sorting.UnitMatchingParamSet.insert1(
            {
                "matching_paramset_id": 99,
                "matching_method": "spike_time_overlap",
                "seed_block_start": blocks[0]["block_start"],
                "matching_paramset_description": "guard test",
                "params": {},
            },
            skip_duplicates=True,
        )
        bad_key = {
            **{
                k: blocks[1][k]
                for k in ("experiment_name", "subject", "insertion_number", "block_start", "block_end")
            },
            "electrode_group": "shank3",
            "paramset_id": "400",
            "matching_paramset_id": 99,
        }
        with pytest.raises(ValueError, match="seed"):
            ctx.spike_sorting.UnitMatching().make(bad_key)


class TestUnitMatchingBehaviour:
    def test_overlap_produces_at_least_one_match(self, ephys_unit_matching_populated, ctx):
        """Blocks 1 and 2 share 3 chunks, so some units should be the same neuron.

        A floor, not an exact count: the number comes from compare_two_sorters on real data
        and will move with any spikeinterface version bump.
        """
        blocks = ephys_unit_matching_populated["blocks"]
        per_block = []
        for block in blocks:
            per_block.append(
                {
                    int(g)
                    for g in (
                        ctx.spike_sorting.UnitMatching.Unit
                        & {
                            "experiment_name": ctx.cfg["experiment_name"],
                            "block_start": block["block_start"],
                        }
                    ).to_arrays("global_unit")
                }
            )
        shared = per_block[0] & per_block[1]
        assert len(shared) >= 1, (
            f"No global units shared across the overlap "
            f"(block1={len(per_block[0])}, block2={len(per_block[1])})"
        )
        print(f"\n  shared global units across overlap: {len(shared)}")
