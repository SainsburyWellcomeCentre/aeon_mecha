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

    def test_earlier_block_owns_the_overlap(self, ephys_unit_matching_populated, ctx):
        """For a global unit in both blocks, block 1 owns the shared chunks.

        Uniqueness of (global_unit, chunk_start) is already enforced by a UNIQUE index on
        Spikes, so asserting it here could never fail - a duplicate would raise an
        IntegrityError during populate. What the index does NOT enforce is WHICH block wins,
        and that an implementation writing zero Spikes rows for block 2 would be caught.
        """
        exp = {"experiment_name": ctx.cfg["experiment_name"]}
        blocks = ephys_unit_matching_populated["blocks"]
        per_block = [
            {
                int(g)
                for g in (
                    ctx.spike_sorting.UnitMatching.Unit & exp & {"block_start": b["block_start"]}
                ).to_arrays("global_unit")
            }
            for b in blocks
        ]
        shared = per_block[0] & per_block[1]
        assert shared, "no global unit spans the overlap - ownership is untested"

        rows = (ctx.spike_sorting.UnitMatching.Spikes & exp).to_dicts()
        assert rows, "Spikes is empty"

        # Every chunk of the epoch that either block covers must be owned exactly once, and
        # the union must extend past block 1's own chunks - i.e. block 2 contributed.
        block1_chunks = {
            c["chunk_start"]
            for c in (ctx.ephys.EphysBlockInfo.Chunk & blocks[0]).to_dicts()
        }
        all_owned = {r["chunk_start"] for r in rows}
        assert all_owned - block1_chunks, (
            "no Spikes rows outside block 1's chunks - block 2 contributed nothing"
        )

        # For a unit present in both blocks, the shared chunks are owned by block 1.
        gu = next(iter(shared))
        owned_by_block = {
            (r["chunk_start"], r["block_start"]) for r in rows if r["global_unit"] == gu
        }
        for chunk_start, owner in owned_by_block:
            if chunk_start in block1_chunks:
                assert owner == blocks[0]["block_start"], (
                    f"chunk {chunk_start} of global unit {gu} is owned by {owner}, "
                    f"expected the earlier block {blocks[0]['block_start']}"
                )

    def test_global_unit_ids_contiguous_from_one(self, ephys_unit_matching_populated, ctx):
        ids = sorted(
            int(i)
            for i in (
                ctx.spike_sorting.GlobalUnit & {"experiment_name": ctx.cfg["experiment_name"]}
            ).to_arrays("global_unit")
        )
        assert ids, "GlobalUnit is empty - nothing was matched"
        assert ids == list(range(1, len(ids) + 1))

    def test_noise_units_excluded(self, ephys_unit_matching_populated, ctx):
        """Units labelled noise stay in SortedSpikes but get no global identity."""
        for block_start, noise_units in ephys_unit_matching_populated["noise_units"].items():
            matched = (
                ctx.spike_sorting.UnitMatching.Unit
                & {"experiment_name": ctx.cfg["experiment_name"], "block_start": block_start}
            ).to_arrays("unit")
            matched_ids = {int(u) for u in matched}
            assert matched_ids, f"no units matched in block {block_start} - assertion is vacuous"
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
        try:
            with pytest.raises(ValueError, match="seed"):
                ctx.spike_sorting.UnitMatching().make(bad_key)
        finally:
            # UnitMatching.key_source is `eligible * UnitMatchingParamSet`, so a leaked
            # paramset permanently widens it for the session and makes the suite order-
            # dependent (any -k selection running this first would double the populated rows).
            (
                ctx.spike_sorting.UnitMatchingParamSet & {"matching_paramset_id": 99}
            ).delete_quick()


class TestUnitMatchingBehaviour:
    def test_overlap_produces_at_least_one_match(
        self, ephys_unit_matching_populated, ctx, record_property
    ):
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
        record_property("shared_global_units", len(shared))
        print(f"\n  shared global units across overlap: {len(shared)}")
