"""Auto-approved curation: ApplyOfficialCuration must be a no-op on the raw sorting.

The golden fixture registers a ManualCuration with parent_curation_id = -1 and no curation
file, which is the state ApplyOfficialCuration's auto-approve branch is written for. It
records the approval and returns, leaving SortedSpikes on the raw sorting. The real curation
chain (an actual curation file, apply_curation, re-derived SortedSpikes) is out of scope.
"""

import pytest

# Specialized, not integration: ephys_curation_applied drives the same ~18-minute chain
# (PostProcessing + SortedSpikes + SyncedSpikes over 2 blocks x 96 channels) as the matching
# tests. Keeping the two files in the same tier means one fixture build, not two.
pytestmark = pytest.mark.specialized


class TestAutoApprovedCuration:
    def test_apply_is_a_no_op(self, ephys_curation_applied, ctx):
        rows = (
            ctx.spike_sorting_curation.ApplyOfficialCuration
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(rows) == 2
        for row in rows:
            assert row["new_unit_count"] == 0
            assert row["removed_unit_count"] == 0

    def test_sorted_spikes_stays_on_the_raw_sorting(self, ephys_curation_applied, ctx):
        """Auto-approval records the decision; it must not re-derive SortedSpikes."""
        curation_ids = (
            ctx.spike_sorting.SortedSpikes & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_arrays("curation_id")
        assert {int(c) for c in curation_ids} == {-1}

    def test_unit_set_unchanged_by_apply(self, ephys_curation_applied, ctx):
        """Every injected unit survives the apply - count matched against the artifact.

        A bare `> 0` would pass if the apply deleted all but one unit, which is exactly the
        failure mode worth catching here.
        """
        import spikeinterface as si

        for block in ephys_curation_applied["blocks"]:
            block_key = {
                k: block[k]
                for k in ("experiment_name", "subject", "insertion_number", "block_start", "block_end")
            }
            sorting_dir = ephys_curation_applied["sorting_dirs"][block["block_start"]]
            expected = len(si.load(sorting_dir / "in_container_sorting").unit_ids)
            actual = len(ctx.spike_sorting.SortedSpikes.Unit & block_key)
            assert actual == expected, (
                f"Block {block['block_start']}: {actual} units in SortedSpikes, "
                f"{expected} in the injected artifact"
            )

    def test_noise_units_are_marked(self, ephys_noise_units_marked, ctx):
        """Two units per block, four in total, carry unit_quality='noise'."""
        noise = (
            ctx.spike_sorting.SortedSpikes.Unit
            & {"experiment_name": ctx.cfg["experiment_name"], "unit_quality": "noise"}
        )
        assert len(noise) == 4
        assert sum(len(v) for v in ephys_noise_units_marked.values()) == 4
