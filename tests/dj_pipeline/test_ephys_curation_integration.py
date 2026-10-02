"""Auto-approved curation: ApplyOfficialCuration must be a no-op on the raw sorting.

ManualCuration with parent_curation_id = -1 and no curation file triggers the auto-approve
branch. The real curation chain is out of scope.
"""

import pytest

# Specialized: drives the same ~18-minute chain as the matching tests, so one tier means
# one fixture build.
pytestmark = pytest.mark.specialized


class TestAutoApprovedCuration:
    def test_apply_is_a_no_op(self, ephys_curation_applied, ctx):
        rows = (
            ctx.spike_sorting_curation.ApplyOfficialCuration
            & {"experiment_name": ctx.cfg["experiment_name"]}
        ).to_dicts()
        assert len(rows) == len(ephys_curation_applied["blocks"])
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
        """Every injected unit survives the apply, matched against the artifact."""
        import spikeinterface as si

        for block in ephys_curation_applied["blocks"]:
            # Full sorting key: block alone counts every electrode group and paramset.
            block_key = {
                **{
                    k: block[k]
                    for k in ("experiment_name", "subject", "insertion_number", "block_start", "block_end")
                },
                "electrode_group": ephys_curation_applied["electrode_group"],
                "paramset_id": ephys_curation_applied["paramset_id"],
            }
            sorting_dir = ephys_curation_applied["sorting_dirs"][block["block_start"]]
            expected = {int(u) for u in si.load(sorting_dir / "in_container_sorting").unit_ids}
            actual = {
                int(u) for u in (ctx.spike_sorting.SortedSpikes.Unit & block_key).to_arrays("unit")
            }
            assert actual == expected, (
                f"Block {block['block_start']}: SortedSpikes units differ from the artifact "
                f"(missing {sorted(expected - actual)[:5]}, extra {sorted(actual - expected)[:5]})"
            )
