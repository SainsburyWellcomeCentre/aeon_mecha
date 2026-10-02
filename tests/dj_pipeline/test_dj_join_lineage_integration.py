"""DataJoint 2.x rejects joins on same-named attributes with no common lineage.

SortedSpikes.Unit and SyncedSpikes.Unit each declare a secondary `spike_count`
independently, which is why _load_block_unit_spike_trains must .proj() its restricting
operand. See PR #609 for the same bug class in SpikeSorting.make_insert.

SCOPE: this pins the DataJoint LIBRARY behaviour on two throwaway tables - it would stay
green if the production .proj() were removed. The end-to-end guard for that is
tests/dj_pipeline/test_unit_matching_integration.py, whose fixtures cannot populate
without it.
"""

import pytest

pytestmark = pytest.mark.integration


def test_namesake_secondary_attribute_requires_projection(dj_config_integration):
    import datajoint as dj

    schema = dj.Schema(dj.config.database.database_prefix + "lineage_guard")

    @schema
    class Parent(dj.Manual):
        definition = """
        pid: int
        ---
        spike_count: int
        unit_quality: varchar(32)
        """

    @schema
    class Child(dj.Manual):
        definition = """
        -> Parent
        cid: int
        ---
        spike_count: int
        """

    Parent.insert(
        [
            {"pid": 1, "spike_count": 100, "unit_quality": "good"},
            {"pid": 2, "spike_count": 200, "unit_quality": "noise"},
        ]
    )
    Child.insert(
        [
            {"pid": 1, "cid": 0, "spike_count": 7},
            {"pid": 2, "cid": 0, "spike_count": 9},
        ]
    )

    try:
        non_noise = Parent - {"unit_quality": "noise"}
        assert len(non_noise) == 1

        # Match the lineage condition, not just the attribute name.
        with pytest.raises(dj.DataJointError, match=r"(?is)spike_count.*lineage|lineage.*spike_count"):
            len(Child & non_noise)

        assert len(Child & non_noise.proj()) == 1
    finally:
        # Unconditional: a leaked schema breaks the next run under TEST_DB_PREFIX.
        schema.drop()
