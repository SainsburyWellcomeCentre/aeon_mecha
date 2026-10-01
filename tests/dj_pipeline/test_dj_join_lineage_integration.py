"""DataJoint 2.x rejects joins on same-named attributes with no common lineage.

SortedSpikes.Unit and SyncedSpikes.Unit each declare a secondary `spike_count`
independently. Restricting one by the other without .proj() raises. This pins the
rule so the projection in _load_block_unit_spike_trains is not removed again.
See PR #609 for the same bug class in SpikeSorting.make_insert.
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

    non_noise = Parent - {"unit_quality": "noise"}
    assert len(non_noise) == 1

    with pytest.raises(dj.DataJointError, match="spike_count"):
        len(Child & non_noise)

    assert len(Child & non_noise.proj()) == 1

    schema.drop()
