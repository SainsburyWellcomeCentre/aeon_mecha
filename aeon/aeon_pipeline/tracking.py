import datajoint as dj

from . import experiment
from . import get_schema_name


schema = dj.schema(get_schema_name('tracking'))


@schema
class AnimalPosition(dj.Imported):
    definition = """
    -> experiment.SubjectEpoch
    ---
    timestamps:        longblob  # (s) timestamps of the position data, w.r.t the start of the TimeBlock containing this Epoch
    position_x:        longblob  # (m) animal's x-position, in the arena's coordinate frame
    position_y:        longblob  # (m) animal's y-position, in the arena's coordinate frame
    position_z=null:   longblob  # (m) animal's z-position, in the arena's coordinate frame
    speed:             longblob  # (m/s) speed
    """


@schema
class EpochPosition(dj.Computed):
    definition = """  # All unique positions (x,y,z) of an animal in a given epoch
    -> AnimalPosition
    x: decimal(5, 3)
    y: decimal(5, 3)
    z: decimal(5, 3)
    """

    def make(self, key):
        unique_positions = set((AnimalPosition & key).fetch1(
            'position_x', 'position_y', 'position_z'))
        self.insert([{**key, 'x': x, 'y': y, 'z': z}
                     for x, y, z in unique_positions])
