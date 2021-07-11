import datajoint as dj
import pandas as pd
import numpy as np
from matplotlib import path

from aeon.preprocess import api as aeon_api

from . import lab, experiment, tracking
from . import get_schema_name


schema = dj.schema(get_schema_name('analysis'))


@schema
class SessionStatistics(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    time_fraction_in_nest: float  # fraction of time the animal spent in the nest in this session
    distance_travelled: float  # total distance the animal travelled during this session
    """

    class FoodPatchStatistics(dj.Part):
        definition = """
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        in_patch_timestamps: longblob  # timestamps of the time the animal spent on this patch
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        total_wheel_distance_travelled: float  # total wheel travel distance during this session
        """

    # Work on finished Session with SubjectDistance fully populated only
    key_source = (experiment.Session
                  & (experiment.Session * experiment.SessionEnd * experiment.SessionEpoch
                     & tracking.SubjectDistance & 'epoch_end = session_end').proj()
                  )

    # Animal's distance from the food-patch position to be considered "time spent in the food patch"
    distance_threshold = 80

    def make(self, key):
        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)
        session_start, session_end = (experiment.Session * experiment.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject's position data in the epochs
        timestamps, position_x, position_y, speed, area = (
                tracking.SubjectPosition & key).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

        # stack and structure in pandas DataFrame
        position = pd.DataFrame(dict(x=np.hstack(position_x),
                                     y=np.hstack(position_y),
                                     speed=np.hstack(speed),
                                     area=np.hstack(area)),
                                index=np.hstack(timestamps))

        position_diff = np.sqrt(np.square(np.diff(position.x)) + np.square(np.diff(position.y)))
        distance_travelled = np.nancumsum(position_diff)[-1]

        is_in_nest = is_position_in_nest(key, position.x, position.y)
        time_fraction_in_nest = sum(is_in_nest) / len(is_in_nest)

        # food patch data
        food_patch_keys = (
                experiment.Session * experiment.SessionEnd
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistics = []
        for food_patch_key in food_patch_keys:
            distance = (tracking.SubjectDistance.FoodPatch
                        & key & food_patch_key).fetch('distance')
            distance = pd.DataFrame(dict(distance=np.hstack(distance)),
                                    index=np.hstack(timestamps))

            distance['is_in_patch'] = np.logical_and(~np.isnan(distance.distance),
                                                     distance.distance <= self.distance_threshold)

            time_fraction_in_patch = sum(distance.is_in_patch) / len(distance.is_in_patch)

            # wheel data
            food_patch_description = (experiment.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle).values

            food_patch_statistics.append({
                **key, **food_patch_key,
                'in_patch_timestamps': np.hstack(timestamps)[distance.is_in_patch],
                'time_fraction_in_patch': time_fraction_in_patch,
                'total_wheel_distance_travelled': wheel_distance_travelled[-1]})

        self.insert1({**key,
                      'time_fraction_in_nest': time_fraction_in_nest,
                      'distance_travelled': distance_travelled})
        self.FoodPatchStatistics.insert(food_patch_statistics)


def is_position_in_nest(experiment_key, position_x, position_y):
    """
    Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest
    """

    assert len(position_x) == len(position_y), f'Mismatch length in x and y'

    nest_vertices = list(zip(*(lab.ArenaNest.Vertex & experiment_key).fetch(
        'vertex_x', 'vertex_y')))

    nest_path = path.Path(nest_vertices)

    return nest_path.contains_points(np.vstack([position_x, position_y]).T)
