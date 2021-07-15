import datajoint as dj
import pandas as pd
import numpy as np
from matplotlib import path
import datetime

from aeon.preprocess import api as aeon_api

from . import lab, experiment, tracking
from . import get_schema_name


schema = dj.schema(get_schema_name('analysis'))


@schema
class SessionTimeDistribution(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    time_fraction_in_corridor: float  # fraction of time the animal spent in the corridor in this session
    in_corridor_timestamps: longblob  # timestamps of the time the animal spent in the corridor
    time_fraction_in_arena: float     # fraction of time the animal spent in the arena in this session
    in_arena_timestamps: longblob     # timestamps of the time the animal spent the arena
    """

    class Nest(dj.Part):
        definition = """  # Time spent in nest
        -> master
        -> lab.ArenaNest
        ---
        time_fraction_in_nest: float  # fraction of time the animal spent in this nest in this session
        in_nest_timestamps: longblob  # timestamps of the time the animal spent in this nest
        """

    class FoodPatch(dj.Part):
        definition = """ # Time spent in food patch
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        in_patch_timestamps: longblob  # timestamps of the time the animal spent on this patch
        """

    # Work on finished Session with SubjectPosition fully populated only
    key_source = (experiment.Session
                  & (experiment.Session * experiment.SessionEnd * experiment.SessionEpoch
                     & tracking.SubjectPosition & 'epoch_end = session_end').proj())

    def make(self, key):
        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)
        session_start, session_end = (experiment.Session * experiment.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject's position data in the epochs
        timestamps, position_x, position_y, speed, area = (
                tracking.SubjectPosition & key).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

        # stack and structure in pandas DataFrame
        position_df = pd.DataFrame(dict(x=np.hstack(position_x),
                                        y=np.hstack(position_y),
                                        speed=np.hstack(speed),
                                        area=np.hstack(area)),
                                   index=np.hstack(timestamps))

        valid_position = (position_df.area > 0) & (
                    position_df.area < 1000)  # filter for objects of the correct size
        position_df = position_df[valid_position]
        position_df.x = position_df.x * tracking.pixel_scale
        position_df.y = position_df.y * tracking.pixel_scale

        timestamps = (position_df.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        # in corridor
        distance_from_center = tracking.compute_distance(
            position_df[['x', 'y']],
            (tracking.arena_center_x, tracking.arena_center_y))
        in_corridor = (distance_from_center < tracking.arena_outer_radius) & (distance_from_center > tracking.arena_inner_radius)

        in_arena = ~in_corridor

        # in nests
        in_nest_times = []
        for nest_key in (lab.ArenaNest & key).fetch('KEY'):
            in_nest = is_position_in_nest(position_df, nest_key)
            in_nest_times.append(
                {**key, **nest_key,
                 'time_fraction_in_nest': in_nest.mean(),
                 'in_nest_timestamps': timestamps[in_nest]})
            in_arena = in_arena & ~in_nest

        # in food patches
        food_patch_keys = (
                experiment.Session * experiment.SessionEnd
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        in_food_patch_times = []
        for food_patch_key in food_patch_keys:
            # wheel data
            food_patch_description = (experiment.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle)

            patch_position = (experiment.ExperimentFoodPatch.Position & food_patch_key).fetch1(
                'food_patch_position_x', 'food_patch_position_y')

            in_patch = tracking.is_in_patch(position_df, patch_position,
                                            wheel_distance_travelled, patch_radius=0.2)

            in_food_patch_times.append({
                **key, **food_patch_key,
                'time_fraction_in_patch': in_patch.mean(),
                'in_patch_timestamps': timestamps[in_patch]})

            in_arena = in_arena & ~in_patch

        self.insert1({**key,
                      'time_fraction_in_corridor': in_corridor.mean(),
                      'in_corridor_timestamps': timestamps[in_corridor],
                      'time_fraction_in_arena': in_arena.mean(),
                      'in_arena_timestamps': timestamps[in_arena]})
        self.Nest.insert(in_nest_times)
        self.FoodPatch.insert(in_food_patch_times)


@schema
class SessionSummary(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    total_distance_travelled: float  # (m) total distance the animal travelled during this session
    total_pellet_count: int  # total pellet delivered for all patches during this session
    total_wheel_distance_travelled: float  # total wheel distance for all patches
    change_in_weight: float  # weight change before/after the session
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        pellet_count: int  # number of pellets being delivered by this patch during this session
        wheel_distance_travelled: float  # wheel travel distance during this session for this patch
        """

    # Work on finished Session with SubjectPosition fully populated only
    key_source = (experiment.Session
                  & (experiment.Session * experiment.SessionEnd * experiment.SessionEpoch
                     & tracking.SubjectPosition & 'epoch_end = session_end').proj())

    def make(self, key):
        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)
        session_start, session_end = (experiment.Session * experiment.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject weights
        weight_start = (experiment.SubjectWeight.WeightTime
                        & f'weight_time = "{session_start}"').fetch1('weight')
        weight_end = (experiment.SubjectWeight.WeightTime
                      & f'weight_time = "{session_end}"').fetch1('weight')

        # subject's position data in this session
        timestamps, position_x, position_y, speed, area = (
                tracking.SubjectPosition & key).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

        # stack and structure in pandas DataFrame
        position_df = pd.DataFrame(dict(x=np.hstack(position_x),
                                        y=np.hstack(position_y),
                                        speed=np.hstack(speed),
                                        area=np.hstack(area)),
                                   index=np.hstack(timestamps))
        valid_position = (position_df.area > 0) & (
                    position_df.area < 1000)  # filter for objects of the correct size
        position_df = position_df[valid_position]
        position_df.x = position_df.x * tracking.pixel_scale
        position_df.y = position_df.y * tracking.pixel_scale

        position_diff = np.sqrt(np.square(np.diff(position_df.x)) + np.square(np.diff(position_df.y)))
        total_distance_travelled = np.nancumsum(position_diff)[-1]

        # food patch data
        food_patch_keys = (
                experiment.Session * experiment.SessionEnd
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistics = []
        for food_patch_key in food_patch_keys:
            pellet_count = len(experiment.FoodPatchEvent * experiment.EventType
                               & food_patch_key
                               & 'event_type = "PelletDetected"'
                               & f'event_time BETWEEN "{session_start}" AND "{session_end}"')
            # wheel data
            food_patch_description = (experiment.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle).values

            food_patch_statistics.append({
                **key, **food_patch_key,
                'pellet_count': pellet_count,
                'wheel_distance_travelled': wheel_distance_travelled[-1]})

        total_pellet_count = np.sum([p['pellet_count'] for p in food_patch_statistics])
        total_wheel_distance_travelled = np.sum([p['wheel_distance_travelled'] for p in food_patch_statistics])

        self.insert1({**key,
                      'total_pellet_count': total_pellet_count,
                      'total_wheel_distance_travelled': total_wheel_distance_travelled,
                      'change_in_weight': weight_end - weight_start,
                      'total_distance_travelled': total_distance_travelled})
        self.FoodPatch.insert(food_patch_statistics)


# ---------- HELPER ------------------

def is_position_in_nest(position_df, nest_key):
    """
    Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest
    """
    nest_vertices = list(zip(*(lab.ArenaNest.Vertex & nest_key).fetch(
        'vertex_x', 'vertex_y')))
    nest_path = path.Path(nest_vertices)

    return nest_path.contains_points(position_df[['x', 'y']])
