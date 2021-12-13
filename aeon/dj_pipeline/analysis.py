import datajoint as dj
import pandas as pd
import numpy as np
from matplotlib import path


from aeon.preprocess import api as aeon_api
from aeon.util import utils as aeon_utils

from . import lab, acquisition, tracking
from . import get_schema_name


schema = dj.schema(get_schema_name('analysis'))


# -------------- Session-level analysis ---------------------


@schema
class SessionTimeDistribution(dj.Computed):
    definition = """
    -> acquisition.Session
    ---
    time_fraction_in_corridor: float  # fraction of time the animal spent in the corridor in this session
    in_corridor: longblob             # array of boolean for if the animal is in the corridor (same length as position data)
    time_fraction_in_arena: float     # fraction of time the animal spent in the arena in this session
    in_arena: longblob                # array of boolean for if the animal is in the arena (same length as position data)
    """

    class Nest(dj.Part):
        definition = """  # Time spent in nest
        -> master
        -> lab.ArenaNest
        ---
        time_fraction_in_nest: float  # fraction of time the animal spent in this nest in this session
        in_nest: longblob             # array of boolean for if the animal is in this nest (same length as position data)
        """

    class FoodPatch(dj.Part):
        definition = """ # Time spent in food patch
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        in_patch: longblob             # array of boolean for if the animal is in this patch (same length as position data)
        """

    # Work on finished Session with TimeSlice and SubjectPosition fully populated only
    key_source = (acquisition.Session
                  & (acquisition.Session * acquisition.SessionEnd * acquisition.TimeSlice
                     & 'time_slice_end = session_end').proj()
                  & (acquisition.Session.aggr(acquisition.TimeSlice, time_slice_count='count(time_slice_start)')
                     * acquisition.Session.aggr(tracking.SubjectPosition, tracking_count='count(time_slice_start)')
                     & 'time_slice_count = tracking_count'))

    def make(self, key):
        raw_data_dir = acquisition.Experiment.get_data_directory(key)
        session_start, session_end = (acquisition.Session * acquisition.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject's position data in the time_slices
        position = tracking.SubjectPosition.get_session_position(key)

        # filter for objects of the correct size
        valid_position = (position.area > 0) & (position.area < 1000)
        position[~valid_position] = np.nan

        # in corridor
        distance_from_center = tracking.compute_distance(
            position[['x', 'y']],
            (tracking.arena_center_x, tracking.arena_center_y))
        in_corridor = (distance_from_center < tracking.arena_outer_radius) & (distance_from_center > tracking.arena_inner_radius)

        in_arena = ~in_corridor

        # in nests - loop through all nests in this experiment
        in_nest_times = []
        for nest_key in (lab.ArenaNest & key).fetch('KEY'):
            in_nest = is_position_in_nest(position, nest_key)
            in_nest_times.append(
                {**key, **nest_key,
                 'time_fraction_in_nest': in_nest.mean(),
                 'in_nest': in_nest})
            in_arena = in_arena & ~in_nest

        # in food patches - loop through all in-use patches during this session
        food_patch_keys = (
                acquisition.Session * acquisition.SessionEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        in_food_patch_times = []
        for food_patch_key in food_patch_keys:
            # wheel data
            food_patch_description = (acquisition.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle)

            patch_position = (acquisition.ExperimentFoodPatch.Position & food_patch_key).fetch1(
                'food_patch_position_x', 'food_patch_position_y')

            in_patch = tracking.is_in_patch(position, patch_position,
                                            wheel_distance_travelled, patch_radius=0.2)

            in_food_patch_times.append({
                **key, **food_patch_key,
                'time_fraction_in_patch': in_patch.mean(),
                'in_patch': in_patch.values})

            in_arena = in_arena & ~in_patch

        self.insert1({**key,
                      'time_fraction_in_corridor': in_corridor.mean(),
                      'in_corridor': in_corridor.values,
                      'time_fraction_in_arena': in_arena.mean(),
                      'in_arena': in_arena.values})
        self.Nest.insert(in_nest_times)
        self.FoodPatch.insert(in_food_patch_times)


@schema
class SessionSummary(dj.Computed):
    definition = """
    -> acquisition.Session
    ---
    total_distance_travelled: float  # (m) total distance the animal travelled during this session
    total_pellet_count: int  # total pellet delivered (triggered) for all patches during this session
    total_wheel_distance_travelled: float  # total wheel travelled distance for all patches
    change_in_weight: float  # weight change before/after the session
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        pellet_count: int  # number of pellets being delivered (triggered) by this patch during this session
        wheel_distance_travelled: float  # wheel travelled distance during this session for this patch
        """

    # Work on finished Session with TimeSlice and SubjectPosition fully populated only
    key_source = (acquisition.Session
                  & (acquisition.Session * acquisition.SessionEnd * acquisition.TimeSlice
                     & 'time_slice_end = session_end').proj()
                  & (acquisition.Session.aggr(acquisition.TimeSlice, time_slice_count='count(time_slice_start)')
                     * acquisition.Session.aggr(tracking.SubjectPosition, tracking_count='count(time_slice_start)')
                     & 'time_slice_count = tracking_count'))

    def make(self, key):
        raw_data_dir = acquisition.Experiment.get_data_directory(key)
        session_start, session_end = (acquisition.Session * acquisition.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # subject weights
        weight_start = (acquisition.SubjectWeight.WeightTime
                        & f'weight_time = "{session_start}"').fetch1('weight')
        weight_end = (acquisition.SubjectWeight.WeightTime
                      & f'weight_time = "{session_end}"').fetch1('weight')

        # subject's position data in this session
        position = tracking.SubjectPosition.get_session_position(key)

        valid_position = (position.area > 0) & (position.area < 1000)  # filter for objects of the correct size
        position = position[valid_position]

        position_diff = np.sqrt(np.square(np.diff(position.x)) + np.square(np.diff(position.y)))
        total_distance_travelled = np.nancumsum(position_diff)[-1]

        # food patch data
        food_patch_keys = (
                acquisition.Session * acquisition.SessionEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistics = []
        for food_patch_key in food_patch_keys:
            pellet_events = (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{session_start}" AND "{session_end}"').fetch(
                'event_time')
            # wheel data
            food_patch_description = (acquisition.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle).values

            food_patch_statistics.append({
                **key, **food_patch_key,
                'pellet_count': len(pellet_events),
                'wheel_distance_travelled': wheel_distance_travelled[-1]})

        total_pellet_count = np.sum([p['pellet_count'] for p in food_patch_statistics])
        total_wheel_distance_travelled = np.sum([p['wheel_distance_travelled'] for p in food_patch_statistics])

        self.insert1({**key,
                      'total_pellet_count': total_pellet_count,
                      'total_wheel_distance_travelled': total_wheel_distance_travelled,
                      'change_in_weight': weight_end - weight_start,
                      'total_distance_travelled': total_distance_travelled})
        self.FoodPatch.insert(food_patch_statistics)


@schema
class SessionRewardRate(dj.Computed):
    definition = """
    -> acquisition.Session
    ---
    pellet_rate_timestamps: longblob  # timestamps of the pellet rate over time
    patch2_patch1_rate_diff: longblob  # rate differences between Patch 2 and Patch 1
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        pellet_rate: longblob  # computed rate of pellet delivery over time
        """

    key_source = SessionSummary()

    def make(self, key):
        session_start, session_end = (acquisition.Session * acquisition.SessionEnd & key).fetch1(
            'session_start', 'session_end')

        # food patch data
        food_patch_keys = (
                acquisition.Session * acquisition.SessionEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').proj(
            'food_patch_description').fetch(as_dict=True)

        pellet_rate_timestamps = None
        rates = {}
        food_patch_reward_rates = []
        for food_patch_key in food_patch_keys:
            no_pellets = False
            pellet_events = (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{session_start}" AND "{session_end}"').fetch(
                'event_time')

            if not pellet_events.size:
                pellet_events = np.array([session_start, session_end])
                no_pellets = True

            pellet_rate = aeon_utils.get_events_rates(
                events=pd.DataFrame({'event_time': pellet_events}).set_index('event_time'),
                window_len_sec=600,
                start=pd.Timestamp(session_start),
                end=pd.Timestamp(session_end),
                frequency='5s', smooth='120s',
                center=True)

            if no_pellets:
                pellet_rate = pd.Series(index=pellet_rate.index, data=np.full(len(pellet_rate), 0))

            if pellet_rate_timestamps is None:
                pellet_rate_timestamps = pellet_rate.index.to_pydatetime()

            rates[food_patch_key.pop('food_patch_description')] = pellet_rate.values

            food_patch_reward_rates.append({
                **key, **food_patch_key,
                'pellet_rate': pellet_rate.values})

        self.insert1({**key,
                      'pellet_rate_timestamps': pellet_rate_timestamps,
                      'patch2_patch1_rate_diff': rates['Patch2'] - rates['Patch1']})
        self.FoodPatch.insert(food_patch_reward_rates)


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
