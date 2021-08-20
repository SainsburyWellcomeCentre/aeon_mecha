import datajoint as dj
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import acquisition
from . import get_schema_name


schema = dj.schema(get_schema_name('tracking'))

pixel_scale = 0.00192  # 1 px = 1.92 mm
arena_center_x, arena_center_y = 1.475, 1.075  # center
arena_inner_radius = 0.93  # inner
arena_outer_radius = 0.97  # outer


@schema
class SubjectPosition(dj.Imported):
    definition = """
    -> acquisition.TimeSlice
    ---
    timestamps:        longblob  # (datetime) timestamps of the position data
    position_x:        longblob  # (px) animal's x-position, in the arena's coordinate frame
    position_y:        longblob  # (px) animal's y-position, in the arena's coordinate frame
    position_z=null:   longblob  # (px) animal's z-position, in the arena's coordinate frame
    area=null:         longblob  # (px^2) animal's size detected in the camera
    speed=null:        longblob  # (px/s) speed
    """

    def make(self, key):
        """
        The ingest logic here relies on the assumption that there is only one subject in the arena at a time
        The positiondata is associated with that one subject currently in the arena at any timepoints
        However, we need to take into account if the subject is entered or exited during this time slice
        """
        time_slice_start, time_slice_end = (acquisition.TimeSlice & key).fetch1('time_slice_start', 'time_slice_end')

        raw_data_dir = acquisition.Experiment.get_raw_data_directory(key)
        positiondata = aeon_api.positiondata(raw_data_dir.as_posix(),
                                             start=pd.Timestamp(time_slice_start),
                                             end=pd.Timestamp(time_slice_end))

        if not len(positiondata):
            raise ValueError(f'No position data between {time_slice_start} and {time_slice_end}')

        timestamps = (positiondata.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        x = positiondata.x.values
        y = positiondata.y.values
        z = np.full_like(x, 0.0)
        area = positiondata.area.values

        # speed - TODO: confirm with aeon team if this calculation is sufficient (any smoothing needed?)
        position_diff = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)) + np.square(np.diff(z)))
        time_diff = [t.total_seconds() for t in np.diff(timestamps)]
        speed = position_diff / time_diff
        speed = np.hstack((speed[0], speed))

        self.insert1({**key,
                      'timestamps': timestamps,
                      'position_x': x,
                      'position_y': y,
                      'position_z': z,
                      'area': area,
                      'speed': speed})

    @classmethod
    def get_session_position(cls, session_key):
        """
        Given a key to a single session, return a Pandas DataFrame for the position data
        of the subject for the specified session
        """
        assert len(acquisition.Session & session_key) == 1
        # subject's position data in the time slice
        timestamps, position_x, position_y, speed, area = (cls & session_key).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='time_slice_start')

        # stack and structure in pandas DataFrame
        position = pd.DataFrame(dict(x=np.hstack(position_x),
                                     y=np.hstack(position_y),
                                     speed=np.hstack(speed),
                                     area=np.hstack(area)),
                                index=np.hstack(timestamps))
        position.x = position.x * pixel_scale
        position.y = position.y * pixel_scale
        position.speed = position.speed * pixel_scale

        return position


@schema
class SubjectDistance(dj.Computed):
    definition = """
    -> SubjectPosition
    """

    class FoodPatch(dj.Part):
        definition = """  # distances of the animal away from the food patch, for each timestamp
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        distance: longblob
        """

    def make(self, key):
        food_patch_keys = (
                SubjectPosition * acquisition.TimeSlice
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'time_slice_start >= food_patch_install_time'
                & 'time_slice_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_distance_list = []
        for food_patch_key in food_patch_keys:
            patch_position = (acquisition.ExperimentFoodPatch.Position & food_patch_key).fetch1(
                'food_patch_position_x', 'food_patch_position_y', 'food_patch_position_z')
            subject_positions = (SubjectPosition & key).fetch1(
                'position_x', 'position_y', 'position_z')
            subject_positions = np.array([*zip(subject_positions)]).squeeze().T
            distances = np.linalg.norm(
                subject_positions
                - np.tile(patch_position, (subject_positions.shape[0], 1)), axis=1)

            food_patch_distance_list.append({**food_patch_key, 'distance': distances})

        self.insert1(key)
        self.FoodPatch.insert(food_patch_distance_list)


# ---------- HELPER ------------------

def compute_distance(position_df, target):
    assert len(target) == 2
    return np.sqrt(np.square(position_df[['x', 'y']] - target).sum(axis=1))


def is_in_patch(position_df, patch_position, wheel_distance_travelled, patch_radius=0.2):
    distance_from_patch = compute_distance(position_df, patch_position)
    in_patch = distance_from_patch < patch_radius
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel_distance_travelled.diff().rolling('1s').sum() > 1).reindex(
        position_df.index, method='pad')
    time_slice = exit_patch.cumsum()
    return in_wheel.groupby(time_slice).apply(lambda x:x.cumsum()) > 0
