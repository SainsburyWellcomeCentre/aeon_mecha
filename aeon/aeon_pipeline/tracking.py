import datajoint as dj
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import experiment
from . import get_schema_name


schema = dj.schema(get_schema_name('tracking'))


@schema
class SubjectPosition(dj.Imported):
    definition = """
    -> experiment.SessionEpoch
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
        The ingestion logic here relies on the assumption that there is only one subject in the arena at a time
        The positiondata is associated with that one subject currently in the arena at any timepoints
        However, we need to take into account if the subject is entered or exited during this epoch
        """
        epoch_start, epoch_end = (experiment.SessionEpoch & key).fetch1('epoch_start', 'epoch_end')

        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)
        positiondata = aeon_api.positiondata(raw_data_dir.as_posix(),
                                             start=pd.Timestamp(epoch_start),
                                             end=pd.Timestamp(epoch_end))

        if not len(positiondata):
            raise ValueError(f'No position data between {epoch_start} and {epoch_end}')

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


@schema
class SubjectDistance(dj.Computed):
    definition = """
    -> SubjectPosition
    """

    class FoodPatch(dj.Part):
        definition = """  # distances of the animal away from the food patch, for each timestamp
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        distance: longblob
        """

    def make(self, key):
        food_patch_keys = (
                SubjectPosition * experiment.SessionEpoch
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'epoch_start >= food_patch_install_time'
                & 'epoch_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_distance_list = []
        for food_patch_key in food_patch_keys:
            patch_position = (experiment.ExperimentFoodPatch.Position & food_patch_key).fetch1(
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
