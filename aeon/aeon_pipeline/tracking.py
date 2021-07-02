import datajoint as dj
import pathlib
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import experiment
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('tracking'))


@schema
class SubjectPosition(dj.Imported):
    definition = """
    -> experiment.Epoch.Subject
    ---
    timestamps:        longblob  # (datetime) timestamps of the position data
    position_x:        longblob  # (mm) animal's x-position, in the arena's coordinate frame
    position_y:        longblob  # (mm) animal's y-position, in the arena's coordinate frame
    position_z=null:   longblob  # (mm) animal's z-position, in the arena's coordinate frame
    area=null:         longblob  # (mm^2) animal's size detected in the camera
    speed=null:        longblob  # (mm/s) speed
    """

    class UniquePosition(dj.Part):
        definition = """  # All unique positions (x,y,z) of an animal in a given epoch
        -> master
        x: decimal(6, 2)
        y: decimal(6, 2)
        z: decimal(6, 2)
        """

    def make(self, key):
        """
        The ingestion logic here relies on the assumption that there is only one subject in the arena at a time
        The positiondata is associated with that one subject currently in the arena at any timepoints
        However, we need to take into account if the subject is entered or exited during this epoch
        """
        repo_name, path = (experiment.Experiment.Directory
                           & 'directory_type = "raw"'
                           & key).fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        raw_data_dir = root / path

        epoch_start, epoch_end = (experiment.Epoch.Subject & key).fetch1(
            'epoch_start', 'epoch_end')

        positiondata = aeon_api.positiondata(raw_data_dir.as_posix(),
                                             start=pd.Timestamp(epoch_start),
                                             end=pd.Timestamp(epoch_end))

        timestamps = (positiondata.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        # account for the animal entering/exiting during this epoch
        enter_exit_times = (experiment.SubjectEnterExit.Time & {'subject': key['subject']}
                            & f'enter_exit_time BETWEEN "{epoch_start}" AND "{epoch_end}"')
        if not enter_exit_times:
            # no enter/exit event - i.e. subject in the arena for the whole epoch
            is_in_arena = np.full(len(positiondata), True, dtype=bool)
        else:
            event_types, event_times = enter_exit_times.fetch(
                'enter_exit_event', 'enter_exit_time', order_by='enter_exit_time')
            current_status = event_types[0] == 'exit'
            is_in_arena = np.full(len(positiondata), current_status, dtype=bool)
            for event_type, event_time in zip(event_types, event_times):
                current_status = not current_status
                is_in_arena = np.where(positiondata.index >= event_time, current_status, is_in_arena)

        x = np.where(is_in_arena, positiondata.x.values, np.nan)
        y = np.where(is_in_arena, positiondata.y.values, np.nan)
        z = np.full_like(x, 0.0)
        area = np.where(is_in_arena, positiondata.area.values, np.nan)

        # unique positions
        unique_positions = set(list(zip(np.round(z, 2),
                                        np.round(y, 2),
                                        np.round(z, 2))))
        self.insert1({**key,
                      'timestamps': timestamps,
                      'position_x': x,
                      'position_y': y,
                      'position_z': z,
                      'area': area})
        self.UniquePosition.insert([{**key, 'x': x, 'y': y, 'z': z}
                                    for x, y, z in unique_positions
                                    if not np.any(np.where(np.isnan([x, y, z])))])


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

    class UniqueDistance(dj.Part):
        definition = """ # unique distances of the animal away from the food patch for this epoch
        -> master.FoodPatch
        -> SubjectPosition.UniquePosition
        ---
        distance: decimal(6, 2)
        """

    def make(self, key):
        food_patch_keys = (
                experiment.Epoch.Subject
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'epoch_start >= food_patch_install_time'
                & 'epoch_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_distance_list, unique_distance_list = [], []
        for food_patch_key in food_patch_keys:
            patch_position = (experiment.ExperimentFoodPatch.Position & food_patch_key).fetch1(
                'food_patch_position_x', 'food_patch_position_y', 'food_patch_position_z')
            subject_positions = (SubjectPosition & key).fetch1(
                'position_x', 'position_y', 'position_z')
            subject_positions = np.array([*zip(subject_positions)]).squeeze().T
            distances = np.linalg.norm(
                subject_positions
                - np.tile(patch_position, (subject_positions.shape[0], 1)), axis=1)

            unique_positions = (SubjectPosition.UniquePosition & key).fetch('x', 'y', 'z')
            unique_positions = np.array([*zip(unique_positions)]).squeeze(axis=1).T
            unique_distances = np.linalg.norm(
                unique_positions.astype(float)
                - np.tile(patch_position, (unique_positions.shape[0], 1)), axis=1)

            food_patch_distance_list.append({**food_patch_key, 'distance': distances})
            unique_distance_list.extend([{**food_patch_key,
                                          'x': x, 'y': y, 'z': z,
                                          'distance': d}
                                         for (x, y, z), d in zip(
                    unique_positions, unique_distances)])

        self.insert1(key)
        self.FoodPatch.insert(food_patch_distance_list)
        self.UniqueDistance.insert(unique_distance_list)
