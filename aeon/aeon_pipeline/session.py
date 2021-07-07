import datajoint as dj
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import experiment, tracking
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('session'))


@schema
class Session(dj.Computed):
    definition = """
    -> experiment.Experiment.Subject        # the subject in this Epoch
    session_start: datetime(3)
    ---
    session_end: datetime(3)
    session_duration: float  # (hour)
    """

    @property
    def key_source(self):
        return (dj.U('experiment_name', 'subject', 'session_start')
                & (experiment.SubjectEnterExit.Time & 'enter_exit_event = "enter"').proj(
                    session_start='enter_exit_time'))

    def make(self, key):
        session_start = key['session_start']
        subject_exit = (experiment.SubjectEnterExit.Time
                        & {'subject': key['subject']}
                        & f'enter_exit_time > "{session_start}"').fetch(
            as_dict=True, limit=1, order_by='enter_exit_time ASC')
        if len(subject_exit) < 1:
            # subject still in the arena
            return
        else:
            subject_exit = subject_exit[0]

        if subject_exit['enter_exit_event'] != 'exit':
            raise ValueError(f'Subject {key["subject"]} never exited after {session_start}')

        duration = (subject_exit['enter_exit_time'] - session_start).total_seconds() / 3600

        self.insert1({**key,
                      'session_end': subject_exit['enter_exit_time'],
                      'session_duration': duration})


@schema
class SessionStatistics(dj.Computed):
    definition = """
    -> Session
    ---
    distance_travelled: float  # total distance the animal travelled during this session
    """

    class FoodPatchStatistics(dj.Part):
        definition = """
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        time_spent_in_patch: float  # (hour) total time the animal spent on this patch
        wheel_distance_travelled: float  # total wheel travel distance during this session
        """

    # Work on Session with "tracking.SubjectPosition" fully populated only
    key_source = Session & (Session * experiment.Epoch.Subject
                            * tracking.SubjectPosition & 'epoch_end > session_end').proj()

    # animal's distance from the food-patch position to be considered "time spent in the food patch"
    distance_threshold = 150

    def make(self, key):
        first_epoch = (Session * experiment.Epoch.Subject
                       & key & 'session_start BETWEEN epoch_start AND epoch_end').fetch1("epoch_start")
        last_epoch = (Session * experiment.Epoch.Subject
                      & key & 'session_end BETWEEN epoch_start AND epoch_end').fetch1("epoch_end")
        session_epochs = (Session * experiment.Epoch.Subject & key
                          & f'epoch_start BETWEEN "{first_epoch}" AND "{last_epoch}"')

        session_start, session_end = (Session & key).fetch1('session_start', 'session_end')

        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)
        positiondata = aeon_api.positiondata(raw_data_dir.as_posix(),
                                             start=pd.Timestamp(session_start),
                                             end=pd.Timestamp(session_end))

        position_diff = np.sqrt(np.square(np.diff(positiondata.x)) + np.square(np.diff(positiondata.y)))
        distance_travelled = np.nancumsum(position_diff)[-1]

        food_patch_keys = (
                Session
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistic = []
        for food_patch_key in food_patch_keys:
            timestamps, distances = (tracking.SubjectPosition
                                     * tracking.SubjectDistance.FoodPatch
                                     & session_epochs & food_patch_key).fetch('timestamps', 'distance')
            timestamps = np.hstack(timestamps)
            distances = np.hstack(distances)

            is_in_patch = np.logical_and(~np.isnan(distances),
                                         distances <= self.distance_threshold)
            if sum(is_in_patch):
                time_spent_in_patch = np.diff(timestamps[is_in_patch])
                time_spent_in_patch = time_spent_in_patch[time_spent_in_patch < datetime.timedelta(seconds=1)].sum().total_seconds()
            else:
                time_spent_in_patch = 0

            # wheel data
            food_patch_description = (experiment.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle)[-1]

            food_patch_statistic.append({**key, **food_patch_key,
                                         'time_spent_in_patch': time_spent_in_patch / 3600, # convert to hours
                                         'wheel_distance_travelled': wheel_distance_travelled})

        self.insert1({**key, 'distance_travelled': distance_travelled})
        self.FoodPatchStatistics.insert(food_patch_statistic)
