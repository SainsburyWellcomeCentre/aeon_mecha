import datajoint as dj
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import experiment, tracking, pipeline_api
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

    class Annotation(dj.Part):
        definition = """
        -> master
        annotation_time: datetime(3)  # datetime of the annotation
        ---
        -> experiment.SubjectAnnotation.Annotation        
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

        session_end = subject_exit['enter_exit_time']

        duration = (session_end - session_start).total_seconds() / 3600

        # annotations
        annotations = (
                experiment.SubjectAnnotation.Annotation
                & f'annotation_time BETWEEN "{session_start}" AND "{session_end}"').proj(
            session_start=f'"{session_start}"')

        # insert
        self.insert1({**key,
                      'session_end': session_end,
                      'session_duration': duration})
        self.Annotation.insert(annotations)


@schema
class SessionStatistics(dj.Computed):
    definition = """
    -> Session
    ---
    time_fraction_in_nest: float  # fraction of time the animal spent in the nest in this session
    distance_travelled: float  # total distance the animal travelled during this session
    """

    class FoodPatchStatistics(dj.Part):
        definition = """
        -> master
        -> experiment.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        wheel_distance_travelled: float  # total wheel travel distance during this session
        """

    # Work on Session with "tracking.SubjectPosition" fully populated only
    key_source = Session & (Session * experiment.Epoch.Subject
                            * tracking.SubjectPosition & 'epoch_end > session_end').proj()

    def make(self, key):
        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)

        session_epochs = find_session_epochs(key)
        session_start, session_end = (Session & key).fetch1('session_start', 'session_end')

        # subject's position data in the epochs
        timestamps, position_x, position_y, speed, area = (
                    tracking.SubjectPosition & session_epochs).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

        # stack and structure in pandas DataFrame
        position = pd.DataFrame(dict(x=np.hstack(position_x),
                                     y=np.hstack(position_y),
                                     speed=np.hstack(speed),
                                     area=np.hstack(area)),
                                index=np.hstack(timestamps))
        position = position[session_start:session_end]

        position_diff = np.sqrt(np.square(np.diff(position.x)) + np.square(np.diff(position.y)))
        distance_travelled = np.nancumsum(position_diff)[-1]

        is_in_nest = pipeline_api.is_in_nest(key, position.x, position.y)
        time_fraction_in_nest = sum(is_in_nest) / len(is_in_nest)

        # food patch data
        food_patch_keys = (
                Session
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistic = []
        for food_patch_key in food_patch_keys:
            is_in_patch = pipeline_api.is_in_patch(food_patch_key, position.x, position.y)
            time_fraction_in_patch = sum(is_in_patch) / len(is_in_patch)

            # wheel data
            food_patch_description = (experiment.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(session_start),
                                               end=pd.Timestamp(session_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle)[-1]

            food_patch_statistic.append({**key, **food_patch_key,
                                         'time_fraction_in_patch': time_fraction_in_patch,
                                         'wheel_distance_travelled': wheel_distance_travelled})

        self.insert1({**key,
                      'time_fraction_in_nest': time_fraction_in_nest,
                      'distance_travelled': distance_travelled})
        self.FoodPatchStatistics.insert(food_patch_statistic)


# ---------- HELPER FUNCTIONS -----------

def find_session_timebins(session_key):
    """
    Given a "session_key", return a query for all timebins belonging to this session
    """
    first_timebin = (Session * experiment.TimeBin
                     & session_key & 'session_start BETWEEN time_bin_start AND time_bin_end').fetch1("time_bin_start")
    last_timebin = (Session * experiment.TimeBin
                    & session_key & 'session_end BETWEEN time_bin_start AND time_bin_end').fetch1("time_bin_end")
    return (Session * experiment.TimeBin & session_key
            & f'time_bin_start BETWEEN "{first_timebin}" AND "{last_timebin}"')


def find_session_epochs(session_key):
    """
    Given a "session_key", return a query for all epochs belonging to this session
    """
    first_epoch = (Session * experiment.Epoch.Subject
                   & session_key & 'session_start BETWEEN epoch_start AND epoch_end').fetch1("epoch_start")
    last_epoch = (Session * experiment.Epoch.Subject
                  & session_key & 'session_end BETWEEN epoch_start AND epoch_end').fetch1("epoch_end")
    return (Session * experiment.Epoch.Subject & session_key
            & f'epoch_start BETWEEN "{first_epoch}" AND "{last_epoch}"')
