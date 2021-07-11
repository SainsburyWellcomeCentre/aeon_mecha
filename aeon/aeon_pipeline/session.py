import datajoint as dj
import pandas as pd
import datetime
import numpy as np
from matplotlib import path

from aeon.preprocess import api as aeon_api

from . import lab, experiment, tracking
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

    class Epoch(dj.Part):
        definition = """  # Epochs belonging to this session
        -> master
        -> experiment.Epoch.Subject  
        """

    class Annotation(dj.Part):
        definition = """  # Annotation for this subject from this session
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

        # epochs
        first_epoch = (experiment.Epoch.Subject & key
                       & f'"{session_start}" BETWEEN epoch_start AND epoch_end').fetch1(
            "epoch_start")
        last_epoch = (experiment.Epoch.Subject & key
                      & f'"{session_end}" BETWEEN epoch_start AND epoch_end').fetch1(
            "epoch_end")
        epochs = (experiment.Epoch.Subject & key
                  & f'epoch_start BETWEEN "{first_epoch}" AND "{last_epoch}"').proj(
            session_start=f'"{session_start}"')

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
        self.Epoch.insert(epochs)


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
        in_patch_timestamps: longblob  # timestamps of the time the animal spent on this patch
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        total_wheel_distance_travelled: float  # total wheel travel distance during this session
        """

    # Work on Session with "tracking.SubjectPosition" fully populated only
    key_source = Session & (Session * experiment.Epoch.Subject
                            * tracking.SubjectPosition
                            & tracking.SubjectDistance
                            & 'epoch_end > session_end').proj()

    # animal's distance from the food-patch position to be considered "time spent in the food patch"
    distance_threshold = 80

    def make(self, key):
        raw_data_dir = experiment.Experiment.get_raw_data_directory(key)

        session_epochs = Session.Epoch & key
        session_start, session_end = (Session & key).fetch1('session_start', 'session_end')

        # subject's position data in the epochs
        timestamps, position_x, position_y, speed, area = (
                    tracking.SubjectPosition & session_epochs).fetch(
            'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

        # timestamps of position data within this session
        session_timestamps = np.hstack(timestamps)
        session_timestamps = session_timestamps[np.logical_and(
            session_timestamps >= session_start,
            session_timestamps <= session_end)]

        # stack and structure in pandas DataFrame
        position = pd.DataFrame(dict(x=np.hstack(position_x),
                                     y=np.hstack(position_y),
                                     speed=np.hstack(speed),
                                     area=np.hstack(area)),
                                index=np.hstack(timestamps))
        position = position[session_start:session_end]

        position_diff = np.sqrt(np.square(np.diff(position.x)) + np.square(np.diff(position.y)))
        distance_travelled = np.nancumsum(position_diff)[-1]

        is_in_nest = is_position_in_nest(key, position.x, position.y)
        time_fraction_in_nest = sum(is_in_nest) / len(is_in_nest)

        # food patch data
        food_patch_keys = (
                Session
                * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'session_start >= food_patch_install_time'
                & 'session_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistics = []
        for food_patch_key in food_patch_keys:
            distance = (tracking.SubjectDistance.FoodPatch
                        & session_epochs & food_patch_key).fetch('distance')
            distance = pd.DataFrame(dict(distance=np.hstack(distance)),
                                    index=np.hstack(timestamps))
            distance = distance[session_start:session_end]

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
                'in_patch_timestamps': session_timestamps[distance.is_in_patch],
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

    mtl_path = path.Path(nest_vertices)

    return mtl_path.contains_points(np.vstack([position_x, position_y]).T)
