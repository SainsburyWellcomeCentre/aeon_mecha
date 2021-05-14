import datajoint as dj
import datetime
import pathlib
import numpy as np

from aeon.preprocess import exp0_api

from . import lab, subject
from . import get_schema_name


schema = dj.schema(get_schema_name('experiment'))


# ------------------- GENERAL INFORMATION ABOUT AN EXPERIMENT --------------------


@schema
class Experiment(dj.Manual):
    definition = """
    experiment_name: char(8)  # e.g exp0-a
    ---
    experiment_start_time: datetime(3)  # datetime of the start of this experiment
    experiment_description: varchar(1000)
    -> lab.Arena
    -> lab.Location  # lab/room where a particular experiment takes place
    """

    class Subject(dj.Part):
        definition = """  # the subjects participating in this experiment
        -> master
        -> subject.Subject
        """


@schema
class Camera(dj.Manual):
    definition = """
    -> Experiment
    camera:            varchar(24)    # device type/function
    ---
    sampling_rate:          decimal(8, 4)  # sampling rate (Hz)
    camera_position_x:      float          # (m) x-position, in the arena's coordinate frame
    camera_position_y:      float          # (m) y-position, in the arena's coordinate frame
    camera_position_z=0:    float          # (m) z-position, in the arena's coordinate frame
    camera_description='':  varchar(100)   # device description
    """


@schema
class FoodPatch(dj.Manual):
    definition = """
    -> Experiment
    food_patch:            varchar(24)    # device type/function
    ---
    food_patch_position_x:      float          # (m) x-position, in the arena's coordinate frame
    food_patch_position_y:      float          # (m) y-position, in the arena's coordinate frame
    food_patch_position_z=0:    float          # (m) z-position, in the arena's coordinate frame
    food_patch_description='':  varchar(100)   # device description
    """


# ------------------- DATASET ------------------------

@schema
class DataCategory(dj.Lookup):
    definition = """
    data_category: varchar(24)  
    ---
    category_description: varchar(500)  # Short description of dataset type
    """
    contents = [
        ['SessionMeta', 'Meta information of session'],
        ['VideoCamera', 'Data from camera'],
        ['VideoEvents', 'Events from video camera'],
        ['PatchEvents', 'Events from food patch'],
        ['Wheel', 'Events from wheel device'],
        ['Audio', 'Audio data']
    ]

    category_mapper = {'SessionData': 'SessionMeta',
                       'PatchEvents': 'PatchEvents',
                       'VideoEvents': 'VideoEvents',
                       'FrameSide': 'VideoCamera',
                       'FrameTop': 'VideoCamera',
                       'WheelThreshold': 'Wheel',
                       'AudioAmbient': 'Audio'}


@schema
class DataRepository(dj.Lookup):
    definition = """
    repository_name: varchar(16)
    ---
    repository_path: varchar(255)  # path to the data directory of this repository (posix path)
    """

    contents = [('ceph_aeon_test2', '/ceph/aeon/test2/data')]


@schema
class TimeBin(dj.Manual):
    definition = """  # A recording period corresponds to an N-hour data acquisition
    -> Experiment
    time_bin_start: datetime(3)  # datetime of the start of this recorded TimeBin
    ---
    time_bin_end: datetime(3)    # datetime of the end of this recorded TimeBin
    """

    class File(dj.Part):
        definition = """
        -> master
        file_number: tinyint
        ---
        file_name: varchar(128)
        -> DataCategory
        -> DataRepository
        file_path: varchar(255)  # path of the file, relative to the data repository
        """


@schema
class SubjectPassageEvent(dj.Imported):
    definition = """
    -> Experiment.Subject
    passage_event: enum('enter', 'exit')
    passage_time: datetime(3)  # datetime of subject entering/exiting the arena
    ---
    -> TimeBin  # the TimeBin where this entering/exiting event occur
    """

    _passage_event_mapper = {'Start': 'enter', 'End': 'exit'}

    @property
    def key_source(self):
        return TimeBin - self

    def make(self, key):
        file_repo, file_path = (TimeBin.File * DataRepository
                                & 'data_category = "SessionMeta"' & key).fetch1(
            'repository_path', 'file_path')
        sessiondata_file = pathlib.Path(file_repo) / file_path
        sessiondata = exp0_api.sessionreader(sessiondata_file.as_posix())

        self.insert({**key, 'subject': r.id,
                     'passage_event': self._passage_event_mapper[r.event],
                     'passage_time': r.name} for _, r in sessiondata.iterrows())

# ------------------- SUBJECT PERIOD --------------------


@schema
class SubjectEpoch(dj.Imported):
    definition = """
    # A short time-chunk (e.g. 30 seconds) of the recording of a given animal in the arena
    -> Experiment.Subject        # the subject in this Epoch
    epoch_start: datetime(3)  # datetime of the start of this Epoch
    ---
    epoch_end: datetime(3)    # datetime of the end of this Epoch
    -> TimeBin                # the TimeBin containing this Epoch
    """

    _epoch_duration = datetime.timedelta(hours=0, minutes=30)

    @property
    def key_source(self):
        """
        A candidate TimeBin is to be processed only when
          SubjectPassageEvent.populate() is completed
          for all TimeBin occurred prior to this candidate TimeBin
        """
        prior_timebin = TimeBin.proj().aggr(
            TimeBin.proj(tbin_start='time_bin_start'),
            prior_timebin_count=('count(time_bin_start>tbin_start)'))
        prior_passage_event = TimeBin.proj().aggr(
            SubjectPassageEvent.proj(tbin_start='time_bin_start'),
            prior_passage_event_count=('count(time_bin_start>tbin_start)'))
        key_source = (Experiment.Subject
                      * (TimeBin & (prior_timebin * prior_passage_event
                                    & 'prior_passage_event_count >= prior_timebin_count')))

        return key_source.proj() - self

    def make(self, key):
        file_repo, file_path = (TimeBin.File * DataRepository
                                & 'data_category = "SessionMeta"' & key).fetch1(
            'repository_path', 'file_path')
        sessiondata_file = pathlib.Path(file_repo) / file_path
        sessiondata = exp0_api.sessionreader(sessiondata_file.as_posix())
        subject_sessiondata = sessiondata[sessiondata.id == key['subject']]

        time_bin_start, time_bin_end = (TimeBin & key).fetch1(
            'time_bin_start', 'time_bin_end')

        # Loop through each epoch - insert the epoch if at least one condition is met:
        # 1. if there's an entering or exiting event for the animal
        # 2. if no event, insert if the most recent passage event before this epoch
        #    (from SubjectPassageEvent) is `enter`

        subject_epoch_list = []
        epoch_start = time_bin_start
        while epoch_start < time_bin_end:
            epoch_end = epoch_start + self._epoch_duration

            has_passage_event = np.any(
                np.logical_and(subject_sessiondata.index >= epoch_start,
                               subject_sessiondata.index < epoch_end))

            if not has_passage_event:  # no entering/exiting event in this epoch
                recent_event = (SubjectPassageEvent
                                & {'subject': key['subject']}
                                & f'passage_time < "{epoch_start}"').fetch(
                    'passage_event', order_by='passage_time DESC', limit=1)
                if not len(recent_event) or recent_event[0] != 'enter':  # most recent event is not "enter"
                    epoch_start = epoch_end
                    continue

            subject_epoch_list.append({**key, 'epoch_start': epoch_start,
                                       'epoch_end': epoch_end})
            epoch_start = epoch_end

        self.insert(subject_epoch_list)


@schema
class EventType(dj.Lookup):
    definition = """
    event_code: smallint
    ---
    event_type: varchar(24)
    """

    contents = [(0, 'food-drop')]


@schema
class Event(dj.Imported):
    definition = """  # events associated with a given animal in a given SubjectEpoch
    -> SubjectEpoch
    event_number: smallint
    ---
    -> EventType
    event_time: decimal(8, 2)  # (s) event time w.r.t to the start of this TimeBin
    """

    class FoodPatch(dj.Part):
        definition = """  # The food patch associated with a food-drop event
        -> master
        ---
        -> FoodPatch
        """
