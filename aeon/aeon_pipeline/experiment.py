import datajoint as dj
import datetime
import pathlib
import numpy as np
import pandas as pd

from aeon.preprocess import exp0_api

from . import lab, subject
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('experiment'))


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

    contents = [('ceph_aeon_test2', '/ceph/aeon/test2')]


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

    class Directory(dj.Part):
        definition = """
        -> master
        ---
        -> DataRepository
        directory_path: varchar(255)
        """


@schema
class ExperimentCamera(dj.Manual):
    definition = """
    # Camera placement and operation for a particular time period, at a certain location, for a given experiment 
    -> Experiment
    -> lab.Camera
    camera_installed_time: datetime(3)   # time of the camera placed and started operation at this position
    ---
    sampling_rate: float  # (Hz) sampling rate
    """

    class Position(dj.Part):
        definition = """
        camera_position_x: float    # (m) x-position, in the arena's coordinate frame
        camera_position_y: float    # (m) y-position, in the arena's coordinate frame
        camera_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        camera_rotation_x: float    # 
        camera_rotation_y: float    # 
        camera_rotation_z: float    # 
        """

    # class OpticalConfiguration(dj.Part):
    #     definition = """
    #     -> master
    #     """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        camera_removed_time: datetime(3)  # time of the camera being removed from this position
        """


@schema
class ExperimentFoodPatch(dj.Manual):
    definition = """  
    # Food patch placement and operation for a particular time period, at a certain location, for a given experiment 
    -> Experiment
    -> lab.FoodPatch
    food_patch_installed_time: datetime(3)   # time of the food_patch placed and started operation at this position
    """

    class Position(dj.Part):
        definition = """
        food_patch_position_x: float    # (m) x-position, in the arena's coordinate frame
        food_patch_position_y: float    # (m) y-position, in the arena's coordinate frame
        food_patch_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        food_patch_removed_time: datetime(3)  # time of the food_patch being removed from this position
        """


# ------------------- TIME BIN --------------------


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

    _bin_duration = datetime.timedelta(hours=3)

    @classmethod
    def generate_timebins(cls, experiment_name):
        repo, path = (Experiment.Directory * DataRepository
                      & {'experiment_name': experiment_name}).fetch1(
            'repository_path', 'directory_path')
        root = pathlib.Path(repo) / path
        sessiondata_files = sorted(list(root.rglob('SessionData*.csv')))

        time_bin_list, file_list = [], []
        for sessiondata_file in sessiondata_files:
            time_bin_start = datetime.datetime.strptime(
                sessiondata_file.stem.replace('SessionData_', ''), '%Y-%m-%dT%H-%M-%S')
            time_bin_end = time_bin_start + cls._bin_duration

            # --- insert to TimeBin ---
            time_bin_key = {'experiment_name': experiment_name,
                            'time_bin_start': time_bin_start}

            if time_bin_key in cls.proj():
                continue

            time_bin_list.append({**time_bin_key,
                                  'time_bin_end': time_bin_end})

            # -- files --
            file_datetime_str = sessiondata_file.stem.replace('SessionData_', '')
            files = list(pathlib.Path(sessiondata_file.parent).glob(f'*{file_datetime_str}*'))

            repositories = {p: n for n, p in zip(*DataRepository.fetch(
                'repository_name', 'repository_path'))}

            data_root_dir = paths.find_root_directory(list(repositories.keys()), files[0])
            repository_name = repositories[data_root_dir.as_posix()]
            file_list.extend(
                {**time_bin_key,
                 'file_number': f_idx,
                 'file_name': f.name,
                 'data_category': DataCategory.category_mapper[
                     f.name.split('_')[0]],
                 'repository_name': repository_name,
                 'file_path': f.relative_to(data_root_dir).as_posix()}
                for f_idx, f in enumerate(files))

        # insert
        print(f'Insert {len(time_bin_list)} new TimeBin')

        with cls.connection.transaction:
            cls.insert(time_bin_list)
            cls.File.insert(file_list)


@schema
class SubjectCrossingEvent(dj.Imported):
    definition = """  # Records of subjects entering/exiting the arena
    -> Experiment.Subject
    passage_event: enum('enter', 'exit')
    passage_time: datetime(3)  # datetime of subject entering/exiting the arena
    ---
    -> TimeBin  # the TimeBin where this entering/exiting event occur
    """

    _crossing_event_mapper = {'Start': 'enter', 'End': 'exit'}

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
                     'passage_event': self._crossing_event_mapper[r.event],
                     'passage_time': r.name} for _, r in sessiondata.iterrows())


# ------------------- SUBJECT EPOCH --------------------


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
            SubjectCrossingEvent.proj(tbin_start='time_bin_start'),
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
                recent_event = (SubjectCrossingEvent
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


# ------------------- EVENTS --------------------


@schema
class EventType(dj.Lookup):
    definition = """
    event_code: smallint
    ---
    event_type: varchar(24)
    """

    contents = [(35, 'TriggerPellet'),
                (32, 'PelletDetected')]


@schema
class FoodPatchEvent(dj.Imported):
    definition = """  # events associated with a given animal in a given SubjectEpoch
    -> TimeBin
    -> ExperimentFoodPatch
    event_number: smallint
    ---
    event_time: datetime(3)  # event time
    -> EventType
    """

    @property
    def key_source(self):
        removed_foodpatch = (
                TimeBin * ExperimentFoodPatch
                * ExperimentFoodPatch.RemovalTime
                & 'time_bin_start >= food_patch_installed_time'
                & 'time_bin_start < food_patch_removed_time')
        current_foodpatch = (
                TimeBin * (ExperimentFoodPatch - ExperimentFoodPatch.RemovalTime)
                & 'time_bin_start >= food_patch_installed_time')
        return removed_foodpatch.proj() + current_foodpatch.proj()

    def make(self, key):
        start, end = (TimeBin & key).fetch1('time_bin_start', 'time_bin_end')
        device_sn = (lab.FoodPatch * ExperimentFoodPatch & key).fetch1('food_patch_serial_number')

        file_repo, file_path = (TimeBin.File * DataRepository
                                & 'data_category = "PatchEvents"'
                                & f'file_name LIKE "%{device_sn}%"'
                                & key).fetch('repository_path', 'file_path', limit=1)
        data_dir = (pathlib.Path(file_repo[0]) / file_path[0]).parent

        pelletdata = exp0_api.pelletdata(data_dir.parent.as_posix(),
                                         device=device_sn,
                                         start=pd.Timestamp(start), end=pd.Timestamp(end))

        event_code_mapper = {name: code for code, name
                             in zip(*EventType.fetch('event_code', 'event_type'))}

        event_list = []
        for r_idx, (r_time, r) in enumerate(pelletdata.iterrows()):
            event_list.append({**key, 'event_number': r_idx,
                               'event_time': r_time,
                               'event_code': event_code_mapper[r.event]})

        self.insert(event_list)


@schema
class SubjectEvent(dj.Imported):
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
        -> ExperimentFoodPatch
        """
