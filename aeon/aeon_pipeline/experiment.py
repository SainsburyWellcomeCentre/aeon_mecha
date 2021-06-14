import datajoint as dj
import datetime
import pathlib
import numpy as np
import pandas as pd

from aeon.preprocess import api as aeon_api

from . import lab, subject
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('experiment'))


# ------------------- DATASET ------------------------

@schema
class PipelineRepository(dj.Lookup):
    definition = """
    repository_name: varchar(16)
    """

    contents = zip(['ceph_aeon'])


# ------------------- GENERAL INFORMATION ABOUT AN EXPERIMENT --------------------


@schema
class Experiment(dj.Manual):
    definition = """
    experiment_name: char(12)  # e.g exp0-a
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
        directory_type: enum('raw', 'preprocessing', 'analysis')
        ---
        -> PipelineRepository
        directory_path: varchar(255)
        """


@schema
class ExperimentCamera(dj.Manual):
    definition = """
    # Camera placement and operation for a particular time period, at a certain location, for a given experiment 
    -> Experiment
    -> lab.Camera
    camera_install_time: datetime(3)   # time of the camera placed and started operation at this position
    ---
    sampling_rate: float  # (Hz) sampling rate
    """

    class Position(dj.Part):
        definition = """
        -> master
        ---
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
    #     ---
    #     calibration_factor: float  # px to mm
    #     """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        camera_remove_time: datetime(3)  # time of the camera being removed from this position
        """


@schema
class ExperimentFoodPatch(dj.Manual):
    definition = """  
    # Food patch placement and operation for a particular time period, at a certain location, for a given experiment 
    -> Experiment
    -> lab.FoodPatch
    food_patch_install_time: datetime(3)   # time of the food_patch placed and started operation at this position
    """

    class Position(dj.Part):
        definition = """
        -> master
        ---
        food_patch_position_x: float    # (m) x-position, in the arena's coordinate frame
        food_patch_position_y: float    # (m) y-position, in the arena's coordinate frame
        food_patch_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        food_patch_remove_time: datetime(3)  # time of the food_patch being removed from this position
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
        file_number: int
        ---
        file_name: varchar(128)
        -> PipelineRepository
        file_path: varchar(255)  # path of the file, relative to the data repository
        """

    @classmethod
    def generate_timebins(cls, experiment_name):
        repo_name, path = (Experiment.Directory
                           & {'experiment_name': experiment_name}
                           & 'directory_type = "raw"').fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        raw_data_dir = root / path

        time_bin_rep_file_str = 'FrameTop_'
        time_bin_rep_files = sorted(list(raw_data_dir.rglob(f'{time_bin_rep_file_str}*.csv')))

        time_bin_list, file_list, file_name_list = [], [], []
        for time_bin_rep_file in time_bin_rep_files:
            time_bin_start = datetime.datetime.strptime(
                time_bin_rep_file.stem.replace(time_bin_rep_file_str, ''), '%Y-%m-%dT%H-%M-%S')
            time_bin_end = time_bin_start + datetime.timedelta(hours=aeon_api.BIN_SIZE)

            # --- insert to TimeBin ---
            time_bin_key = {'experiment_name': experiment_name,
                            'time_bin_start': time_bin_start}

            if time_bin_key in cls.proj() or time_bin_rep_file.name in file_name_list:
                continue

            time_bin_list.append({**time_bin_key,
                                  'time_bin_end': time_bin_end})
            file_name_list.append(time_bin_rep_file.name)  # handle duplicated files in different folders - TODO: confirm why?

            # -- files --
            file_datetime_str = time_bin_rep_file.stem.replace(time_bin_rep_file_str, '')
            files = list(pathlib.Path(raw_data_dir).rglob(f'*{file_datetime_str}*'))

            file_list.extend(
                {**time_bin_key,
                 'file_number': f_idx,
                 'file_name': f.name,
                 'repository_name': repo_name,
                 'file_path': f.relative_to(root).as_posix()}
                for f_idx, f in enumerate(files))

        # insert
        print(f'Insert {len(time_bin_list)} new TimeBin')

        with cls.connection.transaction:
            cls.insert(time_bin_list)
            cls.File.insert(file_list)


@schema
class SubjectEnterExit(dj.Imported):
    definition = """  # Records of subjects entering/exiting the arena
    -> TimeBin  
    """

    _enter_exit_event_mapper = {'Start': 'enter', 'End': 'exit'}

    class Time(dj.Part):
        definition = """
        -> master
        -> Experiment.Subject
        enter_exit_time: datetime(3)  # datetime of subject entering/exiting the arena
        ---
        enter_exit_event: enum('enter', 'exit')       
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch('subject')
        time_bin_start, time_bin_end = (TimeBin & key).fetch1(
            'time_bin_start', 'time_bin_end')

        repo_name, path = (Experiment.Directory
                           & 'directory_type = "raw"'
                           & key).fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        raw_data_dir = root / path
        sessiondata = aeon_api.sessiondata(raw_data_dir.as_posix(),
                                           start=pd.Timestamp(time_bin_start),
                                           end=pd.Timestamp(time_bin_end))
        self.insert1(key)
        self.Time.insert({**key, 'subject': r.id,
                          'enter_exit_event': self._enter_exit_event_mapper[r.event],
                          'enter_exit_time': r.name} for _, r in sessiondata.iterrows()
                         if r.id in subject_list)


@schema
class SubjectAnnotation(dj.Imported):
    definition = """  # Experimenter's annotations 
    -> TimeBin  
    """

    class Annotation(dj.Part):
        definition = """
        -> master
        -> Experiment.Subject
        annotation_time: datetime(3)  # datetime of the annotation
        ---
        annotation: varchar(1000)   
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch('subject')
        time_bin_start, time_bin_end = (TimeBin & key).fetch1(
            'time_bin_start', 'time_bin_end')

        repo_name, path = (Experiment.Directory
                           & 'directory_type = "raw"'
                           & key).fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        raw_data_dir = root / path
        annotations = aeon_api.annotations(raw_data_dir.as_posix(),
                                           start=pd.Timestamp(time_bin_start),
                                           end=pd.Timestamp(time_bin_end))

        self.insert1(key)
        self.Time.insert({**key, 'subject': r.id,
                          'annotation': r.annotation,
                          'annotation_time': r.name} for _, r in annotations.iterrows()
                         if r.id in subject_list)


# ------------------- SUBJECT EPOCH --------------------


@schema
class Epoch(dj.Imported):
    definition = """
    -> TimeBin
    """

    class Subject(dj.Part):
        definition = """
        # A short time-chunk (e.g. 30 seconds) of the recording of a given animal in the arena
        -> master
        -> Experiment.Subject        # the subject in this Epoch
        epoch_start: datetime(3)  # datetime of the start of this Epoch
        ---
        epoch_end: datetime(3)    # datetime of the end of this Epoch
        """

    _epoch_duration = datetime.timedelta(hours=0, minutes=0, seconds=30)

    def make(self, key):
        repo_name, file_path = (TimeBin.File * PipelineRepository
                                & 'data_source = "SessionMeta"' & key).fetch1(
            'repository_name', 'file_path')
        sessiondata_file = paths.get_repository_path(repo_name) / file_path
        sessiondata = aeon_api.sessionreader(sessiondata_file.as_posix())
        time_bin_start, time_bin_end = (TimeBin & key).fetch1(
            'time_bin_start', 'time_bin_end')

        subject_epoch_list = []
        for subject_key in (Experiment.Subject & key).fetch('KEY'):
            subject_sessiondata = sessiondata[sessiondata.id == subject_key['subject']]

            if not len(subject_sessiondata):
                continue

            # Loop through each epoch - insert the epoch if at least one condition is met:
            # 1. if there's an entering or exiting event for the animal
            # 2. if no event, insert if the most recent 'enter' event before this epoch

            epoch_start = time_bin_start
            while epoch_start < time_bin_end:
                epoch_end = epoch_start + self._epoch_duration

                has_passage_event = np.any(
                    np.logical_and(subject_sessiondata.index >= epoch_start,
                                   subject_sessiondata.index < epoch_end))

                if not has_passage_event:  # no entering/exiting event in this epoch
                    recent_event = (SubjectEnterExit.Time
                                    & {'subject': subject_key['subject']}
                                    & f'enter_exit_time < "{epoch_start}"').fetch(
                        'enter_exit_event', order_by='enter_exit_time DESC', limit=1)
                    if not len(recent_event) or recent_event[0] != 'enter':  # most recent event is not "enter"
                        epoch_start = epoch_end
                        continue

                subject_epoch_list.append({**key, **subject_key,
                                           'epoch_start': epoch_start,
                                           'epoch_end': epoch_end})
                epoch_start = epoch_end

        self.insert1(key)
        self.Subject.insert(subject_epoch_list)


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
    definition = """  # events associated with a given animal in a given ExperimentFoodPatch
    -> TimeBin
    -> ExperimentFoodPatch
    event_number: smallint
    ---
    event_time: datetime(3)  # event time
    -> EventType
    """

    @property
    def key_source(self):
        """
        Only the combination of TimeBin and ExperimentFoodPatch with overlapping time
        """
        # TimeBin(s) that started after FoodPatch install time and ended before FoodPatch remove time
        removed_foodpatch = (
                TimeBin * ExperimentFoodPatch
                * ExperimentFoodPatch.RemovalTime
                & 'time_bin_start >= food_patch_install_time'
                & 'time_bin_start < food_patch_remove_time')
        # TimeBin(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        current_foodpatch = (
                TimeBin * (ExperimentFoodPatch - ExperimentFoodPatch.RemovalTime)
                & 'time_bin_start >= food_patch_install_time')
        return removed_foodpatch.proj() + current_foodpatch.proj()

    def make(self, key):
        time_bin_start, time_bin_end = (TimeBin & key).fetch1('time_bin_start', 'time_bin_end')
        device_sn = (lab.FoodPatch * ExperimentFoodPatch & key).fetch1('food_patch_serial_number')

        repo_name, file_path = (TimeBin.File * PipelineRepository
                                & 'data_source = "PatchEvents"'
                                & f'file_name LIKE "%{device_sn}%"'
                                & key).fetch('repository_name', 'file_path', limit=1)
        data_dir = (paths.get_repository_path(repo_name[0]) / file_path[0]).parent

        pelletdata = aeon_api.pelletdata(data_dir.parent.as_posix(),
                                         device=device_sn,
                                         start=pd.Timestamp(time_bin_start),
                                         end=pd.Timestamp(time_bin_end))

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
    -> Epoch.Subject
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
