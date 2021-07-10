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

    @classmethod
    def get_raw_data_directory(cls, experiment_key):
        repo_name, path = (cls.Directory
                           & experiment_key
                           & 'directory_type = "raw"').fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        return root / path


@schema
class ExperimentCamera(dj.Manual):
    definition = """
    # Camera placement and operation for a particular time period, at a certain location, for a given experiment 
    -> Experiment
    -> lab.Camera
    camera_install_time: datetime(3)   # time of the camera placed and started operation at this position
    ---
    camera_description: varchar(36)    
    sampling_rate: float  # (Hz) sampling rate
    """

    class Position(dj.Part):
        definition = """
        -> master
        ---
        camera_position_x: float    # (m) x-position, in the arena's coordinate frame
        camera_position_y: float    # (m) y-position, in the arena's coordinate frame
        camera_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        camera_rotation_x=null: float    # 
        camera_rotation_y=null: float    # 
        camera_rotation_z=null: float    # 
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
    ---
    food_patch_description: varchar(36)
    """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        food_patch_remove_time: datetime(3)  # time of the food_patch being removed from this position
        """

    class Position(dj.Part):
        definition = """
        -> master
        ---
        food_patch_position_x: float    # (m) x-position, in the arena's coordinate frame
        food_patch_position_y: float    # (m) y-position, in the arena's coordinate frame
        food_patch_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        """

    class TileVertex(dj.Part):
        definition = """
        -> master
        vertex: int
        ---
        vertex_x: float    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float  # (m) z-coordinate of the vertex, in the arena's coordinate frame
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
        assert Experiment & {'experiment_name': experiment_name}, f'Experiment {experiment_name} does not exist!'

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

        raw_data_dir = Experiment.get_raw_data_directory(key)
        sessiondata = aeon_api.sessiondata(raw_data_dir.as_posix(),
                                           start=pd.Timestamp(time_bin_start),
                                           end=pd.Timestamp(time_bin_end))

        self.insert1(key)
        self.Time.insert(({**key, 'subject': r.id,
                           'enter_exit_event': self._enter_exit_event_mapper[r.event],
                           'enter_exit_time': r.name} for _, r in sessiondata.iterrows()
                         if r.id in subject_list), skip_duplicates=True)


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

        raw_data_dir = Experiment.get_raw_data_directory(key)
        annotations = aeon_api.annotations(raw_data_dir.as_posix(),
                                           start=pd.Timestamp(time_bin_start),
                                           end=pd.Timestamp(time_bin_end))

        self.insert1(key)
        self.Annotation.insert({**key, 'subject': r.id,
                                'annotation': r.annotation,
                                'annotation_time': r.name} for _, r in annotations.iterrows()
                               if r.id in subject_list)


# ------------------- SUBJECT EPOCH --------------------


@schema
class Epoch(dj.Imported):
    definition = """
    -> SubjectEnterExit
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
        time_bin_start, time_bin_end = (TimeBin & key).fetch1(
            'time_bin_start', 'time_bin_end')

        subject_epoch_list = []
        for subject_key in (Experiment.Subject & key).fetch('KEY'):
            subject_enter_exit_events = (SubjectEnterExit.Time
                                         & subject_key
                                         & f'enter_exit_time BETWEEN "{time_bin_start}" and "{time_bin_end}"')

            if not subject_enter_exit_events:
                continue

            # Loop through each epoch - insert the epoch if at least one condition is met:
            # 1. if there's an entering or exiting event for the animal
            # 2. if no event, insert if the most recent 'enter' event before this epoch

            epoch_start = time_bin_start
            while epoch_start < time_bin_end:
                epoch_end = epoch_start + self._epoch_duration

                has_enter_exit_event = (SubjectEnterExit.Time
                                        & subject_key
                                        & f'enter_exit_time BETWEEN "{epoch_start}" and "{epoch_end}"')

                if not has_enter_exit_event:
                    # If there is no entering/exiting event in this epoch
                    # then if the most recent entering/exiting event is "enter",
                    # this means the animal is still in the arena
                    recent_event = (SubjectEnterExit.Time
                                    & {'subject': subject_key['subject']}
                                    & f'enter_exit_time < "{epoch_start}"').fetch(
                        'enter_exit_event', order_by='enter_exit_time DESC', limit=1)
                    if not len(recent_event) or recent_event[0] != 'enter':
                        # If most recent event is not "enter", the animal is not in the arena,
                        # skip this epoch
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
                (32, 'PelletDetected'),
                (1000, 'No Events')]


@schema
class FoodPatchEvent(dj.Imported):
    definition = """  # events associated with a given ExperimentFoodPatch
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
        +  TimeBin(s) that started after FoodPatch install time and ended before FoodPatch remove time
        +  TimeBin(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        """
        return (TimeBin
                * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
                & 'time_bin_start >= food_patch_install_time'
                & 'time_bin_start < IFNULL(food_patch_remove_time, "2200-01-01")')

    def make(self, key):
        time_bin_start, time_bin_end = (TimeBin & key).fetch1('time_bin_start', 'time_bin_end')
        food_patch_description = (ExperimentFoodPatch & key).fetch1('food_patch_description')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        pelletdata = aeon_api.pelletdata(raw_data_dir.as_posix(),
                                         device=food_patch_description,
                                         start=pd.Timestamp(time_bin_start),
                                         end=pd.Timestamp(time_bin_end))

        if not len(pelletdata):
            event_list = [{**key, 'event_number': 0,
                           'event_time': time_bin_start, 'event_code': 1000}]
        else:
            event_code_mapper = {name: code for code, name
                                 in zip(*EventType.fetch('event_code', 'event_type'))}
            event_list = [{**key, 'event_number': r_idx, 'event_time': r_time,
                           'event_code': event_code_mapper[r.event]}
                          for r_idx, (r_time, r) in enumerate(pelletdata.iterrows())]

        self.insert(event_list)


@schema
class FoodPatchWheel(dj.Imported):
    definition = """  # wheel data associated with a given ExperimentFoodPatch
    -> TimeBin
    -> ExperimentFoodPatch
    ---
    timestamps:        longblob  # (datetime) timestamps of wheel encoder data
    angle:             longblob   # measured angles of the wheel 
    intensity:         longblob
    """

    @property
    def key_source(self):
        """
        Only the combination of TimeBin and ExperimentFoodPatch with overlapping time
        +  TimeBin(s) that started after FoodPatch install time and ended before FoodPatch remove time
        +  TimeBin(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        """
        return (TimeBin
                * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
                & 'time_bin_start >= food_patch_install_time'
                & 'time_bin_start < IFNULL(food_patch_remove_time, "2200-01-01")')

    def make(self, key):
        time_bin_start, time_bin_end = (TimeBin & key).fetch1('time_bin_start', 'time_bin_end')
        food_patch_description = (ExperimentFoodPatch & key).fetch1('food_patch_description')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                           device=food_patch_description,
                                           start=pd.Timestamp(time_bin_start),
                                           end=pd.Timestamp(time_bin_end))

        timestamps = (encoderdata.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        self.insert1({**key, 'timestamps': timestamps,
                      'angle': encoderdata.angle.values,
                      'intensity': encoderdata.intensity.values})
