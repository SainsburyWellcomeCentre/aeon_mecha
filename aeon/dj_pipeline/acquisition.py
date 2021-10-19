import datajoint as dj
import datetime
import pathlib
import numpy as np
import pandas as pd

from aeon.preprocess import api as aeon_api

from . import lab, subject
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('acquisition'))


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
    experiment_start_time: datetime(6)  # datetime of the start of this experiment
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
    camera_install_time: datetime(6)   # time of the camera placed and started operation at this position
    ---
    camera_description: varchar(36)
    camera_sampling_rate: float  # (Hz) camera frame rate
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
        camera_remove_time: datetime(6)  # time of the camera being removed from this position
        """


@schema
class ExperimentFoodPatch(dj.Manual):
    definition = """
    # Food patch placement and operation for a particular time period, at a certain location, for a given experiment
    -> Experiment
    -> lab.FoodPatch
    food_patch_install_time: datetime(6)   # time of the food_patch placed and started operation at this position
    ---
    food_patch_description: varchar(36)
    wheel_sampling_rate: float  # (Hz) wheel's sampling rate
    """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        food_patch_remove_time: datetime(6)  # time of the food_patch being removed from this position
        """

    class Position(dj.Part):
        definition = """
        -> master
        ---
        food_patch_position_x: float    # (m) x-position, in the arena's coordinate frame
        food_patch_position_y: float    # (m) y-position, in the arena's coordinate frame
        food_patch_position_z=0: float  # (m) z-position, in the arena's coordinate frame
        """


# ------------------- ACQUISITION CHUNK --------------------


@schema
class Chunk(dj.Manual):
    definition = """  # A recording period corresponds to a 1-hour data acquisition
    -> Experiment
    chunk_start: datetime(6)  # datetime of the start of a given acquisition chunk
    ---
    chunk_end: datetime(6)    # datetime of the end of a given acquisition chunk
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
    def generate_chunks(cls, experiment_name):
        assert Experiment & {'experiment_name': experiment_name}, f'Experiment {experiment_name} does not exist!'

        repo_name, path = (Experiment.Directory
                           & {'experiment_name': experiment_name}
                           & 'directory_type = "raw"').fetch1(
            'repository_name', 'directory_path')
        root = paths.get_repository_path(repo_name)
        raw_data_dir = root / path

        chunk_rep_file_str = 'FrameTop_'
        chunk_rep_files = sorted(list(raw_data_dir.rglob(f'{chunk_rep_file_str}*.csv')))

        chunk_list, file_list, file_name_list = [], [], []
        for chunk_rep_file in chunk_rep_files:
            chunk_start = datetime.datetime.strptime(
                chunk_rep_file.stem.replace(chunk_rep_file_str, ''), '%Y-%m-%dT%H-%M-%S')
            chunk_end = chunk_start + datetime.timedelta(hours=aeon_api.BIN_SIZE)

            # --- insert to Chunk ---
            chunk_key = {'experiment_name': experiment_name,
                            'chunk_start': chunk_start}

            if chunk_key in cls.proj() or chunk_rep_file.name in file_name_list:
                continue

            chunk_list.append({**chunk_key,
                                  'chunk_end': chunk_end})
            file_name_list.append(chunk_rep_file.name)  # handle duplicated files in different folders

            # -- files --
            file_datetime_str = chunk_rep_file.stem.replace(chunk_rep_file_str, '')
            files = list(pathlib.Path(raw_data_dir).rglob(f'*{file_datetime_str}*'))

            file_list.extend(
                {**chunk_key,
                 'file_number': f_idx,
                 'file_name': f.name,
                 'repository_name': repo_name,
                 'file_path': f.relative_to(root).as_posix()}
                for f_idx, f in enumerate(files))

        # insert
        print(f'Insert {len(chunk_list)} new Chunk')

        with cls.connection.transaction:
            cls.insert(chunk_list)
            cls.File.insert(file_list)


@schema
class SubjectEnterExit(dj.Imported):
    definition = """  # Records of subjects entering/exiting the arena
    -> Chunk
    """

    _enter_exit_event_mapper = {'Start': 'enter', 'End': 'exit'}

    class Time(dj.Part):
        definition = """  # Timestamps of each entering/exiting events
        -> master
        -> Experiment.Subject
        enter_exit_time: datetime(6)  # datetime of subject entering/exiting the arena
        ---
        enter_exit_event: enum('enter', 'exit')
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch('subject')
        chunk_start, chunk_end = (Chunk & key).fetch1(
            'chunk_start', 'chunk_end')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        session_info = aeon_api.sessiondata(raw_data_dir.as_posix(),
                                            start=pd.Timestamp(chunk_start),
                                            end=pd.Timestamp(chunk_end))

        self.insert1(key)
        self.Time.insert(({**key, 'subject': r.id,
                           'enter_exit_event': self._enter_exit_event_mapper[r.event],
                           'enter_exit_time': r.name} for _, r in session_info.iterrows()
                         if r.id in subject_list), skip_duplicates=True)


@schema
class SubjectWeight(dj.Imported):
    definition = """  # Records of subjects weights
    -> Chunk
    """

    class WeightTime(dj.Part):
        definition = """  # # Timestamps of each subject weight measurements
        -> master
        -> Experiment.Subject
        weight_time: datetime(6)  # datetime of subject weighting
        ---
        weight: float  #
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch('subject')
        chunk_start, chunk_end = (Chunk & key).fetch1(
            'chunk_start', 'chunk_end')
        raw_data_dir = Experiment.get_raw_data_directory(key)
        session_info = aeon_api.sessiondata(raw_data_dir.as_posix(),
                                            start=pd.Timestamp(chunk_start),
                                            end=pd.Timestamp(chunk_end))
        self.insert1(key)
        self.WeightTime.insert(({**key, 'subject': r.id,
                                 'weight': r.weight,
                                 'weight_time': r.name} for _, r in session_info.iterrows()
                                if r.id in subject_list), skip_duplicates=True)


@schema
class SubjectAnnotation(dj.Imported):
    definition = """  # Experimenter's annotations
    -> Chunk
    """

    class Annotation(dj.Part):
        definition = """
        -> master
        -> Experiment.Subject
        annotation_time: datetime(6)  # datetime of the annotation
        ---
        annotation: varchar(1000)
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch('subject')
        chunk_start, chunk_end = (Chunk & key).fetch1(
            'chunk_start', 'chunk_end')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        annotations = aeon_api.annotations(raw_data_dir.as_posix(),
                                           start=pd.Timestamp(chunk_start),
                                           end=pd.Timestamp(chunk_end))

        self.insert1(key)
        self.Annotation.insert({**key, 'subject': r.id,
                                'annotation': r.annotation,
                                'annotation_time': r.name} for _, r in annotations.iterrows()
                               if r.id in subject_list)


# ------------------- EVENTS --------------------


@schema
class EventType(dj.Lookup):
    definition = """  # Experimental event type
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
    -> Chunk
    -> ExperimentFoodPatch
    event_number: smallint
    ---
    event_time: datetime(6)  # event time
    -> EventType
    """

    @property
    def key_source(self):
        """
        Only the combination of Chunk and ExperimentFoodPatch with overlapping time
        +  Chunk(s) that started after FoodPatch install time and ended before FoodPatch remove time
        +  Chunk(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        """
        return (Chunk
                * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
                & 'chunk_start >= food_patch_install_time'
                & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")')

    def make(self, key):
        chunk_start, chunk_end = (Chunk & key).fetch1('chunk_start', 'chunk_end')
        food_patch_description = (ExperimentFoodPatch & key).fetch1('food_patch_description')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        pellet_data = aeon_api.pelletdata(raw_data_dir.as_posix(),
                                          device=food_patch_description,
                                          start=pd.Timestamp(chunk_start),
                                          end=pd.Timestamp(chunk_end))

        if not len(pellet_data):
            event_list = [{**key, 'event_number': 0,
                           'event_time': chunk_start, 'event_code': 1000}]
        else:
            event_code_mapper = {name: code for code, name
                                 in zip(*EventType.fetch('event_code', 'event_type'))}
            event_list = [{**key, 'event_number': r_idx, 'event_time': r_time,
                           'event_code': event_code_mapper[r.event]}
                          for r_idx, (r_time, r) in enumerate(pellet_data.iterrows())]

        self.insert(event_list)


@schema
class FoodPatchWheel(dj.Imported):
    definition = """  # Wheel data associated with a given ExperimentFoodPatch
    -> Chunk
    -> ExperimentFoodPatch
    ---
    timestamps:        longblob   # (datetime) timestamps of wheel encoder data
    angle:             longblob   # measured angles of the wheel
    intensity:         longblob
    """

    @property
    def key_source(self):
        """
        Only the combination of Chunk and ExperimentFoodPatch with overlapping time
        +  Chunk(s) that started after FoodPatch install time and ended before FoodPatch remove time
        +  Chunk(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        """
        return (Chunk
                * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
                & 'chunk_start >= food_patch_install_time'
                & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")')

    def make(self, key):
        chunk_start, chunk_end = (Chunk & key).fetch1('chunk_start', 'chunk_end')
        food_patch_description = (ExperimentFoodPatch & key).fetch1('food_patch_description')

        raw_data_dir = Experiment.get_raw_data_directory(key)
        wheel_data = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                          device=food_patch_description,
                                          start=pd.Timestamp(chunk_start),
                                          end=pd.Timestamp(chunk_end))
        timestamps = (wheel_data.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        self.insert1({**key, 'timestamps': timestamps,
                      'angle': wheel_data.angle.values,
                      'intensity': wheel_data.intensity.values})


@schema
class WheelState(dj.Imported):
    definition = """  # Wheel states associated with a given ExperimentFoodPatch
    -> Chunk
    -> ExperimentFoodPatch
    """

    class Time(dj.Part):
        definition = """  # Threshold, d1, delta state of the wheel
        -> master
        state_timestamp: datetime(6)
        ---
        threshold: float
        d1: float
        delta: float
        """

    @property
    def key_source(self):
        """
        Only the combination of Chunk and ExperimentFoodPatch with overlapping time
        +  Chunk(s) that started after FoodPatch install time and ended before FoodPatch remove time
        +  Chunk(s) that started after FoodPatch install time for FoodPatch that are not yet removed
        """
        return (Chunk
                * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
                & 'chunk_start >= food_patch_install_time'
                & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")')

    def make(self, key):
        chunk_start, chunk_end = (Chunk & key).fetch1('chunk_start', 'chunk_end')
        food_patch_description = (ExperimentFoodPatch & key).fetch1('food_patch_description')
        raw_data_dir = Experiment.get_raw_data_directory(key)
        wheel_state = aeon_api.patchdata(raw_data_dir.as_posix(),
                                         patch=food_patch_description,
                                         start=pd.Timestamp(chunk_start),
                                         end=pd.Timestamp(chunk_end))
        self.insert1(key)
        self.Time.insert([{**key,
                           'state_timestamp': r.name,
                           'threshold': r.threshold,
                           'd1': r.d1,
                           'delta': r.delta} for _, r in wheel_state.iterrows()])


# ------------------- SESSION --------------------


@schema
class SessionType(dj.Lookup):
    definition = """
    session_type: varchar(32)
    ---
    type_description: varchar(1000)
    """

    contents = [('foraging', 'Freely behaving foraging session,'
                             ' defined as the time period between the animal'
                             ' entering and exiting the arena')]


@schema
class Session(dj.Computed):
    definition = """  # A session spans the time when the animal firsts enter the arena to when it exits the arena
    -> Experiment.Subject
    session_start: datetime(6)
    ---
    -> SessionType
    """

    @property
    def key_source(self):
        return (dj.U('experiment_name', 'subject', 'session_start')
                & (SubjectEnterExit.Time & 'enter_exit_event = "enter"').proj(
                    session_start='enter_exit_time'))

    def make(self, key):
        self.insert1({**key, 'session_type': 'foraging'})


@schema
class NeverExitedSession(dj.Manual):
    definition = """  # Bad session where the animal seemed to have never exited
    -> Session
    """


@schema
class SessionEnd(dj.Computed):
    definition = """
    -> Session
    ---
    session_end: datetime(6)
    session_duration: float  # (hour)
    """

    key_source = (Session
                  - NeverExitedSession
                  & (Session * SubjectEnterExit.Time
                     & 'enter_exit_event = "exit"'
                     & 'enter_exit_time > session_start'))

    def make(self, key):
        session_start = key['session_start']
        subject_exit = (SubjectEnterExit.Time
                        & {'subject': key['subject']}
                        & f'enter_exit_time > "{session_start}"').fetch(
            as_dict=True, limit=1, order_by='enter_exit_time ASC')[0]

        if subject_exit['enter_exit_event'] != 'exit':
            NeverExitedSession.insert1(key, skip_duplicates=True)
            return

        session_end = subject_exit['enter_exit_time']
        duration = (session_end - session_start).total_seconds() / 3600

        # insert
        self.insert1({**key,
                      'session_end': session_end,
                      'session_duration': duration})


# ------------------- ACQUISITION TIMESLICE --------------------


@schema
class TimeSlice(dj.Computed):
    definition = """
    # A short time-slice (e.g. 10 minutes) of the recording of a given animal in the arena
    -> Session
    -> Chunk
    time_slice_start: datetime(6)  # datetime of the start of this time slice
    ---
    time_slice_end: datetime(6)    # datetime of the end of this time slice
    """

    @property
    def key_source(self):
        """
        Chunk for all sessions:
        + are not "NeverExitedSession"
        + session_start during this Chunk - i.e. first chunk of the session
        + session_end during this Chunk - i.e. last chunk of the session
        + chunk starts after session_start and ends before session_end (or NOW() - i.e. session still on going)
        """
        return (Session.join(SessionEnd, left=True).proj(
            session_end='IFNULL(session_end, NOW())') * Chunk
                - NeverExitedSession
                & SubjectEnterExit
                & ['session_start BETWEEN chunk_start AND chunk_end',
                   'session_end BETWEEN chunk_start AND chunk_end',
                   'chunk_start >= session_start AND chunk_end <= session_end'])

    _time_slice_duration = datetime.timedelta(hours=0, minutes=10, seconds=0)

    def make(self, key):
        chunk_start, chunk_end = (Chunk & key).fetch1(
            'chunk_start', 'chunk_end')

        # -- Determine the time to start time_slicing in this chunk
        if chunk_start < key['session_start'] < chunk_end:
            # For chunk containing the session_start - i.e. first chunk of this session
            start_time = key['session_start']
        else:
            # For chunks after the first chunk of this session
            start_time = chunk_start

        # -- Determine the time to end time_slicing in this chunk
        # get the enter/exit events in this chunk that are after the session_start
        next_enter_exit_events = (SubjectEnterExit.Time
                                  & key & f'enter_exit_time > "{key["session_start"]}"')
        if not next_enter_exit_events:
            # No enter/exit event: time_slices from this whole chunk
            end_time = chunk_end
        else:
            next_event = next_enter_exit_events.fetch(
                as_dict=True, order_by='enter_exit_time DESC', limit=1)[0]
            if next_event['enter_exit_event'] == 'enter':
                NeverExitedSession.insert1(key, ignore_extra_fields=True, skip_duplicates=True)
                return
            end_time = next_event['enter_exit_time']

        chunk_time_slices = []
        time_slice_start = start_time
        while time_slice_start < end_time:
            time_slice_end = time_slice_start + min(self._time_slice_duration,
                                                    end_time - time_slice_start)
            chunk_time_slices.append({**key,
                                      'time_slice_start': time_slice_start,
                                      'time_slice_end': time_slice_end})
            time_slice_start = time_slice_end

        self.insert(chunk_time_slices)


# ---- Task Protocol categorization ----


@schema
class TaskProtocol(dj.Lookup):
    definition = """
    task_protocol: int
    ---
    protocol_params: longblob
    protocol_description: varchar(255)
    """


@schema
class TimeSliceProtocol(dj.Computed):
    definition = """
    -> TimeSlice
    ---
    -> TaskProtocol
    """

    def make(self, key):
        pass
