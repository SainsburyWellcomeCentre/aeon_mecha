import datajoint as dj
import datetime
import pathlib
import numpy as np
import pandas as pd

from aeon.io import api as aeon_api

from . import lab, subject
from . import get_schema_name, paths
from .ingest import extract_epoch_metadata, ingest_epoch_metadata


schema = dj.schema(get_schema_name("acquisition"))


# ------------------- Type Lookup ------------------------


@schema
class ExperimentType(dj.Lookup):
    definition = """
    experiment_type: varchar(32)
    """

    contents = zip(['foraging', 'social'])


@schema
class EventType(dj.Lookup):
    definition = """  # Experimental event type
    event_code: smallint
    ---
    event_type: varchar(36)
    """

    contents = [(220, 'SubjectEnteredArena'), (221, 'SubjectExitedArena'), (222, 'SubjectRemovedFromArena'),
                (35, "TriggerPellet"), (32, "PelletDetected"), (1000, "No Events")]


# ------------------- Data repository/directory ------------------------


@schema
class PipelineRepository(dj.Lookup):
    definition = """
    repository_name: varchar(16)
    """

    contents = zip(["ceph_aeon"])


@schema
class DirectoryType(dj.Lookup):
    definition = """
    directory_type: varchar(16)
    """

    contents = zip(["raw", "preprocessing", "analysis", "quality-control"])


# ------------------- GENERAL INFORMATION ABOUT AN EXPERIMENT --------------------


@schema
class Experiment(dj.Manual):
    definition = """
    experiment_name: varchar(32)  # e.g exp0-r0
    ---
    experiment_start_time: datetime(6)  # datetime of the start of this experiment
    experiment_description: varchar(1000)
    -> lab.Arena
    -> lab.Location  # lab/room where a particular experiment takes place
    -> ExperimentType
    """

    class Subject(dj.Part):
        definition = """  # the subjects participating in this experiment
        -> master
        -> subject.Subject
        """

    class Directory(dj.Part):
        definition = """
        -> master
        -> DirectoryType
        ---
        -> PipelineRepository
        directory_path: varchar(255)
        """

    @classmethod
    def get_data_directory(cls, experiment_key, directory_type="raw", as_posix=False):
        repo_name, dir_path = (
            cls.Directory & experiment_key & {"directory_type": directory_type}
        ).fetch1("repository_name", "directory_path")
        data_directory = paths.get_repository_path(repo_name) / dir_path
        if not data_directory.exists():
            return None
        return data_directory.as_posix() if as_posix else data_directory

    @classmethod
    def get_data_directories(
        cls, experiment_key, directory_types=["raw"], as_posix=False
    ):
        return [
            cls.get_data_directory(experiment_key, dir_type, as_posix=as_posix)
            for dir_type in directory_types
        ]


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
    camera_gain=null: float      # gain value applied to the acquired video
    camera_bin=null: int         # bin-size applied to the acquired video
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
    wheel_radius=null: float    # (cm)
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

    class Vertex(dj.Part):
        definition = """
        -> master
        vertex: int
        ---
        vertex_x: float    # (m) x-coordinate of the vertex, in the arena's coordinate frame
        vertex_y: float    # (m) y-coordinate of the vertex, in the arena's coordinate frame
        vertex_z=0: float  # (m) z-coordinate of the vertex, in the arena's coordinate frame
        """


@schema
class ExperimentWeightScale(dj.Manual):
    definition = """
    # Scale for measuring animal weights
    -> Experiment
    -> lab.WeightScale
    weight_scale_install_time: datetime(6)   # time of the weight_scale placed and started operation at this position
    ---
    -> lab.ArenaNest
    weight_scale_description: varchar(36)
    weight_scale_sampling_rate=null: float  # (Hz) scale sampling rate
    """

    class RemovalTime(dj.Part):
        definition = """
        -> master
        ---
        weight_scale_remove_time: datetime(6)  # time of the weight_scale being removed from this position
        """


# ------------------- ACQUISITION EPOCH --------------------


@schema
class Epoch(dj.Manual):
    definition = """  # A recording period reflecting on/off of the hardware acquisition system
    -> Experiment
    epoch_start: datetime(6)
    """

    class Config(dj.Part):
        definition = """
        -> master
        ---
        bonsai_workflow: varchar(36)
        revision: varchar(64)   # e.g. git hash of aeon_experiment used to generated this particular epoch
        source='': varchar(16)  # e.g. aeon_experiment or aeon_acquisition (or others)
        metadata: longblob
        -> Experiment.Directory
        metadata_file_path: varchar(255)  # path of the file, relative to the experiment repository
        """

    experiment_ref_device_mapper = {'exp0.1-r0': 'FrameTop',
                                    'exp0.2-r0': 'CameraTop'}

    @classmethod
    def ingest_epochs(cls, experiment_name):
        device_name = Epoch.experiment_ref_device_mapper.get(experiment_name, 'CameraTop')

        all_chunks, raw_data_dirs = _get_all_chunks(experiment_name, device_name)

        epoch_list = []
        for i, (_, chunk) in enumerate(all_chunks.iterrows()):
            chunk_rep_file = pathlib.Path(chunk.path)
            epoch_dir = pathlib.Path(chunk_rep_file.as_posix().split(device_name)[0])
            epoch_start = datetime.datetime.strptime(epoch_dir.name, '%Y-%m-%dT%H-%M-%S')

            # --- insert to Epoch ---
            epoch_key = {"experiment_name": experiment_name, "epoch_start": epoch_start}

            if cls & epoch_key or epoch_key in epoch_list:
                # skip over those already ingested
                continue

            epoch_config, metadata_yml_filepath = None, None
            if experiment_name != 'exp0.1-r0':
                metadata_yml_filepath = epoch_dir / 'Metadata.yml'
                if metadata_yml_filepath.exists():
                    epoch_config = extract_epoch_metadata(experiment_name, metadata_yml_filepath)

                    metadata_yml_filepath = epoch_config['metadata_file_path']

                    _, directory, repo_path = _match_experiment_directory(
                        experiment_name, epoch_config['metadata_file_path'], raw_data_dirs)
                    epoch_config = {
                        **epoch_config,
                        **directory,
                        'metadata_file_path': epoch_config['metadata_file_path'].relative_to(repo_path).as_posix()}

            # find previous epoch end-time
            previous_epoch_end = None
            if i > 0:
                previous_chunk = all_chunks.iloc[i-1]
                previous_chunk_path = pathlib.Path(previous_chunk.path)
                previous_epoch_dir = pathlib.Path(previous_chunk_path.as_posix().split(device_name)[0])
                previous_epoch_start = datetime.datetime.strptime(previous_epoch_dir.name, '%Y-%m-%dT%H-%M-%S')
                previous_chunk_end = previous_chunk.name + datetime.timedelta(hours=aeon_api.CHUNK_DURATION)
                previous_epoch_end = min(previous_chunk_end, epoch_start)

            with cls.connection.transaction:
                cls.insert1(epoch_key)
                if previous_epoch_end:
                    EpochEnd.insert1(
                        {"experiment_name": experiment_name,
                         "epoch_start": previous_epoch_start,
                         "epoch_end": previous_epoch_end,
                         "epoch_duration": (previous_epoch_end - previous_epoch_start).total_seconds() / 3600})
                if epoch_config:
                    cls.Config.insert1(epoch_config)
                    ingest_epoch_metadata(experiment_name, metadata_yml_filepath)

            epoch_list.append(epoch_key)

        print(f"Inserted {len(epoch_list)} new Epochs")


@schema
class EpochEnd(dj.Manual):
    definition = """ 
    -> Epoch
    ---
    epoch_end: datetime(6)
    epoch_duration: float  # (hour)
    """

# ------------------- ACQUISITION CHUNK --------------------


@schema
class Chunk(dj.Manual):
    definition = """  # A recording period corresponds to a 1-hour data acquisition
    -> Experiment
    chunk_start: datetime(6)  # datetime of the start of a given acquisition chunk
    ---
    chunk_end: datetime(6)    # datetime of the end of a given acquisition chunk
    -> Experiment.Directory   # the data directory storing the acquired data for a given chunk
    -> Epoch
    """

    class File(dj.Part):
        definition = """
        -> master
        file_number: int
        ---
        file_name: varchar(128)
        -> Experiment.Directory
        file_path: varchar(255)  # path of the file, relative to the data repository
        """

    @classmethod
    def ingest_chunks(cls, experiment_name):
        device_name = Epoch.experiment_ref_device_mapper.get(experiment_name, 'CameraTop')

        all_chunks, raw_data_dirs = _get_all_chunks(experiment_name, device_name)

        chunk_starts, chunk_list, file_list, file_name_list = [], [], [], []
        for _, chunk in all_chunks.iterrows():
            chunk_rep_file = pathlib.Path(chunk.path)
            epoch_dir = pathlib.Path(chunk_rep_file.as_posix().split(device_name)[0])
            epoch_start = datetime.datetime.strptime(epoch_dir.name, '%Y-%m-%dT%H-%M-%S')

            epoch_key = {"experiment_name": experiment_name, "epoch_start": epoch_start}
            if not (Epoch & epoch_key):
                # skip over if epoch is not yet inserted
                continue

            chunk_start = chunk.name
            chunk_end = chunk_start + datetime.timedelta(hours=aeon_api.CHUNK_DURATION)

            # --- insert to Chunk ---
            chunk_key = {"experiment_name": experiment_name, "chunk_start": chunk_start}

            if cls.proj() & chunk_key:
                # skip over those already ingested
                continue

            if chunk_start in chunk_starts:
                # handle cases where two chunks with identical start_time
                # (starts in the same hour) but from 2 consecutive epochs
                # using epoch_start as chunk_start in this case
                chunk_key['chunk_start'] = epoch_start

            # chunk file and directory
            raw_data_dir, directory, repo_path = _match_experiment_directory(
                experiment_name, chunk_rep_file, raw_data_dirs)

            chunk_starts.append(chunk_key['chunk_start'])
            chunk_list.append({**chunk_key, **directory, "chunk_end": chunk_end, **epoch_key})
            file_name_list.append(
                chunk_rep_file.name
            )  # handle duplicated files in different folders

            # -- files --
            file_datetime_str = chunk_rep_file.stem.replace(f"{device_name}_", "")
            files = list(pathlib.Path(raw_data_dir).rglob(f"*{file_datetime_str}*"))

            file_list.extend(
                {
                    **chunk_key,
                    **directory,
                    "file_number": f_idx,
                    "file_name": f.name,
                    "file_path": f.relative_to(repo_path).as_posix(),
                }
                for f_idx, f in enumerate(files)
            )

        # insert
        print(f"Insert {len(chunk_list)} new Chunks")

        with cls.connection.transaction:
            cls.insert(chunk_list)
            cls.File.insert(file_list)


@schema
class SubjectEnterExit(dj.Imported):
    definition = """  # Records of subjects entering/exiting the arena
    -> Chunk
    """

    _enter_exit_event_mapper = {"Start": 220, "End": 221}

    class Time(dj.Part):
        definition = """  # Timestamps of each entering/exiting events
        -> master
        -> Experiment.Subject
        enter_exit_time: datetime(6)  # datetime of subject entering/exiting the arena
        ---
        -> EventType
        """

    def make(self, key):
        subject_list = (Experiment.Subject & key).fetch("subject")
        chunk_start, chunk_end = (Chunk & key).fetch1("chunk_start", "chunk_end")

        raw_data_dir = Experiment.get_data_directory(key)
        session_info = aeon_api.sessiondata(
            raw_data_dir.as_posix(),
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        self.insert1(key)
        self.Time.insert(
            (
                {
                    **key,
                    "subject": r.id,
                    "event_code": self._enter_exit_event_mapper[r.event],
                    "enter_exit_time": r.name,
                }
                for _, r in session_info.iterrows()
                if r.id in subject_list
            ),
            skip_duplicates=True,
        )


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
        subject_list = (Experiment.Subject & key).fetch("subject")
        chunk_start, chunk_end = (Chunk & key).fetch1("chunk_start", "chunk_end")
        raw_data_dir = Experiment.get_data_directory(key)
        session_info = aeon_api.sessiondata(
            raw_data_dir.as_posix(),
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )
        self.insert1(key)
        self.WeightTime.insert(
            (
                {**key, "subject": r.id, "weight": r.weight, "weight_time": r.name}
                for _, r in session_info.iterrows()
                if r.id in subject_list
            ),
            skip_duplicates=True,
        )


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
        subject_list = (Experiment.Subject & key).fetch("subject")
        chunk_start, chunk_end = (Chunk & key).fetch1("chunk_start", "chunk_end")

        raw_data_dir = Experiment.get_data_directory(key)
        annotations = aeon_api.annotations(
            raw_data_dir.as_posix(),
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        self.insert1(key)
        self.Annotation.insert(
            {
                **key,
                "subject": r.id,
                "annotation": r.annotation,
                "annotation_time": r.name,
            }
            for _, r in annotations.iterrows()
            if r.id in subject_list
        )


# ------------------- EVENTS --------------------


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
        return (
            Chunk * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
            & "chunk_start >= food_patch_install_time"
            & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")'
        )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (Chunk & key).fetch1('chunk_start', 'chunk_end', 'directory_type')
        food_patch_description = (ExperimentFoodPatch & key).fetch1(
            "food_patch_description"
        )

        raw_data_dir = Experiment.get_data_directory(key, directory_type=dir_type)
        pellet_data = aeon_api.pelletdata(
            raw_data_dir.as_posix(),
            device=food_patch_description,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        if not len(pellet_data):
            event_list = [
                {
                    **key,
                    "event_number": 0,
                    "event_time": chunk_start,
                    "event_code": 1000,
                }
            ]
        else:
            event_code_mapper = {
                name: code
                for code, name in zip(*EventType.fetch("event_code", "event_type"))
            }
            event_list = [
                {
                    **key,
                    "event_number": r_idx,
                    "event_time": r_time,
                    "event_code": event_code_mapper[r.event],
                }
                for r_idx, (r_time, r) in enumerate(pellet_data.iterrows())
            ]

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
        return (
            Chunk * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
            & "chunk_start >= food_patch_install_time"
            & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")'
        )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (Chunk & key).fetch1('chunk_start', 'chunk_end', 'directory_type')
        food_patch_description = (ExperimentFoodPatch & key).fetch1(
            "food_patch_description"
        )

        raw_data_dir = Experiment.get_data_directory(key, directory_type=dir_type)
        wheel_data = aeon_api.encoderdata(
            raw_data_dir.as_posix(),
            device=food_patch_description,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )
        timestamps = wheel_data.index.to_pydatetime()

        self.insert1(
            {
                **key,
                "timestamps": timestamps,
                "angle": wheel_data.angle.values,
                "intensity": wheel_data.intensity.values,
            }
        )


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
        return (
            Chunk * ExperimentFoodPatch.join(ExperimentFoodPatch.RemovalTime, left=True)
            & "chunk_start >= food_patch_install_time"
            & 'chunk_start < IFNULL(food_patch_remove_time, "2200-01-01")'
        )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (Chunk & key).fetch1('chunk_start', 'chunk_end', 'directory_type')
        food_patch_description = (ExperimentFoodPatch & key).fetch1(
            "food_patch_description"
        )
        raw_data_dir = Experiment.get_data_directory(key, directory_type=dir_type)
        wheel_state = aeon_api.patchdata(
            raw_data_dir.as_posix(),
            patch=food_patch_description,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )
        self.insert1(key)
        self.Time.insert(
            [
                {
                    **key,
                    "state_timestamp": r.name,
                    "threshold": r.threshold,
                    "d1": r.d1,
                    "delta": r.delta,
                }
                for _, r in wheel_state.iterrows()
            ]
        )


@schema
class WeightMeasurement(dj.Imported):
    definition = """  # Raw scale measurement associated with a given ExperimentScale
    -> Chunk
    -> ExperimentWeightScale
    ---
    timestamps:        longblob   # (datetime) timestamps of scale data
    weight:            longblob   # measured weights
    confidence:        longblob   # confidence level of the measured weights [0-1]
    """

    @property
    def key_source(self):
        """
        Only the combination of Chunk and ExperimentWeightScale with overlapping time
        +  Chunk(s) that started after WeightScale install time and ended before WeightScale remove time
        +  Chunk(s) that started after WeightScale install time for WeightScale that are not yet removed
        """
        return (
            Chunk * ExperimentWeightScale.join(ExperimentWeightScale.RemovalTime, left=True)
            & "chunk_start >= weight_scale_install_time"
            & 'chunk_start < IFNULL(weight_scale_remove_time, "2200-01-01")'
        )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (Chunk & key).fetch1('chunk_start', 'chunk_end', 'directory_type')
        weight_scale_description = (ExperimentWeightScale & key).fetch1('weight_scale_description')

        raw_data_dir = Experiment.get_data_directory(key, directory_type=dir_type)
        scale_data = aeon_api.weightdata(raw_data_dir.as_posix(),
                                         device=weight_scale_description,
                                         start=pd.Timestamp(chunk_start),
                                         end=pd.Timestamp(chunk_end))

        timestamps = scale_data.index.to_pydatetime()

        self.insert1({**key, 'timestamps': timestamps,
                      'weight': scale_data.weight.values,
                      'confidence': scale_data.stable.values.astype(float)})


# @schema
# class MultiAnimalSession(dj.Computed):
#     definition = """
#     -> Experiment
#     session_start: datetime(6)
#     ---
#     -> SessionType
#     session_end: datetime(6)
#     session_duration: float  # (hour)
#     """
#
#     class Session(dj.Part):
#         definition = """
#         -> master
#         -> Session
#         """


# ---- Task Protocol categorization ----


@schema
class TaskProtocol(dj.Lookup):
    definition = """
    task_protocol: int
    ---
    protocol_params: longblob
    protocol_description: varchar(255)
    """


# ---- HELPERS ----

def _get_all_chunks(experiment_name, device_name):
    raw_data_dirs = Experiment.get_data_directories(
        {"experiment_name": experiment_name},
        directory_types=["quality-control", "raw"],
        as_posix=True,
    )
    raw_data_dirs = {
        dir_type: data_dir
        for dir_type, data_dir in zip(["quality-control", "raw"], raw_data_dirs)
        if data_dir
    }

    return aeon_api.chunkdata(raw_data_dirs.values(), device_name), raw_data_dirs


def _match_experiment_directory(experiment_name, path, directories):
    for k, v in directories.items():
        raw_data_dir = v
        if pathlib.Path(raw_data_dir) in list(path.parents):
            directory = (
                    Experiment.Directory.proj("repository_name")
                    & {"experiment_name": experiment_name, "directory_type": k}
            ).fetch1()
            repo_path = paths.get_repository_path(
                directory.pop("repository_name")
            )
            break
    else:
        raise FileNotFoundError(
            f"Unable to identify the directory"
            f" where this chunk is from: {path}"
        )

    return raw_data_dir, directory, repo_path
