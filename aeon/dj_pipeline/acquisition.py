import datetime
import pathlib
import re
import datajoint as dj
import numpy as np
import pandas as pd
import json

from aeon.io import api as io_api
from aeon.schema import schemas as aeon_schemas
from aeon.io import reader as io_reader
from aeon.analysis import utils as analysis_utils

from aeon.dj_pipeline import get_schema_name, lab, subject
from aeon.dj_pipeline.utils import paths

logger = dj.logger
schema = dj.schema(get_schema_name("acquisition"))

# ------------------- Some Constants --------------------------

_ref_device_mapping = {
    "exp0.1-r0": "FrameTop",
    "social0-r1": "FrameTop",
    "exp0.2-r0": "CameraTop",
}


# ------------------- Type Lookup ------------------------


@schema
class ExperimentType(dj.Lookup):
    definition = """
    experiment_type: varchar(32)
    """

    contents = zip(["foraging", "social"])


@schema
class EventType(dj.Lookup):
    definition = """  # Experimental event type
    event_code: smallint
    ---
    event_type: varchar(36)
    """

    contents = [
        (220, "SubjectEnteredArena"),
        (221, "SubjectExitedArena"),
        (222, "SubjectRemovedFromArena"),
        (223, "SubjectRemainedInArena"),
        (35, "TriggerPellet"),
        (32, "PelletDetected"),
        (1000, "No Events"),
    ]


@schema
class DevicesSchema(dj.Lookup):
    definition = """
    devices_schema_name: varchar(32)
    """

    contents = zip(aeon_schemas.__all__)


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

    contents = zip(["raw", "processed", "qc"])


# ------------------- GENERAL INFORMATION ABOUT AN EXPERIMENT --------------------


@schema
class Experiment(dj.Manual):
    definition = """
    experiment_name: varchar(32)  # e.g exp0-aeon3
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
        load_order=1: int  # order of priority to load the directory
        """

    class DevicesSchema(dj.Part):
        definition = """
        -> master
        ---
        -> DevicesSchema
        """

    class Note(dj.Part):
        definition = """
        -> master
        note_timestamp: datetime
        ---
        note_type: varchar(64)
        note: varchar(1000)
        """

    @classmethod
    def get_data_directory(cls, experiment_key, directory_type="raw", as_posix=False):
        try:
            repo_name, dir_path = (
                cls.Directory & experiment_key & {"directory_type": directory_type}
            ).fetch1("repository_name", "directory_path")
        except dj.errors.DataJointError:
            return

        dir_path = pathlib.Path(dir_path)
        if dir_path.exists():
            assert dir_path.is_relative_to(paths.get_repository_path(repo_name))
            data_directory = dir_path
        else:
            data_directory = paths.get_repository_path(repo_name) / dir_path
            if not data_directory.exists():
                return
        return data_directory.as_posix() if as_posix else data_directory

    @classmethod
    def get_data_directories(cls, experiment_key, directory_types=None, as_posix=False):
        if directory_types is None:
            directory_types = (cls.Directory & experiment_key).fetch(
                "directory_type", order_by="load_order"
            )
        return [
            d
            for dir_type in directory_types
            if (d := cls.get_data_directory(experiment_key, dir_type, as_posix=as_posix)) is not None
        ]


# ------------------- ACQUISITION EPOCH --------------------


@schema
class Epoch(dj.Manual):
    definition = """  # A recording period reflecting on/off of the hardware acquisition system.
    -> Experiment
    epoch_start: datetime(6)
    ---
    -> [nullable] Experiment.Directory
    epoch_dir='': varchar(255)  # path of the directory storing the acquired data for a given epoch
    """

    @classmethod
    def ingest_epochs(cls, experiment_name):
        """Ingest epochs for the specified "experiment_name" """
        device_name = _ref_device_mapping.get(experiment_name, "CameraTop")

        all_chunks, raw_data_dirs = _get_all_chunks(experiment_name, device_name)

        epoch_list = []
        for i, (_, chunk) in enumerate(all_chunks.iterrows()):
            chunk_rep_file = pathlib.Path(chunk.path)
            epoch_dir = pathlib.Path(chunk_rep_file.as_posix().split(device_name)[0])
            epoch_start = datetime.datetime.strptime(epoch_dir.name, "%Y-%m-%dT%H-%M-%S")
            # --- insert to Epoch ---
            epoch_key = {"experiment_name": experiment_name, "epoch_start": epoch_start}

            if epoch_start == "2023-12-13 15:20:48":
                break

            # skip over those already ingested
            if cls & epoch_key or epoch_key in epoch_list:
                continue

            raw_data_dir, directory, _ = _match_experiment_directory(
                experiment_name,
                epoch_dir,
                raw_data_dirs,
            )

            # find previous epoch end-time
            previous_epoch_key = None
            if i > 0:
                previous_chunk = all_chunks.iloc[i - 1]
                previous_chunk_path = pathlib.Path(previous_chunk.path)
                previous_epoch_dir = pathlib.Path(previous_chunk_path.as_posix().split(device_name)[0])
                previous_epoch_start = datetime.datetime.strptime(
                    previous_epoch_dir.name, "%Y-%m-%dT%H-%M-%S"
                )
                previous_chunk_end = previous_chunk.name + datetime.timedelta(hours=io_api.CHUNK_DURATION)
                previous_epoch_end = min(previous_chunk_end, epoch_start)
                previous_epoch_key = {
                    "experiment_name": experiment_name,
                    "epoch_start": previous_epoch_start,
                }

            with cls.connection.transaction:
                # insert new epoch
                cls.insert1(
                    {**epoch_key, **directory, "epoch_dir": epoch_dir.relative_to(raw_data_dir).as_posix()}
                )
                epoch_list.append(epoch_key)

                # update previous epoch
                if (
                    previous_epoch_key
                    and (cls & previous_epoch_key)
                    and not (EpochEnd & previous_epoch_key)
                ):
                    # insert end-time for previous epoch
                    EpochEnd.insert1(
                        {
                            **previous_epoch_key,
                            "epoch_end": previous_epoch_end,
                            "epoch_duration": (previous_epoch_end - previous_epoch_start).total_seconds()
                            / 3600,
                        }
                    )
                    # update end-time for last chunk of the previous epoch
                    if Chunk & {
                        "experiment_name": experiment_name,
                        "chunk_start": previous_chunk.name,
                    }:
                        Chunk.update1(
                            {
                                "experiment_name": experiment_name,
                                "chunk_start": previous_chunk.name,
                                "chunk_end": previous_epoch_end,
                            }
                        )

        logger.info(f"Insert {len(epoch_list)} new Epoch(s)")


@schema
class EpochEnd(dj.Manual):
    definition = """
    -> Epoch
    ---
    epoch_end: datetime(6)
    epoch_duration: float  # (hour)
    """


@schema
class EpochConfig(dj.Imported):
    definition = """
    -> Epoch
    """

    class Meta(dj.Part):
        definition = """ # Metadata for the configuration of a given epoch
        -> master
        ---
        bonsai_workflow: varchar(36)
        commit: varchar(64)   # e.g. git commit hash of aeon_experiment used to generated this particular epoch
        source='': varchar(16)  # e.g. aeon_experiment or aeon_acquisition (or others)
        metadata: longblob
        metadata_file_path: varchar(255)  # path of the file, relative to the experiment repository
        """

    class DeviceType(dj.Part):
        definition = """  # Device type(s) used in a particular acquisition epoch
        -> master
        device_type: varchar(36)
        """

    class ActiveRegion(dj.Part):
        definition = """
        -> master
        region_name: varchar(36)
        ---
        region_data: longblob
        """

    def make(self, key):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import (
            extract_epoch_config,
            ingest_epoch_metadata,
            insert_device_types,
        )

        experiment_name = key["experiment_name"]
        devices_schema = getattr(
            aeon_schemas,
            (Experiment.DevicesSchema & {"experiment_name": experiment_name}).fetch1("devices_schema_name"),
        )

        dir_type, epoch_dir = (Epoch & key).fetch1("directory_type", "epoch_dir")
        data_dir = Experiment.get_data_directory(key, dir_type)
        metadata_yml_filepath = data_dir / epoch_dir / "Metadata.yml"

        epoch_config = extract_epoch_config(experiment_name, devices_schema, metadata_yml_filepath)
        epoch_config = {
            **epoch_config,
            "metadata_file_path": metadata_yml_filepath.relative_to(data_dir).as_posix(),
        }

        # Insert new entries for streams.DeviceType, streams.Device.
        insert_device_types(
            devices_schema,
            metadata_yml_filepath,
        )
        # Define and instantiate new devices/stream tables under `streams` schema
        streams_maker.main()
        # Insert devices' installation/removal/settings
        epoch_device_types = ingest_epoch_metadata(experiment_name, devices_schema, metadata_yml_filepath)

        self.insert1(key)
        self.Meta.insert1(epoch_config)
        self.DeviceType.insert(key | {"device_type": n} for n in epoch_device_types or {})
        with metadata_yml_filepath.open("r") as f:
            metadata = json.load(f)
        self.ActiveRegion.insert(
            {**key, "region_name": k, "region_data": v} for k, v in metadata["ActiveRegion"].items()
        )


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
        device_name = _ref_device_mapping.get(experiment_name, "CameraTop")

        all_chunks, raw_data_dirs = _get_all_chunks(experiment_name, device_name)

        chunk_starts, chunk_list, file_list, file_name_list = [], [], [], []
        for _, chunk in all_chunks.iterrows():
            chunk_rep_file = pathlib.Path(chunk.path)
            epoch_dir = pathlib.Path(chunk_rep_file.as_posix().split(device_name)[0])
            epoch_start = datetime.datetime.strptime(epoch_dir.name, "%Y-%m-%dT%H-%M-%S")

            epoch_key = {"experiment_name": experiment_name, "epoch_start": epoch_start}
            if not (Epoch & epoch_key):
                # skip over if epoch is not yet inserted
                continue

            chunk_start = chunk.name
            chunk_end = chunk_start + datetime.timedelta(hours=io_api.CHUNK_DURATION)

            if EpochEnd & epoch_key:
                epoch_end = (EpochEnd & epoch_key).fetch1("epoch_end")
                chunk_end = min(chunk_end, epoch_end)

            if chunk_start in chunk_starts:
                # handle cases where two chunks with identical start_time
                # (starts in the same hour) but from 2 consecutive epochs
                # using epoch_start as chunk_start in this case
                chunk_start = epoch_start

            # --- insert to Chunk ---
            chunk_key = {"experiment_name": experiment_name, "chunk_start": chunk_start}

            if cls.proj() & chunk_key:
                # skip over those already ingested
                continue

            # chunk file and directory
            raw_data_dir, directory, repo_path = _match_experiment_directory(
                experiment_name, chunk_rep_file, raw_data_dirs
            )

            chunk_starts.append(chunk_key["chunk_start"])
            chunk_list.append({**chunk_key, **directory, "chunk_end": chunk_end, **epoch_key})
            file_name_list.append(chunk_rep_file.name)  # handle duplicated files in different folders

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
        logger.info(f"Insert {len(chunk_list)} new Chunk(s)")

        with cls.connection.transaction:
            cls.insert(chunk_list)
            cls.File.insert(file_list)


# ------------------- ENVIRONMENT --------------------


@schema
class Environment(dj.Imported):
    definition = """  # Experiment environments
    -> Chunk
    """

    class EnvironmentState(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        state: longblob
        """

    class BlockState(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        pellet_ct: longblob
        pellet_ct_thresh: longblob
        due_time: longblob
        """

    class LightEvents(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        channel: longblob
        value: longblob
        """

    class MessageLog(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) 
        priority: longblob
        type: longblob
        message: longblob
        """

    class SubjectState(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        id: longblob
        weight: longblob
        type: longblob
        """

    class SubjectVisits(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        id: longblob
        type: longblob
        region: longblob
        """

    class SubjectWeight(dj.Part):
        definition = """
        -> master
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps
        weight: longblob
        confidence: longblob
        subject_id: longblob
        int_id: longblob
        """

    def make(self, key):
        chunk_start, chunk_end = (Chunk & key).fetch1("chunk_start", "chunk_end")

        # Populate the part table
        data_dirs = Experiment.get_data_directories(key)
        devices_schema = getattr(
            aeon_schemas,
            (Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                "devices_schema_name"
            ),
        )
        device = devices_schema.Environment

        self.insert1(key)

        for stream_type, part_table in [
            ("EnvironmentState", self.EnvironmentState),
            ("BlockState", self.BlockState),
            ("LightEvents", self.LightEvents),
            ("MessageLog", self.MessageLog),
            ("SubjectState", self.SubjectState),
            ("SubjectVisits", self.SubjectVisits),
            ("SubjectWeight", self.SubjectWeight),
        ]:
            stream_reader = getattr(device, stream_type)

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            part_table.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        re.sub(r"\([^)]*\)", "", c): stream_data[c].values
                        for c in stream_reader.columns
                        if not c.startswith("_")
                    },
                },
                ignore_extra_fields=True,
            )


# ---- HELPERS ----


def _get_all_chunks(experiment_name, device_name):
    directory_types = ["quality-control", "raw"]
    raw_data_dirs = {
        dir_type: Experiment.get_data_directory(
            experiment_key={"experiment_name": experiment_name}, directory_type=dir_type, as_posix=False
        )
        for dir_type in directory_types
    }
    raw_data_dirs = {k: v for k, v in raw_data_dirs.items() if v}

    if not raw_data_dirs:
        raise ValueError(f"No raw data directory found for experiment: {experiment_name}")

    chunkdata = io_api.load(
        root=list(raw_data_dirs.values()),
        reader=io_reader.Chunk(pattern=device_name + "*", extension="csv"),
    )

    return chunkdata, raw_data_dirs


def _match_experiment_directory(experiment_name, path, directories):
    for k, v in directories.items():
        raw_data_dir = v
        if pathlib.Path(raw_data_dir) in list(path.parents):
            directory = (
                Experiment.Directory.proj("repository_name")
                & {"experiment_name": experiment_name, "directory_type": k}
            ).fetch1()
            repo_path = paths.get_repository_path(directory.pop("repository_name"))
            break
    else:
        raise FileNotFoundError(f"Unable to identify the directory" f" where this chunk is from: {path}")

    return raw_data_dir, directory, repo_path


def create_chunk_restriction(experiment_name, start_time, end_time):
    """
    Create a time restriction string for the chunks between the specified "start" and "end" times
    """
    start_restriction = f'"{start_time}" BETWEEN chunk_start AND chunk_end'
    end_restriction = f'"{end_time}" BETWEEN chunk_start AND chunk_end'
    start_query = Chunk & {"experiment_name": experiment_name} & start_restriction
    end_query = Chunk & {"experiment_name": experiment_name} & end_restriction
    if not (start_query and end_query):
        raise ValueError(f"No Chunk found between {start_time} and {end_time}")
    time_restriction = (
        f'chunk_start >= "{min(start_query.fetch("chunk_start"))}"'
        f' AND chunk_start < "{max(end_query.fetch("chunk_end"))}"'
    )
    return time_restriction
