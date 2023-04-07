import inspect
import re
from collections import defaultdict, namedtuple
from functools import cached_property
from pathlib import Path

import datajoint as dj
import pandas as pd
from dotmap import DotMap

import aeon
import aeon.schema.core as stream
import aeon.schema.foraging as foraging
import aeon.schema.octagon as octagon
from aeon.dj_pipeline import acquisition, dict_to_uuid, get_schema_name
from aeon.io import api as io_api

logger = dj.logger


schema_name = f'u_{dj.config["database.user"]}_streams'  # for testing
# schema_name = get_schema_name("streams")
schema = dj.schema(schema_name)


__all__ = [
    "StreamType",
    "DeviceType",
    "Device",
]


# Read from this list of device configurations
# (device_type, description, streams)
DEVICE_CONFIGS = [
    (
        "Camera",
        "Camera device",
        (stream.video, stream.position, foraging.region),
    ),
    ("Metadata", "Metadata", (stream.metadata,)),
    (
        "ExperimentalMetadata",
        "ExperimentalMetadata",
        (stream.environment, stream.messageLog),
    ),
    (
        "NestScale",
        "Weight scale at nest",
        (foraging.weight,),
    ),
    (
        "FoodPatch",
        "Food patch",
        (foraging.patch,),
    ),
    (
        "Photodiode",
        "Photodiode",
        (octagon.photodiode,),
    ),
    (
        "OSC",
        "OSC",
        (octagon.OSC,),
    ),
    (
        "TaskLogic",
        "TaskLogic",
        (octagon.TaskLogic,),
    ),
    (
        "Wall",
        "Wall",
        (octagon.Wall,),
    ),
]


@schema
class StreamType(dj.Lookup):
    """
    Catalog of all steam types for the different device types used across Project Aeon
    One StreamType corresponds to one reader class in `aeon.io.reader`
    The combination of `stream_reader` and `stream_reader_kwargs` should fully specify
    the data loading routine for a particular device, using the `aeon.io.utils`
    """

    definition = """  # Catalog of all stream types used across Project Aeon
    stream_type          : varchar(20)
    ---
    stream_reader        : varchar(256)     # name of the reader class found in `aeon_mecha` package (e.g. aeon.io.reader.Video)
    stream_reader_kwargs : longblob  # keyword arguments to instantiate the reader class
    stream_description='': varchar(256)
    stream_hash          : uuid    # hash of dict(stream_reader_kwargs, stream_reader=stream_reader)
    unique index (stream_hash)
    """

    @classmethod
    def insert_streams(cls, schema: DotMap):

        stream_entries = get_stream_entries(schema)

        for entry in stream_entries:
            q_param = cls & {"stream_hash": entry["stream_hash"]}
            if q_param:  # If the specified stream type already exists
                pname = q_param.fetch1("stream_type")
                if pname != entry["stream_type"]:
                    # If the existed stream type does not have the same name:
                    #   human error, trying to add the same content with different name
                    raise dj.DataJointError(
                        f"The specified stream type already exists - name: {pname}"
                    )

        cls.insert(stream_entries, skip_duplicates=True)


@schema
class DeviceType(dj.Lookup):
    """
    Catalog of all device types used across Project Aeon
    """

    definition = """  # Catalog of all device types used across Project Aeon
    device_type:             varchar(36)
    ---
    device_description='':   varchar(256)
    """

    class Stream(dj.Part):
        definition = """  # Data stream(s) associated with a particular device type
        -> master
        -> StreamType
        """

    @classmethod
    def insert_devices(cls, device_configs: list[namedtuple] = []):
        if not device_configs:
            device_configs = get_device_configs()
        for device in device_configs:
            stream_entries = StreamType.get_stream_entries(device.streams)
            with cls.connection.transaction:
                cls.insert1(
                    {
                        "device_type": device.type,
                        "device_description": device.desc,
                    },
                    skip_duplicates=True,
                )
                cls.Stream.insert(
                    [
                        {
                            "device_type": device.type,
                            "stream_type": e["stream_type"],
                        }
                        for e in stream_entries
                    ],
                    skip_duplicates=True,
                )


@schema
class Device(dj.Lookup):
    definition = """  # Physical devices, of a particular type, identified by unique serial number
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


## --------- Helper functions & classes --------- ##


def get_device_configs(device_configs=DEVICE_CONFIGS) -> list[namedtuple]:
    """Returns a list of device configurations from DEVICE_CONFIGS"""

    device = namedtuple("device", "type desc streams")
    return [device._make(c) for c in device_configs]


def get_device_template(device_type):
    """Returns table class template for ExperimentDevice"""
    device_title = device_type
    device_type = dj.utils.from_camel_case(device_type)

    class ExperimentDevice(dj.Manual):
        definition = f"""
        # {device_title} placement and operation for a particular time period, at a certain location, for a given experiment (auto-generated with aeon_mecha-{aeon.__version__})
        -> acquisition.Experiment
        -> Device
        {device_type}_install_time: datetime(6)   # time of the {device_type} placed and started operation at this position
        ---
        {device_type}_name: varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # metadata/attributes (e.g. FPS, config, calibration, etc.) associated with this experimental device
            -> master
            attribute_name    : varchar(32)
            ---
            attribute_value='': varchar(2000)
            """

        class RemovalTime(dj.Part):
            definition = f"""
            -> master
            ---
            {device_type}_remove_time: datetime(6)  # time of the camera being removed from this position
            """

    ExperimentDevice.__name__ = f"Experiment{device_title}"

    return ExperimentDevice


def get_device_stream_template(device_type, stream_type):
    """Returns table class template for DeviceDataStream"""

    ExperimentDevice = get_device_template(device_type)
    exp_device_table_name = f"Experiment{device_type}"

    # DeviceDataStream table(s)
    stream_detail = (
        StreamType
        & (DeviceType.Stream & {"device_type": device_type, "stream_type": stream_type})
    ).fetch1()

    for i, n in enumerate(stream_detail["stream_reader"].split(".")):
        if i == 0:
            reader = aeon
        else:
            reader = getattr(reader, n)

    stream = reader(**stream_detail["stream_reader_kwargs"])

    table_definition = f"""  # Raw per-chunk {stream_type} data stream from {device_type} (auto-generated with aeon_mecha-{aeon.__version__})
        -> Experiment{device_type}
        -> acquisition.Chunk
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        timestamps: longblob   # (datetime) timestamps of {stream_type} data
        """

    for col in stream.columns:
        if col.startswith("_"):
            continue
        table_definition += f"{col}: longblob\n\t\t\t"

    class DeviceDataStream(dj.Imported):
        definition = table_definition
        _stream_reader = reader
        _stream_detail = stream_detail

        @property
        def key_source(self):
            f"""
            Only the combination of Chunk and {exp_device_table_name} with overlapping time
            +  Chunk(s) that started after {exp_device_table_name} install time and ended before {exp_device_table_name} remove time
            +  Chunk(s) that started after {exp_device_table_name} install time for {exp_device_table_name} that are not yet removed
            """
            return (
                acquisition.Chunk
                * ExperimentDevice.join(ExperimentDevice.RemovalTime, left=True)
                & f"chunk_start >= {device_type}_install_time"
                & f'chunk_start < IFNULL({device_type}_remove_time, "2200-01-01")'
            )

        def make(self, key):
            chunk_start, chunk_end, dir_type = (acquisition.Chunk & key).fetch1(
                "chunk_start", "chunk_end", "directory_type"
            )
            raw_data_dir = acquisition.Experiment.get_data_directory(
                key, directory_type=dir_type
            )

            device_name = (ExperimentDevice & key).fetch1(f"{device_type}_name")

            stream = self._stream_reader(
                **{
                    k: v.format(**{k: device_name}) if k == "pattern" else v
                    for k, v in self._stream_detail["stream_reader_kwargs"].items()
                }
            )

            stream_data = io_api.load(
                root=raw_data_dir.as_posix(),
                reader=stream,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
                {
                    **key,
                    "sample_count": len(stream_data),
                    "timestamps": stream_data.index.values,
                    **{
                        c: stream_data[c].values
                        for c in stream.columns
                        if not c.startswith("_")
                    },
                }
            )

    DeviceDataStream.__name__ = f"{device_type}{stream_type}"

    return DeviceDataStream


class DeviceTableManager:
    def __init__(self, context=None):

        if context is None:
            self.context = inspect.currentframe().f_back.f_locals
        else:
            self.context = context

        self._schema = dj.schema(context=self.context)
        self._device_tables = []
        self._device_stream_tables = []
        self._device_types = DeviceType.fetch("device_type")
        self._device_stream_map = defaultdict(
            list
        )  # dictionary for showing hierarchical relationship between device type and stream type

    def _add_device_tables(self):
        for device_type in self._device_types:
            table_name = f"Experiment{device_type}"
            if table_name not in self._device_tables:
                self._device_tables.append(table_name)

    def _add_device_stream_tables(self):
        for device_type in self._device_types:
            for stream_type in (
                StreamType & (DeviceType.Stream & {"device_type": device_type})
            ).fetch("stream_type"):

                table_name = f"{device_type}{stream_type}"
                if table_name not in self._device_stream_tables:
                    self._device_stream_tables.append(table_name)

                self._device_stream_map[device_type].append(stream_type)

    @property
    def device_types(self):
        return self._device_types

    @cached_property
    def device_tables(self) -> list:
        """
        Name of the device tables to be created
        """

        self._add_device_tables()
        return self._device_tables

    @cached_property
    def device_stream_tables(self) -> list:
        """
        Name of the device stream tables to be created
        """
        self._add_device_stream_tables()
        return self._device_stream_tables

    @cached_property
    def device_stream_map(self) -> dict:
        self._add_device_stream_tables()
        return self._device_stream_map

    def create_device_tables(self):

        for device_table in self.device_tables:

            device_type = re.sub(r"\bExperiment", "", device_table)

            table_class = get_device_template(device_type)

            self.context[table_class.__name__] = table_class
            self._schema(table_class, context=self.context)

        self._schema.activate(schema_name)

    def create_device_stream_tables(self):

        for device_type in self.device_stream_map:

            for stream_type in self.device_stream_map[device_type]:

                table_class = get_device_stream_template(device_type, stream_type)

                self.context[table_class.__name__] = table_class
                self._schema(table_class, context=self.context)

        self._schema.activate(schema_name)


def get_device_info(schema: DotMap) -> dict[dict]:
    """
    Read from the above DotMap object and returns a device dictionary as the following.

    Args:
        schema (DotMap): DotMap object (e.g., exp02, octagon01)

    Returns:
        device_info (dict[dict]): A dictionary of device information

        e.g.   {'CameraTop':
                    {'stream_type': ['Video', 'Position', 'Region'],
                        'reader': [
                                    aeon.io.reader.Video,
                                    aeon.io.reader.Position,
                                    aeon.schema.foraging._RegionReader
                                ],
                        'pattern': ['{pattern}', '{pattern}_200', '{pattern}_201']
                    }
                }
    """
    import json
    from collections import defaultdict

    from aeon.dj_pipeline import dict_to_uuid

    schema_json = json.dumps(schema, default=lambda o: o.__dict__, indent=4)
    schema_dict = json.loads(schema_json)

    device_info = {}

    for device_name in schema:
        if not device_name.startswith("_"):
            device_info[device_name] = defaultdict(list)
            if isinstance(schema[device_name], DotMap):
                for stream_type in schema[device_name].keys():
                    if schema[device_name][stream_type].__class__.__module__ in [
                        "aeon.io.reader",
                        "aeon.schema.foraging",
                        "aeon.schema.octagon",
                    ]:
                        device_info[device_name]["stream_type"].append(stream_type)
                        device_info[device_name]["reader"].append(
                            schema[device_name][stream_type].__class__
                        )
            else:
                stream_type = schema[device_name].__class__.__name__
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["reader"].append(schema[device_name].__class__)

    """Add a kwargs such as pattern, columns, extension, dtype and hash
    e.g., {'pattern': '{pattern}_SubjectState',
            'columns': ['id', 'weight', 'event'],
            'extension': 'csv',
            'dtype': None}"""
    for device_name in device_info:
        if pattern := schema_dict[device_name].get("pattern"):
            schema_dict[device_name]["pattern"] = pattern.replace(
                device_name, "{pattern}"
            )

            # Add stream_reader_kwargs
            kwargs = schema_dict[device_name]
            device_info[device_name]["stream_reader_kwargs"].append(kwargs)
            stream_reader = device_info[device_name]["stream_reader"]
            # Add hash
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({**kwargs, "stream_reader": stream_reader})
            )

        else:
            for stream_type in device_info[device_name]["stream_type"]:
                pattern = schema_dict[device_name][stream_type]["pattern"]
                schema_dict[device_name][stream_type]["pattern"] = pattern.replace(
                    device_name, "{pattern}"
                )
                # Add stream_reader_kwargs
                kwargs = schema_dict[device_name][stream_type]
                device_info[device_name]["stream_reader_kwargs"].append(kwargs)
                stream_ind = device_info[device_name]["stream_type"].index(stream_type)
                stream_reader = device_info[device_name]["stream_reader"][stream_ind]
                # Add hash
                device_info[device_name]["stream_hash"].append(
                    dict_to_uuid({**kwargs, "stream_reader": stream_reader})
                )

    return device_info


def add_device_type(schema: DotMap, metadata_yml_filepath: Path):
    """Update device_info with device_type based on metadata.yml.

    Args:
        schema (DotMap): DotMap object (e.g., exp02)
        metadata_yml_filepath (Path): Path to metadata.yml.

    Returns:
        device_info (dict): Updated device_info.
    """
    from aeon.io import api

    meta_data = (
        api.load(
            str(metadata_yml_filepath.parent),
            schema.Metadata,
        )
        .reset_index()
        .to_dict("records")[0]["metadata"]
    )

    # Get device_type_mapper based on metadata.yml {'CameraTop': 'VideoSource', 'Patch1': 'Patch'}
    device_type_mapper = {}
    for item in meta_data.Devices:
        device_type_mapper[item.Name] = item.Type

    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name, None),
            **device_info[device_name],
        }
        for device_name in device_info
    }

    return device_info


def get_stream_entries(schema: DotMap) -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for a given device,

    Args:
        schema (DotMap): DotMap object (e.g., exp02, octagon01)

    Returns:
    stream_info (list[dict]): list of dictionaries containing the stream entries for a given device,

        e.g. {'stream_type': 'EnvironmentState',
        'stream_reader': aeon.io.reader.Csv,
        'stream_reader_kwargs': {'pattern': '{pattern}_EnvironmentState',
        'columns': ['state'],
        'extension': 'csv',
        'dtype': None}
    """
    device_info = get_device_info(schema)
    return [
        {
            "stream_type": stream_type,
            "stream_reader": stream_reader,
            "stream_reader_kwargs": stream_reader_kwargs,
            "stream_hash": stream_hash,
        }
        for stream_info in device_info.values()
        for stream_type, stream_reader, stream_reader_kwargs, stream_hash in zip(
            stream_info["stream_type"],
            stream_info["stream_reader"],
            stream_info["stream_reader_kwargs"],
            stream_info["stream_hash"],
        )
    ]
