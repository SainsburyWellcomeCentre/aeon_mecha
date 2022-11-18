import datajoint as dj
import pandas as pd
import numpy as np
import inspect
import re

import aeon
from aeon.io import api as io_api
import aeon.schema.core as stream
import aeon.schema.foraging as foraging
import aeon.schema.octagon as octagon

from . import acquisition, dict_to_uuid, get_schema_name

logger = dj.logger

# schema_name = get_schema_name("device_stream")
schema_name = (
    f'u_{dj.config["database.user"]}_device_stream'  # still experimental feature
)
schema = dj.schema(schema_name)


@schema
class StreamType(dj.Lookup):
    """
    Catalog of all steam types for the different device types used across Project Aeon
    One StreamType corresponds to one reader class in `aeon.io.reader`
    The combination of `stream_reader` and `stream_reader_kwargs` should fully specify
    the data loading routine for a particular device, using the `aeon.io.utils`
    """

    definition = """  # Catalog of all stream types used across Project Aeon
    stream_type: varchar(20)
    ---
    stream_reader: varchar(256)     # name of the reader class found in `aeon_mecha` package (e.g. aeon.io.reader.Video)
    stream_reader_kwargs: longblob  # keyword arguments to instantiate the reader class
    stream_description='': varchar(256)
    stream_hash: uuid  # hash of dict(stream_reader_kwargs, stream_reader=stream_reader)
    unique index (stream_hash)
    """

    @classmethod
    def insert_streams(cls, *streams):
        composite = {}
        pattern = "{pattern}"
        for stream_obj in streams:
            if inspect.isclass(stream_obj):
                for method in vars(stream_obj).values():
                    if isinstance(method, staticmethod):
                        composite.update(method.__func__(pattern))
            else:
                composite.update(stream_obj(pattern))

        stream_entries = []
        for stream_name, stream_reader in composite.items():
            if stream_name == pattern:
                stream_name = stream_reader.__class__.__name__
            entry = {
                "stream_type": stream_name,
                "stream_reader": f"{stream_reader.__module__}.{stream_reader.__class__.__name__}",
                "stream_reader_kwargs": {
                    k: v
                    for k, v in vars(stream_reader).items()
                    if k
                    in inspect.signature(stream_reader.__class__.__init__).parameters
                },
            }
            entry["stream_hash"] = dict_to_uuid(
                {
                    **entry["stream_reader_kwargs"],
                    "stream_reader": entry["stream_reader"],
                }
            )
            q_param = cls & {"stream_hash": entry["stream_hash"]}
            if q_param:  # If the specified stream type already exists
                pname = q_param.fetch1("stream_type")
                if pname != stream_name:
                    # If the existed stream type does not have the same name:
                    #   human error, trying to add the same content with different name
                    raise dj.DataJointError(
                        f"The specified stream type already exists - name: {pname}"
                    )

            stream_entries.append(entry)

        cls.insert(stream_entries, skip_duplicates=True)
        return stream_entries


@schema
class DeviceType(dj.Lookup):
    """
    Catalog of all device types used across Project Aeon
    """

    definition = """  # Catalog of all device types used across Project Aeon
    device_type: varchar(36)
    ---
    device_description='': varchar(256)
    """

    class Stream(dj.Part):
        definition = """  # Data stream(s) associated with a particular device type
        -> master
        -> StreamType
        """

    _devices_config = [
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
            "Nest Scale",
            "Weight scale at nest",
            (foraging.weight,),
        ),
        (
            "Food Patch",
            "Food patch",
            (foraging.patch,),
        ),
        (
            "Food Patch",
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

    @classmethod
    def insert_devices(cls):
        for device_type, device_desc, device_streams in cls._devices_config:
            stream_entries = StreamType.insert_streams(*device_streams)
            with cls.connection.transaction:
                cls.insert1((device_type, device_desc), skip_duplicates=True)
                cls.Stream.insert(
                    [
                        {
                            "device_type": device_type,
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


# ---- HELPER ----


def generate_device_table(device_type, context=None):
    if context is None:
        context = inspect.currentframe().f_back.f_locals

    _schema = dj.schema(context=context)

    device_type_key = {"device_type": device_type}
    device_title = _prettify(device_type)
    device_type = dj.utils.from_camel_case(device_title)

    @_schema
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

    exp_device_table_name = f"Experiment{device_title}"
    ExperimentDevice.__name__ = exp_device_table_name
    context[exp_device_table_name] = ExperimentDevice

    # DeviceDataStream table(s)
    for stream_detail in (StreamType & (DeviceType.Stream & device_type_key)).fetch(
        as_dict=True
    ):
        stream_type = stream_detail["stream_type"]
        stream_title = _prettify(stream_type)

        logger.info(f"Creating stream table: {device_title}{stream_title}")

        for i, n in enumerate(stream_detail["stream_reader"].split(".")):
            if i == 0:
                reader = aeon
            else:
                reader = getattr(reader, n)

        stream = reader(**stream_detail["stream_reader_kwargs"])

        table_definition = f"""  # Raw per-chunk {stream_title} data stream from {device_title} (auto-generated with aeon_mecha-{aeon.__version__})
            -> Experiment{device_title}
            -> acquisition.Chunk
            ---
            sample_count: int      # number of data points acquired from this stream for a given chunk
            timestamps: longblob   # (datetime) timestamps of {stream_type} data
            """

        for col in stream.columns:
            if col.startswith("_"):
                continue
            table_definition += f"{col}: longblob\n\t\t\t"

        @_schema
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

        stream_table_name = f"{device_title}{stream_title}"
        DeviceDataStream.__name__ = stream_table_name
        context[stream_table_name] = DeviceDataStream

    _schema.activate(schema_name)


def _prettify(s):
    s = re.sub(r"[A-Z]", lambda m: f"_{m.group(0)}", s)
    return s.replace("_", " ").title().replace(" ", "")


# ---- MAIN BLOCK ----


def main():
    DeviceType.insert_devices()

    context = inspect.currentframe().f_back.f_locals
    for device_type in DeviceType.fetch("device_type"):
        logger.info(f"Generating stream table(s) for: {device_type}")
        generate_device_table(device_type, context=context)


main()
