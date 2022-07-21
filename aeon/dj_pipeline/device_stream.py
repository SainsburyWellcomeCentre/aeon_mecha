import datajoint as dj
import pandas as pd
import numpy as np
import inspect
import re

import aeon
from aeon.io import api as io_api

from . import acquisition
from . import get_schema_name

logger = dj.logger

schema_name = get_schema_name("device_stream")
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
    stream_type: varchar(16)
    ---
    stream_reader: varchar(256)     # name of the reader class found in `aeon_mecha` package (e.g. aeon.io.reader.Video)
    stream_reader_kwargs: longblob  # keyword arguments to instantiate the reader class
    stream_description='': varchar(256)
    """

    contents = [
        ("Video", "aeon.io.reader.Video", {"pattern": "{}"}, "Video frame metadata"),
        (
            "Position",
            "aeon.io.reader.Position",
            {"pattern": "{}_200"},
            "Position tracking data for the specified camera.",
        ),
        (
            "Encoder",
            "aeon.io.reader.Encoder",
            {"pattern": "{}_90"},
            "Wheel magnetic encoder data",
        ),
        (
            "EnvironmentState",
            "aeon.io.reader.Csv",
            {"pattern": "{}_EnvironmentState", "columns": ["state"]},
            "Environment state log",
        ),
        (
            "SubjectState",
            "aeon.io.reader.Subject",
            {"pattern": "{}_SubjectState"},
            "Subject state log",
        ),
        (
            "MessageLog",
            "aeon.io.reader.Log",
            {"pattern": "{}_MessageLog"},
            "Message log data",
        ),
        (
            "Metadata",
            "aeon.io.reader.Metadata",
            {"pattern": "{}"},
            "Metadata for acquisition epochs",
        ),
        (
            "MetadataExp01",
            "aeon.io.reader.Metadata",
            {"pattern": "{}_2", "columns": ["id", "weight", "event"]},
            "Session metadata for Experiment 0.1",
        ),
        (
            "Region",
            "aeon.schema.foraging._RegionReader",
            {"pattern": "{}_201"},
            "Region tracking data for the specified camera",
        ),
        (
            "DepletionState",
            "aeon.schema.foraging._PatchState",
            {"pattern": "{}_State"},
            "State of the linear depletion function for foraging patches",
        ),
        (
            "BeamBreak",
            "aeon.io.reader.BitmaskEvent",
            {"pattern": "{}_32", "value": 0x22, "tag": "PelletDetected"},
            "Beam break events for pellet detection",
        ),
        (
            "DeliverPellet",
            "aeon.io.reader.BitmaskEvent",
            {"pattern": "{}_35", "value": 0x80, "tag": "TriggerPellet"},
            "Pellet delivery commands",
        ),
        (
            "WeightRaw",
            "aeon.schema.foraging._Weight",
            {"pattern": "{}_200"},
            "Raw weight measurement for a specific nest",
        ),
        (
            "WeightFiltered",
            "aeon.schema.foraging._Weight",
            {"pattern": "{}_202"},
            "Filtered weight measurement for a specific nest",
        ),
        (
            "WeightSubject",
            "aeon.schema.foraging._Weight",
            {"pattern": "{}_204"},
            "Subject weight measurement for a specific nest",
        ),
    ]


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

    @classmethod
    def _insert_contents(cls):
        devices_config = [
            ("Camera", "Camera device", ("Video", "Position", "Region")),
            ("Metadata", "Metadata", ("Metadata",)),
            (
                "ExperimentalMetadata",
                "ExperimentalMetadata",
                ("EnvironmentState", "SubjectState", "MessageLog"),
            ),
            (
                "Nest Scale",
                "Weight scale at nest",
                ("WeightRaw", "WeightFiltered", "WeightSubject"),
            ),
            (
                "Food Patch",
                "Food patch",
                ("DepletionState", "Encoder", "BeamBreak", "DeliverPellet"),
            ),
        ]
        for device_type, device_desc, device_streams in devices_config:
            if cls & {"device_type": device_type}:
                continue
            with cls.connection.transaction:
                cls.insert1((device_type, device_desc))
                cls.Stream.insert(
                    [
                        {"device_type": device_type, "stream_type": stream_type}
                        for stream_type in device_streams
                    ]
                )


@schema
class Device(dj.Lookup):
    definition = """  # Physical devices, of a particular type, identified by unique serial number
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


DeviceType._insert_contents()


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
        {device_type}_description: varchar(36)
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

        logger.info(f"Creating stream table: {stream_title}")

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
            timestamps: longblob   # (datetime) timestamps of {stream_type} data
            """

        for col in stream.columns:
            if col.startswith("_"):
                continue
            table_definition += f"{col}: longblob\n\t\t\t"

        @_schema
        class DeviceDataStream(dj.Imported):
            definition = table_definition

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

                device_description = (ExperimentDevice & key).fetch1(
                    f"{device_type}_description"
                )

                stream = reader(
                    **{
                        k: v.format(device_description) if k == "pattern" else v
                        for k, v in stream_detail["stream_reader_kwargs"].items()
                    }
                )

                stream_data = io_api.load(
                    root=raw_data_dir.as_posix(),
                    reader=stream,
                    start=pd.Timestamp(chunk_start),
                    end=pd.Timestamp(chunk_end),
                )

                if not len(stream_data):
                    raise ValueError(f"No stream data found for {key}")

                self.insert1(
                    {
                        **key,
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


for device_type in DeviceType.fetch("device_type"):
    generate_device_table(device_type)
