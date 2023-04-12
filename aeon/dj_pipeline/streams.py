import inspect
import re
from collections import defaultdict
from functools import cached_property
from pathlib import Path

import datajoint as dj
import pandas as pd
from dotmap import DotMap

import aeon
from aeon.dj_pipeline import acquisition, dict_to_uuid, get_schema_name
from aeon.io import api as io_api

logger = dj.logger


# schema_name = f'u_{dj.config["database.user"]}_streams'  # for testing
schema_name = get_schema_name("streams")
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
    stream_type          : varchar(20)
    ---
    stream_reader        : varchar(256)     # name of the reader class found in `aeon_mecha` package (e.g. aeon.io.reader.Video)
    stream_reader_kwargs : longblob  # keyword arguments to instantiate the reader class
    stream_description='': varchar(256)
    stream_hash          : uuid    # hash of dict(stream_reader_kwargs, stream_reader=stream_reader)
    unique index (stream_hash)
    """


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


@schema
class Device(dj.Lookup):
    definition = """  # Physical devices, of a particular type, identified by unique serial number
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


# region Helper functions for creating device tables.


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
            {device_type}_remove_time: datetime(6)  # time of the {device_type} being removed
            """

    ExperimentDevice.__name__ = f"{device_title}"

    return ExperimentDevice


def get_device_stream_template(device_type, stream_type):
    """Returns table class template for DeviceDataStream"""

    ExperimentDevice = get_device_template(device_type)
    exp_device_table_name = f"{device_type}"

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
        -> {device_type}
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
    """Class for managing device tables"""

    def __init__(self, context=None):

        if context is None:
            self.context = inspect.currentframe().f_back.f_locals
        else:
            self.context = context

        self._schema = dj.schema(context=self.context)
        self._device_stream_tables = []
        self._device_stream_map = defaultdict(
            list
        )  # dictionary for showing hierarchical relationship between device type and stream type

    def _add_device_stream_tables(self):
        for device_type in self.device_tables:
            for stream_type in (
                StreamType & (DeviceType.Stream & {"device_type": device_type})
            ).fetch("stream_type"):

                table_name = f"{device_type}{stream_type}"
                if table_name not in self._device_stream_tables:
                    self._device_stream_tables.append(table_name)

                self._device_stream_map[device_type].append(stream_type)

    @cached_property
    def device_tables(self) -> list:
        """
        Name of the device tables to be created
        """

        return list(DeviceType.fetch("device_type"))

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

        for device_type in self.device_tables:

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


# endregion


if __name__ == "__main__":

    # Create device & device stream tables
    tbmg = DeviceTableManager(context=inspect.currentframe().f_back.f_locals)
    tbmg.create_device_tables()
    tbmg.create_device_stream_tables()
