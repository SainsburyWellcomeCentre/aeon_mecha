import datajoint as dj
import pandas as pd
import numpy as np

from aeon.io import reader as io_reader

from . import acquisition
from . import get_schema_name


schema = dj.schema(get_schema_name('device_stream'))


@schema
class StreamType(dj.Lookup):
    """
    Catalog of all steam types for the different device types used across Project Aeon
    One StreamType corresponds to one reader class in `aeon.io.reader`
    The combination of `stream_reader` and `stream_reader_kwargs` should fully specify
    the data loading routine for a particular device, using the `aeon.io.api`
    """
    definition = """  # Catalog of all stream types used across Project Aeon
    stream_type: varchar(16)
    ---
    stream_reader: varchar(256)     # name of the reader class found in `aeon.io.reader`
    stream_reader_kwargs: longblob  # keyword arguments to initialize the reader class
    stream_description='': varchar(256)
    """


@schema
class DeviceType(dj.Lookup):
    """
    Catalog of all device types used across Project Aeon
    """
    definition = """  # Catalog of all device types used across Project Aeon
    device_type: varchar(16)
    ---
    device_description='': varchar(256)
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


# ---- HELPER ----

def generate_device_table(device_type):
    schema = dj.schema()
    device_title = device_type.replace('_', ' ').title().replace(' ', '')

    @schema
    class ExperimentDevice(dj.Manual):
        definition = f"""
        # {device_title} placement and operation for a particular time period, at a certain location, for a given experiment
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

    ExperimentDevice.__name__ = device_title

    # DeviceDataStream table(s)

    for stream_detail in (StreamType & (DeviceType.Stream & {'device_type': device_type})).fetch(as_dict=True):
        stream_type = stream_detail['stream_type']
        stream_title = stream_type.replace('_', ' ').title().replace(' ', '')

        reader = getattr(io_reader, stream_detail['stream_reader'])(**stream_detail['stream_reader_kwargs'])

        table_definition = f"""  # Raw per-chunk {stream_title} data stream from {device_title}
            -> acquisition.Chunk
            -> ExperimentDevice
            ---
            timestamps: longblob   # (datetime) timestamps of {stream_type} data
            """

        for col in reader.columns:
            table_definition += f'{col}: longblob\n\t\t\t'

        @schema
        class DeviceDataStream(dj.Imported):
            definition = table_definition

        DeviceDataStream.__name__ = f'{device_title}{stream_title}Stream'

    return schema
