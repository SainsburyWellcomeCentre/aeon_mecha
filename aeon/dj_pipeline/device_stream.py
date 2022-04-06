import datajoint as dj
import pandas as pd
import numpy as np

from aeon.preprocess import api as aeon_api

from . import acquisition
from . import get_schema_name


schema = dj.schema(get_schema_name('device_stream'))


@schema
class DeviceType(dj.Lookup):
    """
    Catalog of all device types used across Project Aeon
    The combination of `device_loader` and `device_load_kwargs` should fully specify
    the data loading routine for a particular device, using the `preprocess/api.py`
    """
    definition = """  # Catalog of all device types used across Project Aeon
    device_type: varchar(16)
    ---
    device_description: varchar(256)
    device_loader: varchar(256)
    device_load_kwargs: longblob
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
    device_title = device_type.replace('_', ' ').title().replace(' ', '')

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

    class DeviceDataStream(dj.Imported):
        definition = f"""  # Raw per-chunk data stream from {device_title}
        -> acquisition.Chunk
        -> ExperimentDevice
        ---
        timestamps:        longblob   # (datetime) timestamps of {device_type} data
        """