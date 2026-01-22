"""Module for stream-related tables in the analysis schema.

This module provides utilities for auto-generating DataJoint stream tables
based on device types and their associated streams. It uses the Pydantic-based
approach where stream readers are defined via @data_reader methods on Device classes.
"""

import importlib
import inspect
import re
from pathlib import Path

import datajoint as dj
import pandas as pd

import aeon
from aeon.dj_pipeline import acquisition, get_schema_name

logger = dj.logger


# schema_name = f'u_{dj.config["database.user"]}_streams'  # for testing
schema_name = get_schema_name("streams")

_STREAMS_MODULE_FILE = Path(__file__).parent.parent / "streams.py"


class StreamType(dj.Lookup):
    """Catalog of unique stream types used across Project Aeon.

    Each entry represents a unique combination of stream_type name and reader class.
    The stream reader is resolved at runtime via the Pydantic Device class hierarchy,
    using @data_reader decorated methods. The stream_hash uniquely identifies each
    (stream_type, stream_reader) combination to handle cases where different experiments
    define the same stream_type name with different underlying readers.
    """

    definition = """ # Catalog of unique stream types used across Project Aeon
    stream_hash: uuid  # hash(stream_type, stream_reader) - unique identifier
    ---
    stream_type: varchar(36)  # stream type name, e.g., "Video", "BeamBreak"
    stream_reader: varchar(256)  # reader class path for documentation, e.g., "swc.aeon.io.reader.Video"
    stream_description='': varchar(256)
    unique index(stream_type, stream_reader)
    """


class DeviceType(dj.Lookup):
    """Catalog of all device types used across Project Aeon."""

    definition = """  # Catalog of all device types used across Project Aeon
    device_type:             varchar(36)
    ---
    device_description='':   varchar(256)
    """

    class Stream(dj.Part):
        """Links device types to their associated stream types.

        Each entry specifies which StreamType (identified by stream_hash) is
        associated with a given device type. This allows the same device type
        to have multiple streams, and handles cases where different experiments
        use the same stream_type name with different reader implementations.
        """

        definition = """  # Data stream(s) associated with a particular device type
        -> master
        -> StreamType
        """


class Device(dj.Lookup):
    definition = """  # Physical devices, of a particular type, identified by unique serial number
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


# region Helper functions for creating device tables.


def get_device_template(device_type: str):
    """Returns table class template for ExperimentDevice."""
    device_title = device_type
    device_type = dj.utils.from_camel_case(device_type)

    class ExperimentDevice(dj.Manual):
        definition = """ # {device_title} operation for time, location, experiment (v-{aeon.__version__})
        -> acquisition.Experiment
        -> Device
        {device_type}_install_time : datetime(6)  # {device_type} time of placement and start operation
        ---
        {device_type}_name         : varchar(36)
        """

        class Attribute(dj.Part):
            definition = """  # Metadata (e.g. FPS, config, calibration) for this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : longblob
            """

        class RemovalTime(dj.Part):
            definition = """
            -> master
            ---
            {device_type}_removal_time: datetime(6)  # time of the {device_type} being removed
            """

    ExperimentDevice.__name__ = f"{device_title}"

    return ExperimentDevice


def get_device_stream_template(device_type: str, stream_type: str, streams_module):
    """Returns table class template for DeviceDataStream.

    Uses the Pydantic-based approach to get stream reader columns by reconstructing
    the Rig from stored metadata and accessing the @data_reader decorated method.
    """
    from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

    ExperimentDevice = getattr(streams_module, device_type)
    device_type_snake = dj.utils.from_camel_case(device_type)

    # Get stream_reader path for Pose check
    stream_detail = (
        streams_module.StreamType
        & (streams_module.DeviceType.Stream & {"device_type": device_type, "stream_type": stream_type})
    ).fetch1()

    # Skip Pose reader - automatic generation not supported
    if "Pose" in stream_detail["stream_reader"]:
        logger.warning("Automatic generation of stream table for Pose reader is not supported. Skipping...")
        return None, None

    # Get columns by finding an experiment with this device type and reconstructing the reader
    # Find an experiment entry that has this device installed
    experiment_device_query = ExperimentDevice()

    if not experiment_device_query:
        logger.warning(f"No experiments found with device type {device_type}. Skipping stream table.")
        return None, None

    sample_entry = experiment_device_query.fetch(as_dict=True, limit=1)[0]
    experiment_name = sample_entry["experiment_name"]
    device_name = sample_entry[f"{device_type_snake}_name"]
    install_time = sample_entry[f"{device_type_snake}_install_time"]

    # Get the epoch that contains this device installation
    epoch_query = (
        acquisition.Epoch
        & {"experiment_name": experiment_name}
        & f"epoch_start <= '{install_time}'"
    )
    if not epoch_query:
        logger.warning(f"No epoch found for device {device_name} installation. Skipping stream table.")
        return None, None

    epoch_start = epoch_query.fetch("epoch_start", order_by="epoch_start DESC", limit=1)[0]

    # Get the stream reader using the Pydantic approach
    try:
        stream_reader = get_stream_reader_for_epoch(
            experiment_name, device_name, stream_type, epoch_start
        )
    except Exception as e:
        logger.warning(f"Could not get stream reader for {device_type}.{stream_type}: {e}")
        return None, None

    table_definition = f""" # Raw per-chunk {stream_type} from {device_type} (v-{aeon.__version__})
    -> {device_type}
    -> acquisition.Chunk
    ---
    sample_count: int      # number of data points acquired from this stream for a given chunk
    timestamps: longblob   # (datetime) timestamps of {stream_type} data
    """

    for col in stream_reader.columns:
        if col.startswith("_"):
            continue
        new_col = re.sub(r"\([^)]*\)", "", col)
        table_definition += f"{new_col}: longblob\n    "

    class DeviceDataStream(dj.Imported):
        definition = table_definition

        @property
        def key_source(self):
            """Only the combination of Chunk and {device_type} with overlapping time.

            + Chunk(s) started after {device_type} install time & ended before {device_type} remove time
            + Chunk(s) started after {device_type} install time for {device_type} and not yet removed
            """
            return (
                acquisition.Chunk * ExperimentDevice.join(ExperimentDevice.RemovalTime, left=True)
                & "chunk_start >= {device_type_name}_install_time"
                & 'chunk_start < IFNULL({device_type_name}_removal_time,"2200-01-01")'
            )

        def make(self, key):
            """Load and insert the data for the DeviceDataStream table."""
            from swc.aeon.io import api as io_api

            from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

            chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
            data_dirs = acquisition.Experiment.get_data_directories(key)

            device_name = (ExperimentDevice & key).fetch1("{device_type_name}_name")

            # Get stream reader using Pydantic approach (reconstructs Rig from stored metadata)
            stream_reader = get_stream_reader_for_epoch(
                key["experiment_name"], device_name, "{stream_type}", key["epoch_start"]
            )

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            self.insert1(
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

    DeviceDataStream.__name__ = f"{device_type}{stream_type}"

    return DeviceDataStream, table_definition


# endregion


def main(create_tables=True):
    """Main function to create and update stream-related tables in the analysis schema."""
    if not _STREAMS_MODULE_FILE.exists():
        with open(_STREAMS_MODULE_FILE, "w") as f:
            imports_str = (
                "#----                     DO NOT MODIFY                ----\n"
                "#---- THIS FILE IS AUTO-GENERATED BY `streams_maker.py` ----\n\n"
                "import re\n"
                "import datajoint as dj\n"
                "import pandas as pd\n"
                "from uuid import UUID\n\n"
                "import aeon\n"
                "from aeon.dj_pipeline import acquisition, get_schema_name\n"
                "from swc.aeon.io import api as io_api\n\n"
                'schema = dj.Schema(get_schema_name("streams"))\n\n\n'
            )
            f.write(imports_str)
            for table_class in (StreamType, DeviceType, Device):
                device_table_def = inspect.getsource(table_class).lstrip()
                full_def = "@schema \n" + device_table_def + "\n\n"
                f.write(full_def)

    streams = importlib.import_module("aeon.dj_pipeline.streams")

    if create_tables:
        # Create DeviceType tables.
        for device_info in streams.DeviceType.fetch(as_dict=True):
            if hasattr(streams, device_info["device_type"]):
                continue

            table_class = get_device_template(device_info["device_type"])
            streams.__dict__[table_class.__name__] = table_class

            device_table_def = inspect.getsource(table_class).lstrip()
            replacements = {
                "ExperimentDevice": device_info["device_type"],
                "{device_title}": dj.utils.from_camel_case(device_info["device_type"]),
                "{device_type}": dj.utils.from_camel_case(device_info["device_type"]),
                "{aeon.__version__}": aeon.__version__,
            }
            for old, new in replacements.items():
                device_table_def = device_table_def.replace(old, new)
            full_def = "@schema \n" + device_table_def + "\n\n"
            with open(_STREAMS_MODULE_FILE) as f:
                existing_content = f.read()

            if full_def in existing_content:
                continue

            with open(_STREAMS_MODULE_FILE, "a") as f:
                f.write(full_def)

        # Create DeviceDataStream tables.
        for device_info in streams.DeviceType.Stream.fetch(as_dict=True):
            device_type = device_info["device_type"]
            stream_type = device_info["stream_type"]
            table_name = f"{device_type}{stream_type}"

            if hasattr(streams, table_name):
                continue

            table_class, table_definition = get_device_stream_template(
                device_type, stream_type, streams_module=streams
            )

            if table_class is None:
                continue

            device_stream_table_def = inspect.getsource(table_class).lstrip()

            # Replace the definition
            device_type_name = dj.utils.from_camel_case(device_type)
            replacements = {
                "DeviceDataStream": f"{device_type}{stream_type}",
                "ExperimentDevice": device_type,
                "{device_type_name}": device_type_name,
                "{device_type}": device_type,
                "{stream_type}": stream_type,
                "{aeon.__version__}": aeon.__version__,
                "table_definition": f'"""{table_definition}"""',
            }
            for old, new in replacements.items():
                device_stream_table_def = device_stream_table_def.replace(old, new)

            full_def = "@schema \n" + device_stream_table_def + "\n\n"

            with open(_STREAMS_MODULE_FILE) as f:
                existing_content = f.read()

            if full_def in existing_content:
                continue

            with open(_STREAMS_MODULE_FILE, "a") as f:
                f.write(full_def)

        importlib.reload(streams)

    return streams


try:
    streams = main()
except Exception as e:
    logger.debug(f"Could not initialize streams module: {e}")
    streams = None
