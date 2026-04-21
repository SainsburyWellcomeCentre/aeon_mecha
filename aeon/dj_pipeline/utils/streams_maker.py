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
    stream_reader_kwargs=null: <blob>  # JSON dict of reader constructor kwargs (value, tag, columns, etc.)
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


class DeviceName(dj.Lookup):
    definition = """  # Catalog of device instance names
    device_name: varchar(36)
    ---
    -> DeviceType
    """


class Device(dj.Lookup):
    definition = """  # Physical devices identified by serial number or port
    device_serial_number: varchar(12)
    ---
    -> DeviceType
    """


# region Helper functions for creating device tables.


def get_device_template(device_type: str):
    """Returns table class template for ExperimentDevice.

    The template uses DeviceName (device_name) as primary key instead of Device,
    making queries more intuitive (e.g., & {"device_name": "CameraTop"}).
    Device serial number is stored as an optional attribute for hardware tracking.
    """
    device_title = device_type
    device_type = dj.utils.from_camel_case(device_type)

    class ExperimentDevice(dj.Manual):
        definition = """ # {device_title} operation for time, location, experiment (v-{aeon.__version__})
        -> acquisition.Experiment
        -> DeviceName
        {device_type}_install_time : datetime(6)  # {device_type} time of placement and start operation
        ---
        device_serial_number=null : varchar(12)  # Optional: physical device serial/port
        """

        class Attribute(dj.Part):
            definition = """  # Metadata (e.g. FPS, config, calibration) for this experimental device
            -> master
            attribute_name          : varchar(32)
            ---
            attribute_value=null    : <blob>
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

    Extracts columns on-demand by importing the reader class directly from the
    stream_reader path stored in StreamType. Uses stream_reader_kwargs for readers
    that require additional constructor arguments (e.g., BitmaskEvent, Harp).
    """
    ExperimentDevice = getattr(streams_module, device_type)

    # Get stream_reader path and kwargs from catalog
    # Join DeviceType.Stream (has device_type, stream_hash) with StreamType (has stream_type)
    stream_detail = (
        streams_module.StreamType & {"stream_type": stream_type}
        & (streams_module.DeviceType.Stream & {"device_type": device_type})
    ).fetch1()

    stream_reader_path = stream_detail["stream_reader"]
    stream_reader_kwargs = stream_detail.get("stream_reader_kwargs") or {}

    # Skip Pose reader - automatic generation not supported
    if "Pose" in stream_reader_path:
        logger.warning("Automatic generation of stream table for Pose reader is not supported. Skipping...")
        return None, None

    # Extract columns on-demand by importing the reader class
    # Columns are class-level information, not instance-specific
    try:
        module_path, class_name = stream_reader_path.rsplit(".", 1)
        reader_module = importlib.import_module(module_path)
        reader_class = getattr(reader_module, class_name)

        # Instantiate with dummy pattern and stored kwargs
        # (columns are instance attributes set in __init__, not class attributes)
        reader_instance = reader_class("_dummy_pattern_", **stream_reader_kwargs)

        if not hasattr(reader_instance, 'columns'):
            logger.error(
                f"Reader {stream_reader_path} has no 'columns' attribute. "
                "Cannot generate table definition."
            )
            return None, None

        columns = reader_instance.columns
        if columns is None:
            logger.warning(
                f"Reader {stream_reader_path} has None columns. "
                "Table will only have timestamps."
            )
            columns = []

    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Cannot import reader module for {device_type}.{stream_type}: {e}")
        return None, None
    except AttributeError as e:
        logger.error(f"Reader class {stream_reader_path} not found: {e}")
        return None, None
    except Exception as e:
        logger.warning(
            f"Could not extract columns for {device_type}.{stream_type} "
            f"from {stream_reader_path}: {e}"
        )
        return None, None

    table_definition = f""" # Raw per-chunk {stream_type} from {device_type} (v-{aeon.__version__})
    -> {device_type}
    -> acquisition.Chunk
    ---
    sample_count: int32      # number of data points
    timestamps: json         # time range, sampling rate, count
    """

    for col in columns:
        if col.startswith("_"):
            continue
        new_col = re.sub(r"\([^)]*\)", "", col)
        table_definition += f"{new_col}: json             # summary stats\n    "

    table_definition += "stream_df: <aeon_stream>   # full DataFrame via codec\n    "

    class DeviceDataStream(dj.Imported):
        definition = table_definition

        @property
        def key_source(self):
            """Only the combination of Chunk and {device_type} with overlapping time.

            + Chunk(s) started after {device_type} install time & ended before {device_type} remove time
            + Chunk(s) started after {device_type} install time for {device_type} and not yet removed
            """
            return (
                acquisition.Chunk.join(
                    ExperimentDevice.join(ExperimentDevice.RemovalTime, left=True),
                    semantic_check=False,
                )
                & "chunk_start >= {device_type_name}_install_time"
                & 'chunk_start < IFNULL({device_type_name}_removal_time,"2200-01-01")'
            )

        def make(self, key):
            """Load stream data, compute summary stats, and store codec reference."""
            from swc.aeon.io import api as io_api

            from aeon.dj_pipeline.utils.codec import column_stats, timestamp_stats
            from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

            chunk_start, chunk_end, epoch_start = (acquisition.Chunk & key).fetch1(
                "chunk_start", "chunk_end", "epoch_start"
            )
            data_dirs = acquisition.Experiment.get_data_directories(key)
            device_name = key["device_name"]

            stream_reader = get_stream_reader_for_epoch(
                key["experiment_name"], device_name, "{stream_type}", epoch_start
            )

            stream_data = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            # Summary stats for JSON columns
            row = {
                **key,
                "sample_count": len(stream_data),
                "timestamps": timestamp_stats(stream_data.index) if len(stream_data) > 0 else {},
            }
            for col in stream_reader.columns:
                if col.startswith("_"):
                    continue
                clean_col = re.sub(r"\([^)]*\)", "", col)
                row[clean_col] = column_stats(stream_data[col].values) if len(stream_data) > 0 else {}

            # Codec reference for stream_df (self-contained for decode)
            row["stream_df"] = {
                "stream_type": "{stream_type}",
                "experiment_name": key["experiment_name"],
                "device_name": device_name,
                "chunk_start": str(chunk_start),
                "chunk_end": str(chunk_end),
                "epoch_start": str(epoch_start),
            }

            self.insert1(row, ignore_extra_fields=True)

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
            for table_class in (StreamType, DeviceType, DeviceName, Device):
                device_table_def = inspect.getsource(table_class).lstrip()
                full_def = "@schema \n" + device_table_def + "\n\n"
                f.write(full_def)

    streams = importlib.import_module("aeon.dj_pipeline.streams")

    if create_tables:
        # Create DeviceType tables.
        for device_info in streams.DeviceType.to_dicts():
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
        # Join with StreamType to get stream_type (DeviceType.Stream only has stream_hash FK)
        for device_info in (streams.DeviceType.Stream * streams.StreamType).to_dicts():
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
