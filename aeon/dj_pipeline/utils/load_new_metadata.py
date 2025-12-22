"""Load metadata from the experiment and insert into streams schema.

This module handles the Pydantic-based metadata format where devices are organized
in Rig objects with device collections like 'cameras', 'feeders', 'nest', etc.
"""

import datetime
import importlib
import inspect
import json
import pathlib
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, TYPE_CHECKING

import datajoint as dj
import numpy as np
from dotmap import DotMap
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import dict_to_uuid
from aeon.dj_pipeline.utils import streams_maker

if TYPE_CHECKING:
    from swc.aeon.schema import BaseSchema, Device

logger = dj.logger


def get_experiment_class(schema_name: str) -> type["BaseSchema"]:
    """Get Experiment class from schema name.
    
    Schema name is a class path like 'swc.aeon.exp.foragingABC.experiment.ForagingABC'
    or 'swc.aeon.exp.foragingABC.experiment.Experiment'.
    
    Args:
        schema_name: Full class path (module.path.ClassName)
        
    Returns:
        Experiment class type
    """
    module_path, class_name = schema_name.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def extract_rig_from_metadata(experiment_class: type, metadata_filepath: Path):
    """Extract Rig instance from metadata file.
    
    Handles both structures:
    - experiment.experiment.rig (ForagingABC -> Experiment -> Rig)
    - experiment.rig (Experiment -> Rig)
    
    Args:
        experiment_class: Experiment class type
        metadata_filepath: Path to Metadata.json file
        
    Returns:
        Rig instance with device hierarchy intact
    """
    metadata = json.loads(metadata_filepath.read_text())
    experiment = experiment_class.model_validate(metadata)
    
    # Navigate to Rig - try nested structure first
    if hasattr(experiment, 'experiment') and hasattr(experiment.experiment, 'rig'):
        return experiment.experiment.rig
    elif hasattr(experiment, 'rig'):
        return experiment.rig
    else:
        raise ValueError(f"No rig found in {experiment_class}")


def insert_device_types(rig: "BaseSchema", metadata_filepath: Path) -> None:
    """Insert device types into streams.DeviceType and streams.Device.

    Notes: Use Pydantic Rig and metadata file to insert into streams.DeviceType and streams.Device.
    Only insert device types that were defined both in the Rig and metadata file.
    It then creates new device tables under streams schema.
    
    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections
        metadata_filepath: Path to metadata file
    """
    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    device_info: dict[str, dict] = get_device_info(rig)
    device_type_mapper, device_sn = get_device_mapper_from_rig(rig, metadata_filepath)

    # Add device type to device_info. Only add if device types that are defined in metadata file
    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name),
            **device_info[device_name],
        }
        for device_name in device_info
        if device_type_mapper.get(device_name) and device_sn.get(device_name)
    }

    # Create a map of device_type to stream_type.
    device_stream_map: dict[str, list] = {}

    for device_config in device_info.values():
        device_type = device_config["device_type"]
        stream_types = device_config["stream_type"]

        if device_type not in device_stream_map:
            device_stream_map[device_type] = []

        for stream_type in stream_types:
            if stream_type not in device_stream_map[device_type]:
                device_stream_map[device_type].append(stream_type)

    # List only new device & stream types that need to be inserted & created.
    new_device_types = [
        {"device_type": device_type}
        for device_type in device_stream_map
        if not streams.DeviceType & {"device_type": device_type}
    ]

    new_device_stream_types = [
        {"device_type": device_type, "stream_type": stream_type}
        for device_type, stream_list in device_stream_map.items()
        for stream_type in stream_list
        if not streams.DeviceType.Stream & {"device_type": device_type, "stream_type": stream_type}
    ]

    new_devices = [
        {
            "device_serial_number": device_sn[device_name],
            "device_type": device_config["device_type"],
        }
        for device_name, device_config in device_info.items()
        if device_sn[device_name] and not streams.Device & {"device_serial_number": device_sn[device_name]}
    ]

    # Insert new entries.
    if new_device_types:
        streams.DeviceType.insert(new_device_types)

        if new_device_stream_types:
            streams.DeviceType.Stream.insert(new_device_stream_types)

    if new_devices:
        streams.Device.insert(new_devices)


def extract_epoch_config(experiment_name: str, devices_schema: DotMap, metadata_filepath: str) -> dict[str, Any]:
    """Parse experiment metadata file and extract epoch configuration.

    Args:
        experiment_name: Name of the experiment.
        devices_schema: DotMap object containing device schema definitions.
        metadata_filepath: Path to the metadata file.

    Returns:
        dict: epoch_config containing experiment metadata and device configurations.
    """
    metadata_filepath = pathlib.Path(metadata_filepath)
    epoch_start = datetime.datetime.strptime(metadata_filepath.parent.name, "%Y-%m-%dT%H-%M-%S")

    # Load metadata using io_api
    epoch_config: dict = (
        io_api.load(
            metadata_filepath.parent.as_posix(),
            devices_schema.Metadata,
        )
        .reset_index()
        .to_dict("records")[0]
    )

    commit = epoch_config.get("commit")
    if isinstance(commit, float) and np.isnan(commit):
        commit = epoch_config.get("metadata", {}).get("Revision")

    if not commit:
        raise ValueError(f'Neither "commit" nor "Revision" found in {metadata_filepath}')

    workflow = epoch_config.get("workflow")
    if not workflow:
        raise ValueError(f'"workflow" not found in {metadata_filepath}')

    # Extract rig configuration (nested structure)
    rig_config = epoch_config['metadata'].get("rig", {})
    if not rig_config:
        raise ValueError(f'"rig" configuration not found in {metadata_filepath}')

    # Flatten nested device structure into flat dict for compatibility with downstream code
    devices = _flatten_rig_devices(rig_config)

    return {
        "experiment_name": experiment_name,
        "epoch_start": epoch_start,
        "bonsai_workflow": workflow,
        "commit": commit,
        "metadata": devices,
        "rig_config": rig_config,  # Keep original rig config for trigger lookups
        "metadata_file_path": metadata_filepath,
    }


def _flatten_rig_devices(rig_config: dict) -> dict[str, dict]:
    """Flatten nested rig device structure into flat device dict.

    Converts:
        rig.cameras.CameraTop -> devices["CameraTop"]
        rig.feeders.Feeder1 -> devices["Feeder1"]
        rig.nest.Nest -> devices["Nest"]
        etc.

    Args:
        rig_config: Nested rig configuration dict.

    Returns:
        Flat dict of device_name -> device_config.
    """
    devices: dict[str, dict] = {}

    # Extract cameras
    cameras = rig_config.get("cameras", {})
    for camera_name, camera_config in cameras.items():
        devices[camera_name] = dict(camera_config)

    # Extract feeders
    feeders = rig_config.get("feeders", {})
    for feeder_name, feeder_config in feeders.items():
        devices[feeder_name] = dict(feeder_config)

    # Extract nest (WeightScale)
    nest = rig_config.get("nest", {})
    if "Nest" in nest:
        devices["Nest"] = dict(nest["Nest"])

    # Extract camera synchronizer (VideoController equivalent)
    camera_synchronizer = rig_config.get("cameraSynchronizer", {})
    if camera_synchronizer:
        devices["CameraSynchronizer"] = dict(camera_synchronizer)

    # Extract clock synchronizer
    clock_synchronizer = rig_config.get("clockSynchronizer", {})
    if clock_synchronizer:
        devices["ClockSynchronizer"] = dict(clock_synchronizer)

    # Extract light cycle
    light_cycle = rig_config.get("lightCycle", {})
    if light_cycle:
        devices["LightCycle"] = dict(light_cycle)

    return devices


def extract_active_regions(rig_config: dict) -> dict[str, Any]:
    """Extract ActiveRegion data from rig configuration.

    In the new format, regions are nested in camera tracking configurations.
    This function extracts and flattens them into the format expected by EpochConfig.ActiveRegion.

    Args:
        rig_config: Nested rig configuration dict.

    Returns:
        dict: ActiveRegion data in format {region_name: region_data}
    """
    active_regions: dict[str, Any] = {}

    # Extract regions from camera tracking configs
    cameras = rig_config.get("cameras", {})
    for camera_name, camera_config in cameras.items():
        camera_tracking = camera_config.get("cameraTracking")
        if not camera_tracking:
            continue

        # Extract blob tracking regions
        blob_tracking = camera_tracking.get("blobTracking")
        if blob_tracking:
            for region_name, region_config in blob_tracking.items():
                if region_name == "threshold":
                    continue
                # Store with camera prefix to avoid conflicts
                region_key = f"{camera_name}_{region_name}"
                active_regions[region_key] = region_config

    # Extract activity center regions if present
    activity_center = rig_config.get("activityCenter")
    if activity_center and "regions" in activity_center:
        active_regions["ActivityCenter"] = activity_center

    return active_regions


def _infer_device_type_from_rig(device_name: str, rig_config: dict) -> str | None:
    """Infer device type from location in rig structure.

    Device types are deterministically inferred from where they appear in the rig structure.
    This is the source of truth - no cache needed.

    Args:
        device_name: Name of the device.
        rig_config: Nested rig configuration dict.

    Returns:
        Device type string, or None if device not found in rig structure.
    """
    # Check cameras
    if device_name in rig_config.get("cameras", {}):
        return "SpinnakerVideoSource"

    # Check feeders
    if device_name in rig_config.get("feeders", {}):
        return "UndergroundFeeder"

    # Check nest
    if device_name == "Nest" and "Nest" in rig_config.get("nest", {}):
        return "WeightScale"

    # Check camera synchronizer
    if device_name == "CameraSynchronizer" and rig_config.get("cameraSynchronizer"):
        return "CameraController"

    # Check clock synchronizer
    if device_name == "ClockSynchronizer" and rig_config.get("clockSynchronizer"):
        return "TimestampGenerator"

    # Check light cycle
    if device_name == "LightCycle" and rig_config.get("lightCycle"):
        return "EnvironmentCondition"

    return None


def _extract_device_mapper_from_rig(rig_config: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Extract device type mapper and serial numbers from rig structure.

    Calculates device types on-the-fly from rig structure. No persistent cache needed
    since the structure itself is the source of truth.

    Args:
        rig_config: Nested rig configuration dict.

    Returns:
        tuple: (device_type_mapper, device_sn)
            device_type_mapper: {"device_name": "device_type"}
            device_sn: {"device_name": "serial_number"}
    """
    device_type_mapper: dict[str, str] = {}
    device_sn: dict[str, str] = {}

    # Extract cameras
    cameras = rig_config.get("cameras", {})
    for camera_name, camera_config in cameras.items():
        device_type_mapper[camera_name] = "SpinnakerVideoSource"
        device_sn[camera_name] = camera_config.get("serialNumber") or None

    # Extract feeders
    feeders = rig_config.get("feeders", {})
    for feeder_name, feeder_config in feeders.items():
        device_type_mapper[feeder_name] = "UndergroundFeeder"
        device_sn[feeder_name] = feeder_config.get("portName") or None

    # Extract nest
    nest = rig_config.get("nest", {})
    if "Nest" in nest:
        device_type_mapper["Nest"] = "WeightScale"
        device_sn["Nest"] = nest["Nest"].get("portName") or None

    # Extract camera synchronizer
    camera_synchronizer = rig_config.get("cameraSynchronizer", {})
    if camera_synchronizer:
        device_type_mapper["CameraSynchronizer"] = "CameraController"
        device_sn["CameraSynchronizer"] = camera_synchronizer.get("portName") or None

    # Extract clock synchronizer
    clock_synchronizer = rig_config.get("clockSynchronizer", {})
    if clock_synchronizer:
        device_type_mapper["ClockSynchronizer"] = "TimestampGenerator"
        device_sn["ClockSynchronizer"] = clock_synchronizer.get("portName") or None

    # Extract light cycle
    light_cycle = rig_config.get("lightCycle", {})
    if light_cycle:
        device_type_mapper["LightCycle"] = "EnvironmentCondition"
        device_sn["LightCycle"] = None  # Light cycle doesn't have serial/port

    return device_type_mapper, device_sn


def ingest_epoch_metadata(experiment_name: str, devices_schema: DotMap, metadata_filepath: str) -> set[str]:
    """Make entries into device tables.

    Args:
        experiment_name: Name of the experiment.
        devices_schema: DotMap object containing device schema definitions.
        metadata_filepath: Path to metadata file.

    Returns:
        Set of device type names that were processed.
    """
    from aeon.dj_pipeline import acquisition

    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    experiment_key = {"experiment_name": experiment_name}
    metadata_filepath = pathlib.Path(metadata_filepath)
    epoch_config = extract_epoch_config(experiment_name, devices_schema, metadata_filepath)

    previous_epoch = (acquisition.Experiment & experiment_key).aggr(
        acquisition.Epoch & f'epoch_start < "{epoch_config["epoch_start"]}"',
        epoch_start="MAX(epoch_start)",
    )
    if len(acquisition.EpochConfig.Meta & previous_epoch) and epoch_config["commit"] == (
        acquisition.EpochConfig.Meta & previous_epoch
    ).fetch1("commit"):
        # if identical commit -> no changes
        return set()

    device_type_mapper, _ = _extract_device_mapper_from_rig(epoch_config["rig_config"])
    rig_config = epoch_config["rig_config"]


def ingest_epoch_metadata_from_rig(
    experiment_name: str, rig: "BaseSchema", epoch_config: dict[str, Any], metadata_filepath: Path
) -> set[str]:
    """Make entries into device tables using Pydantic Rig.

    Args:
        experiment_name: Name of the experiment.
        rig: Rig instance (Pydantic BaseSchema) containing device collections
        epoch_config: Dictionary containing epoch configuration (epoch_start, commit, rig_config, metadata)
        metadata_filepath: Path to metadata file.

    Returns:
        Set of device type names that were processed.
    """
    from aeon.dj_pipeline import acquisition

    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    experiment_key = {"experiment_name": experiment_name}

    previous_epoch = (acquisition.Experiment & experiment_key).aggr(
        acquisition.Epoch & f'epoch_start < "{epoch_config["epoch_start"]}"',
        epoch_start="MAX(epoch_start)",
    )
    if len(acquisition.EpochConfig.Meta & previous_epoch) and epoch_config["commit"] == (
        acquisition.EpochConfig.Meta & previous_epoch
    ).fetch1("commit"):
        # if identical commit -> no changes
        return set()

    device_type_mapper, _ = get_device_mapper_from_rig(rig, metadata_filepath)
    rig_config = epoch_config["rig_config"]

    # Extract trigger frequencies from camera synchronizer
    camera_synchronizer = rig_config.get("cameraSynchronizer", {})
    trigger_frequencies = {}
    if camera_synchronizer and "triggers" in camera_synchronizer:
        for trigger_name, trigger_config in camera_synchronizer["triggers"].items():
            trigger_frequencies[trigger_name] = trigger_config.get("frequency")

    # Insert into each device table
    epoch_device_types = []
    device_list = []
    device_removal_list = []

    for device_name, device_config in epoch_config["metadata"].items():
        if table := getattr(streams, device_type_mapper.get(device_name) or "", None):
            # Extract serial number or port name
            device_sn = device_config.get("serialNumber") or device_config.get("portName")
            device_key = {"device_serial_number": device_sn}

            if not device_sn:
                logger.warning(
                    f"Device {device_name} has no serial number or port name. Skipping..."
                )
                continue

            if not (streams.Device & device_key):
                logger.warning(
                    f"Device {device_name} (serial number: {device_sn}) is not "
                    "yet registered in streams.Device.\nThis should not happen - "
                    "check if metadata file and schemas dotmap are consistent. Skipping..."
                )
                continue

            device_list.append(device_key)
            epoch_device_types.append(table.__name__)

            table_entry = {
                "experiment_name": experiment_name,
                **device_key,
                f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config["epoch_start"],
                f"{dj.utils.from_camel_case(table.__name__)}_name": device_name,
            }

            # Convert device config to attribute entries
            table_attribute_entry = []
            for attr_name, attr_value in device_config.items():
                # Skip nested structures and internal fields
                if isinstance(attr_value, (dict, list)):
                    continue
                table_attribute_entry.append(
                    {
                        **table_entry,
                        "attribute_name": attr_name,
                        "attribute_value": attr_value,
                    }
                )

            # Add trigger frequency if camera has trigger reference
            if "trigger" in device_config and device_config["trigger"] in trigger_frequencies:
                table_attribute_entry.append(
                    {
                        **table_entry,
                        "attribute_name": "SamplingFrequency",
                        "attribute_value": trigger_frequencies[device_config["trigger"]],
                    }
                )

            # Check if this device is currently installed.
            # If the same device serial number is currently installed check for changes in configuration.
            current_device_query = table - table.RemovalTime & experiment_key & device_key

            if current_device_query:
                current_device_config: list[dict] = (table.Attribute & current_device_query).fetch(
                    "experiment_name",
                    "device_serial_number",
                    "attribute_name",
                    "attribute_value",
                    as_dict=True,
                )
                new_device_config: list[dict] = [
                    {k: v for k, v in entry.items() if dj.utils.from_camel_case(table.__name__) not in k}
                    for entry in table_attribute_entry
                ]

                if dict_to_uuid(
                    {
                        config["attribute_name"]: config["attribute_value"]
                        for config in current_device_config
                    }
                ) == dict_to_uuid(
                    {config["attribute_name"]: config["attribute_value"] for config in new_device_config}
                ):  # Skip if none of the configuration has changed.
                    continue

                # Remove old device
                device_removal_list.append(
                    {
                        **current_device_query.fetch1("KEY"),
                        f"{dj.utils.from_camel_case(table.__name__)}_removal_time": epoch_config[
                            "epoch_start"
                        ],
                    }
                )
                epoch_device_types.remove(table.__name__)

            # Insert into table.
            table.insert1(table_entry, skip_duplicates=True)
            table.Attribute.insert(table_attribute_entry, ignore_extra_fields=True)

    # Remove the currently installed devices that are absent in this config
    device_removal = lambda device_type, device_entry: any(
        dj.utils.from_camel_case(device_type) in k for k in device_entry
    )  # returns True if the device type is found in the attribute name

    for device_type in streams.DeviceType.fetch("device_type"):
        table = getattr(streams, device_type)

        device_removal_list.extend(
            (table - table.RemovalTime - device_list & experiment_key).fetch("KEY")
        )

        for device_entry in device_removal_list:
            if device_removal(device_type, device_entry):
                table.RemovalTime.insert1(device_entry)

    return set(epoch_device_types)


def ingest_epoch_metadata(experiment_name: str, devices_schema: DotMap, metadata_filepath: str) -> set[str]:
    """Make entries into device tables (legacy DotMap version).

    Args:
        experiment_name: Name of the experiment.
        devices_schema: DotMap object containing device schema definitions.
        metadata_filepath: Path to metadata file.

    Returns:
        Set of device type names that were processed.
    """
    from aeon.dj_pipeline import acquisition

    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    experiment_key = {"experiment_name": experiment_name}
    metadata_filepath = pathlib.Path(metadata_filepath)
    epoch_config = extract_epoch_config(experiment_name, devices_schema, metadata_filepath)

    previous_epoch = (acquisition.Experiment & experiment_key).aggr(
        acquisition.Epoch & f'epoch_start < "{epoch_config["epoch_start"]}"',
        epoch_start="MAX(epoch_start)",
    )
    if len(acquisition.EpochConfig.Meta & previous_epoch) and epoch_config["commit"] == (
        acquisition.EpochConfig.Meta & previous_epoch
    ).fetch1("commit"):
        # if identical commit -> no changes
        return set()

    device_type_mapper, _ = _extract_device_mapper_from_rig(epoch_config["rig_config"])
    rig_config = epoch_config["rig_config"]

    # Extract trigger frequencies from camera synchronizer
    camera_synchronizer = rig_config.get("cameraSynchronizer", {})
    trigger_frequencies = {}
    if camera_synchronizer and "triggers" in camera_synchronizer:
        for trigger_name, trigger_config in camera_synchronizer["triggers"].items():
            trigger_frequencies[trigger_name] = trigger_config.get("frequency")

    # Insert into each device table
    epoch_device_types = []
    device_list = []
    device_removal_list = []

    for device_name, device_config in epoch_config["metadata"].items():
        if table := getattr(streams, device_type_mapper.get(device_name) or "", None):
            # Extract serial number or port name
            device_sn = device_config.get("serialNumber") or device_config.get("portName")
            device_key = {"device_serial_number": device_sn}

            if not device_sn:
                logger.warning(
                    f"Device {device_name} has no serial number or port name. Skipping..."
                )
                continue

            if not (streams.Device & device_key):
                logger.warning(
                    f"Device {device_name} (serial number: {device_sn}) is not "
                    "yet registered in streams.Device.\nThis should not happen - "
                    "check if metadata file and schemas dotmap are consistent. Skipping..."
                )
                continue

            device_list.append(device_key)
            epoch_device_types.append(table.__name__)

            table_entry = {
                "experiment_name": experiment_name,
                **device_key,
                f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config["epoch_start"],
                f"{dj.utils.from_camel_case(table.__name__)}_name": device_name,
            }

            # Convert device config to attribute entries
            table_attribute_entry = []
            for attr_name, attr_value in device_config.items():
                # Skip nested structures and internal fields
                if isinstance(attr_value, (dict, list)):
                    continue
                table_attribute_entry.append(
                    {
                        **table_entry,
                        "attribute_name": attr_name,
                        "attribute_value": attr_value,
                    }
                )

            # Add trigger frequency if camera has trigger reference
            if "trigger" in device_config and device_config["trigger"] in trigger_frequencies:
                table_attribute_entry.append(
                    {
                        **table_entry,
                        "attribute_name": "SamplingFrequency",
                        "attribute_value": trigger_frequencies[device_config["trigger"]],
                    }
                )

            # Check if this device is currently installed.
            # If the same device serial number is currently installed check for changes in configuration.
            current_device_query = table - table.RemovalTime & experiment_key & device_key

            if current_device_query:
                current_device_config: list[dict] = (table.Attribute & current_device_query).fetch(
                    "experiment_name",
                    "device_serial_number",
                    "attribute_name",
                    "attribute_value",
                    as_dict=True,
                )
                new_device_config: list[dict] = [
                    {k: v for k, v in entry.items() if dj.utils.from_camel_case(table.__name__) not in k}
                    for entry in table_attribute_entry
                ]

                if dict_to_uuid(
                    {
                        config["attribute_name"]: config["attribute_value"]
                        for config in current_device_config
                    }
                ) == dict_to_uuid(
                    {config["attribute_name"]: config["attribute_value"] for config in new_device_config}
                ):  # Skip if none of the configuration has changed.
                    continue

                # Remove old device
                device_removal_list.append(
                    {
                        **current_device_query.fetch1("KEY"),
                        f"{dj.utils.from_camel_case(table.__name__)}_removal_time": epoch_config[
                            "epoch_start"
                        ],
                    }
                )
                epoch_device_types.remove(table.__name__)

            # Insert into table.
            table.insert1(table_entry, skip_duplicates=True)
            table.Attribute.insert(table_attribute_entry, ignore_extra_fields=True)

    # Remove the currently installed devices that are absent in this config
    device_removal = lambda device_type, device_entry: any(
        dj.utils.from_camel_case(device_type) in k for k in device_entry
    )  # returns True if the device type is found in the attribute name

    for device_type in streams.DeviceType.fetch("device_type"):
        table = getattr(streams, device_type)

        device_removal_list.extend(
            (table - table.RemovalTime - device_list & experiment_key).fetch("KEY")
        )

        for device_entry in device_removal_list:
            if device_removal(device_type, device_entry):
                table.RemovalTime.insert1(device_entry)

    return set(epoch_device_types)


def extract_stream_types_from_device(device_class: type) -> list[str]:
    """Extract @data_reader method names from Device class.
    
    @data_reader methods are defined directly on Device classes and are cached_property
    objects with a function that takes (self, pattern) as parameters.
    
    Args:
        device_class: Device class type
        
    Returns:
        List of @data_reader method names (snake_case)
    """
    stream_types = []
    for name, attr in inspect.getmembers(device_class):
        if isinstance(attr, cached_property):
            func = getattr(attr, 'func', None)
            if func:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                # @data_reader methods have (self, pattern) signature
                if len(params) == 2 and params[1] == 'pattern':
                    stream_types.append(name)
    return stream_types


def to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase for stream_type.
    
    Examples:
        "video" → "Video"
        "weight_raw" → "WeightRaw"
        "beam_break" → "BeamBreak"
        
    Args:
        snake_str: snake_case string
        
    Returns:
        PascalCase string
    """
    components = snake_str.split('_')
    return ''.join(word.capitalize() for word in components)


def get_device_info(rig: "BaseSchema") -> dict[str, dict]:
    """Parse Rig to extract device/stream info.
    
    Streams are defined as @data_reader methods directly on Device classes.
    Uses actual device instances from Rig to ensure pattern resolution works correctly.
    
    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections
        
    Returns:
        dict[str, dict]: A dictionary of device information, keyed by device_name.
        Same structure as legacy function for compatibility:
        {
            "device_name": {
                "stream_type": [...],
                "stream_reader": [...],
                "stream_reader_kwargs": [...],
                "stream_hash": [...]
            }
        }
    """
    def _get_class_path(obj) -> str:
        """Returns the class path of the object."""
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    
    device_info: dict[str, dict] = {}
    
    # Iterate over Rig fields to find Device collections
    for field_name in rig.model_fields:
        field_value = getattr(rig, field_name)
        
        # Handle Dict[Name, Device] - e.g., cameras, feeders, nest
        if isinstance(field_value, dict):
            for device_name, device in field_value.items():
                # Check if it's a Device instance (BaseSchema with device_type)
                if not hasattr(device, 'device_type'):
                    continue
                
                if device_name not in device_info:
                    device_info[device_name] = defaultdict(list)
                
                # Extract @data_reader methods from Device class
                stream_method_names = extract_stream_types_from_device(device.__class__)
                
                for stream_method_name in stream_method_names:
                    # Access @data_reader property on actual device instance
                    # This ensures pattern resolution works via _resolve_pattern_prefix()
                    try:
                        reader = getattr(device, stream_method_name)
                    except Exception as e:
                        logger.warning(
                            f"Failed to access {stream_method_name} on {device_name}: {e}. Skipping..."
                        )
                        continue
                    
                    # Convert snake_case method name to PascalCase stream_type
                    stream_type = to_pascal_case(stream_method_name)  # "video" → "Video"
                    
                    # Extract info (same format as legacy)
                    device_info[device_name]["stream_type"].append(stream_type)
                    device_info[device_name]["stream_reader"].append(_get_class_path(reader))
                    
                    # Extract reader kwargs - pattern is already resolved from device hierarchy
                    stream_reader_kwargs = {
                        "pattern": reader.pattern,
                        "columns": list(reader.columns) if reader.columns else None,
                        "extension": reader.extension,
                    }
                    device_info[device_name]["stream_reader_kwargs"].append(stream_reader_kwargs)
                    
                    # Add hash
                    device_info[device_name]["stream_hash"].append(
                        dict_to_uuid({
                            **stream_reader_kwargs,
                            "stream_reader": _get_class_path(reader),
                        })
                    )
        
        # Handle single Device instance (if any)
        elif hasattr(field_value, 'device_type'):
            # For single devices, use field_name as device_name
            device_name = field_name
            device = field_value
            
            if device_name not in device_info:
                device_info[device_name] = defaultdict(list)
            
            # Same extraction logic as above
            stream_method_names = extract_stream_types_from_device(device.__class__)
            
            for stream_method_name in stream_method_names:
                try:
                    reader = getattr(device, stream_method_name)
                except Exception as e:
                    logger.warning(
                        f"Failed to access {stream_method_name} on {device_name}: {e}. Skipping..."
                    )
                    continue
                
                stream_type = to_pascal_case(stream_method_name)
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["stream_reader"].append(_get_class_path(reader))
                
                stream_reader_kwargs = {
                    "pattern": reader.pattern,
                    "columns": list(reader.columns) if reader.columns else None,
                    "extension": reader.extension,
                }
                device_info[device_name]["stream_reader_kwargs"].append(stream_reader_kwargs)
                device_info[device_name]["stream_hash"].append(
                    dict_to_uuid({
                        **stream_reader_kwargs,
                        "stream_reader": _get_class_path(reader),
                    })
                )
    
    return device_info


def get_stream_entries(rig: "BaseSchema") -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for a given device.

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections

    Returns:
        list[dict]: List of dictionaries containing the stream entries for a given device.
    """
    device_info = get_device_info(rig)
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
            strict=True,
        )
    ]


def get_device_mapper_from_rig(rig: "BaseSchema", metadata_filepath: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Extract device type mapper and serial numbers from Pydantic Rig.
    
    Device types are extracted directly from device.device_type field.
    Serial numbers/port names are extracted from device attributes.

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections
        metadata_filepath: Path to metadata file (unused, kept for signature compatibility)

    Returns:
        tuple: (device_type_mapper, device_sn)
            device_type_mapper: {"device_name": "device_type"}
            device_sn: {"device_name": "serial_number"}
    """
    device_type_mapper: dict[str, str] = {}
    device_sn: dict[str, str] = {}
    
    # Iterate over Rig fields to find Device collections
    for field_name in rig.model_fields:
        field_value = getattr(rig, field_name)
        
        # Handle Dict[Name, Device] - e.g., cameras, feeders, nest
        if isinstance(field_value, dict):
            for device_name, device in field_value.items():
                # Check if it's a Device instance (BaseSchema with device_type)
                if not hasattr(device, 'device_type'):
                    continue
                
                device_type_mapper[device_name] = device.device_type
                # Extract serial number or port name
                device_sn[device_name] = (
                    getattr(device, 'serial_number', None) or
                    getattr(device, 'port_name', None) or
                    None
                )
        
        # Handle single Device instance (if any)
        elif hasattr(field_value, 'device_type'):
            device_name = field_name
            device = field_value
            device_type_mapper[device_name] = device.device_type
            device_sn[device_name] = (
                getattr(device, 'serial_number', None) or
                getattr(device, 'port_name', None) or
                None
            )
    
    return device_type_mapper, device_sn

