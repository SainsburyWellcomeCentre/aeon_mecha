"""Load metadata from the experiment and insert into streams schema.

This module handles the new nested metadata format where devices are organized
under 'rig' with categories like 'cameras', 'feeders', 'nest', etc.
"""

import datetime
import inspect
import json
import pathlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import datajoint as dj
import numpy as np
from dotmap import DotMap
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import dict_to_uuid
from aeon.dj_pipeline.utils import streams_maker

logger = dj.logger


def insert_stream_types() -> None:
    """Insert into streams.streamType table all streams in the aeon schemas."""
    from aeon.schema import ingestion_schemas as aeon_schemas

    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    for devices_schema_name in aeon_schemas.__all__:
        devices_schema = getattr(aeon_schemas, devices_schema_name)
        stream_entries = get_stream_entries(devices_schema)

        for entry in stream_entries:
            try:
                streams.StreamType.insert1(entry)
                logger.info(f"New stream type created: {entry['stream_type']}")
            except dj.errors.DuplicateError:
                existing_stream = (
                    streams.StreamType.proj("stream_reader", "stream_reader_kwargs")
                    & {"stream_type": entry["stream_type"]}
                ).fetch1()
                existing_columns = existing_stream["stream_reader_kwargs"].get("columns")
                entry_columns = entry["stream_reader_kwargs"].get("columns")
                if existing_columns != entry_columns:
                    logger.warning(f"Stream type already exists:\n\t{entry}\n\t{existing_stream}")


def insert_device_types(devices_schema: DotMap, metadata_filepath: Path) -> None:
    """Insert device types into streams.DeviceType and streams.Device.

    Notes: Use aeon.schema.schemas and metadata file to insert into streams.DeviceType and streams.Device.
    Only insert device types that were defined both in the device schema and metadata file.
    It then creates new device tables under streams schema.
    """
    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    device_info: dict[str, dict] = get_device_info(devices_schema)
    device_type_mapper, device_sn = get_device_mapper(devices_schema, metadata_filepath)

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
        try:
            streams.DeviceType.Stream.insert(new_device_stream_types)
        except dj.DataJointError:
            insert_stream_types()
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

    device_type_mapper, _ = get_device_mapper(devices_schema, metadata_filepath)
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


def get_stream_entries(devices_schema: DotMap) -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for a given device.

    Args:
        devices_schema: DotMap object containing device schema definitions.

    Returns:
        list[dict]: List of dictionaries containing the stream entries for a given device.
    """
    device_info = get_device_info(devices_schema)
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


def get_device_info(devices_schema: DotMap) -> dict[str, dict]:
    """Read from the DotMap object and returns a device dictionary.

    Args:
        devices_schema: DotMap object containing device schema definitions.

    Returns:
        dict[str, dict]: A dictionary of device information.
    """
    def _get_class_path(obj) -> str:
        """Returns the class path of the object."""
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

    schema_json = json.dumps(devices_schema, default=lambda x: x.__dict__, indent=4)
    schema_dict = json.loads(schema_json)
    device_info = {}

    for device_name, device in devices_schema.items():
        if device_name.startswith("_"):
            continue

        device_info[device_name] = defaultdict(list)

        if isinstance(device, DotMap):
            for stream_type, stream_obj in device.items():
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["stream_reader"].append(_get_class_path(stream_obj))

                required_args = [
                    k for k in inspect.signature(stream_obj.__init__).parameters if k != "self"
                ]
                pattern = schema_dict[device_name][stream_type].get("pattern")
                if pattern:
                    schema_dict[device_name][stream_type]["pattern"] = pattern.replace(device_name, "{pattern}")

                kwargs = {
                    k: v for k, v in schema_dict[device_name][stream_type].items() if k in required_args
                }
                device_info[device_name]["stream_reader_kwargs"].append(kwargs)
                # Add hash
                device_info[device_name]["stream_hash"].append(
                    dict_to_uuid({**kwargs, "stream_reader": _get_class_path(stream_obj)})
                )
        else:
            stream_type = device.__class__.__name__
            device_info[device_name]["stream_type"].append(stream_type)
            device_info[device_name]["stream_reader"].append(_get_class_path(device))

            required_args = [k for k in inspect.signature(device.__init__).parameters if k != "self"]
            pattern = schema_dict[device_name].get("pattern")
            if pattern:
                schema_dict[device_name]["pattern"] = pattern.replace(device_name, "{pattern}")

            kwargs = {k: v for k, v in schema_dict[device_name].items() if k in required_args}
            device_info[device_name]["stream_reader_kwargs"].append(kwargs)
            # Add hash
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({**kwargs, "stream_reader": _get_class_path(device)})
            )
    return device_info


def get_device_mapper(devices_schema: DotMap, metadata_filepath: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Returns a mapping dictionary of device names to types based on the dataset schema and metadata file.

    Notes: Device types are inferred on-the-fly from the rig structure location.
    The structure itself is the source of truth - no persistent cache needed.

    Args:
        devices_schema: DotMap object containing device schema definitions.
        metadata_filepath: Path to metadata file.

    Returns:
        tuple: (device_type_mapper, device_sn)
            device_type_mapper: {"device_name": "device_type"}
            device_sn: {"device_name": "serial_number"}
    """
    from swc.aeon.io import api as io_api

    metadata_filepath = Path(metadata_filepath)
    meta_data = (
        io_api.load(
            str(metadata_filepath.parent),
            devices_schema.Metadata,
        )
        .reset_index()
        .to_dict("records")[0]
    )

    rig_config = meta_data['metadata'].get("rig", {})
    if not rig_config:
        return {}, {}

    # Extract device types directly from rig structure (on-the-fly calculation)
    return _extract_device_mapper_from_rig(rig_config)

