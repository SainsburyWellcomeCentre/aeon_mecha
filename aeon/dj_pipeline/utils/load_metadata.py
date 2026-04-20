"""Metadata loading and catalog population for streams schema.

This module provides functions for:
- Extracting device and stream information from Pydantic Experiment/Rig classes
- Populating catalog tables (StreamType, DeviceType, Device)
- Ingesting epoch-specific device configurations into stream tables

Devices are organized in Rig objects with collections (cameras, feeders, nest, etc.).
Stream readers are defined via @data_reader decorated methods on Device classes.
"""

import importlib
import inspect
import json
import re
import typing
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import datajoint as dj

from aeon.dj_pipeline import get_schema_name
from aeon.dj_pipeline.utils import dict_to_uuid

if TYPE_CHECKING:
    from swc.aeon.schema import BaseSchema

logger = dj.logger

# Constants for annotation parsing
ANNOTATION_ARGS_COUNT = 2
DATA_READER_PARAMS_COUNT = 2

# region Catalog Population Functions


def get_data_reader_methods(device_class: type) -> list[tuple[str, Any]]:
    """Get @data_reader methods from Device class.

    @data_reader methods are cached_property objects with signature (self, pattern).
    The decorator wraps the original function in a closure, so we check both the
    direct func signature and the closure for the pattern parameter.

    Args:
        device_class: Device class type

    Returns:
        List of (method_name, method_func) tuples for @data_reader methods
    """
    data_reader_methods = []

    def has_pattern_signature(f) -> bool:
        """Check if function has signature (self, pattern)."""
        try:
            params = list(inspect.signature(f).parameters.keys())
            return len(params) == DATA_READER_PARAMS_COUNT and params[1] == "pattern"
        except (TypeError, ValueError):
            return False

    for name, attr in inspect.getmembers(device_class):
        if not isinstance(attr, cached_property):
            continue
        func = getattr(attr, "func", None)
        if func is None:
            continue
        # Direct function check
        if has_pattern_signature(func):
            data_reader_methods.append((name, func))
            continue
        # Check closure for original function (real @data_reader wraps in closure)
        closure = getattr(func, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    orig = cell.cell_contents
                except ValueError:
                    continue
                if callable(orig) and has_pattern_signature(orig):
                    data_reader_methods.append((name, func))
                    break
    return data_reader_methods


def _has_data_readers(cls) -> bool:
    """Check if cls is a device class (has @data_reader decorated methods)."""
    return isinstance(cls, type) and bool(get_data_reader_methods(cls))


def get_device_class_from_field(field_info) -> type | None:
    """Extract Device class from Pydantic field annotation.

    A class is considered a Device if it has @data_reader decorated methods.
    Handles both Dict[Name, Device] and single Device field types.

    Args:
        field_info: Pydantic FieldInfo from model_fields

    Returns:
        Device class type, or None if not a device field
    """
    annotation = field_info.annotation
    origin = typing.get_origin(annotation)

    # Handle Dict[Name, Device] - e.g., Dict[CameraName, Camera]
    if origin is dict:
        args = typing.get_args(annotation)
        if len(args) == ANNOTATION_ARGS_COUNT:
            # args[1] is the value type (Device class)
            device_class = args[1]
            if _has_data_readers(device_class):
                return device_class

    # Handle single Device field
    elif _has_data_readers(annotation):
        return annotation

    return None


def get_reader_path_from_annotation(func) -> str | None:
    """Extract reader class path from @data_reader method return type annotation.

    The return type annotation specifies the reader class, e.g.:
        def video(self, pattern) -> reader.Video:
            ...

    The @data_reader decorator wraps the original function, so we need to
    check the closure for the original function with concrete type annotations.

    Args:
        func: The @data_reader decorated method function (may be wrapper)

    Returns:
        Fully-qualified class path (e.g., "swc.aeon.io.reader.Video"), or None.
        Returns None for TypeVar annotations (base class placeholders).
    """

    def _get_return_type(f) -> type | None:
        """Get return type from function, handling TypeVar."""
        try:
            hints = typing.get_type_hints(f)
            return_type = hints.get("return")
            if return_type is not None and not isinstance(return_type, typing.TypeVar):
                return return_type
        except (NameError, TypeError):
            pass
        return None

    # First try the function directly
    return_type = _get_return_type(func)

    # If TypeVar or None, check closure for original function
    # (The @data_reader decorator wraps the original function in a closure)
    if return_type is None:
        closure = getattr(func, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    orig_func = cell.cell_contents
                    if callable(orig_func):
                        return_type = _get_return_type(orig_func)
                        if return_type is not None:
                            break
                except ValueError:
                    continue

    if return_type is None:
        return None

    # Get the fully-qualified path
    module = getattr(return_type, "__module__", None)
    name = getattr(return_type, "__name__", None)

    return f"{module}.{name}" if module and name else None


def _extract_kwargs_from_reader(reader) -> dict | None:
    """Extract constructor kwargs from reader instance.

    Uses inspect.signature to determine what kwargs the reader's __init__ accepts,
    then extracts those values from the instance attributes.

    Args:
        reader: Reader instance from @data_reader method

    Returns:
        Dict of kwargs (excluding 'pattern'), or None if no special kwargs
    """
    sig = inspect.signature(reader.__class__.__init__)
    kwargs = {}

    for param_name in sig.parameters:
        # Skip 'self' and 'pattern' (the required positional args)
        if param_name in ("self", "pattern") or not hasattr(reader, param_name):
            continue
        value = getattr(reader, param_name, None)
        if value is not None:
            # Handle numpy arrays, tuples -> list for JSON serialization
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            kwargs[param_name] = value

    return kwargs if kwargs else None


def get_reader_kwargs_from_device_class(device_class: type, method_name: str) -> dict | None:
    """Extract reader constructor kwargs by executing @data_reader method.

    Creates a minimal device instance using model_construct() (bypasses validation)
    and executes the @data_reader method to get the configured reader.
    Returns the kwargs dict or None.

    Args:
        device_class: Device class type (Pydantic BaseSchema)
        method_name: Name of the @data_reader method (snake_case)

    Returns:
        Dict of kwargs (excluding 'pattern'), or None if extraction fails
    """
    try:
        # Create minimal device instance bypassing validation
        # model_construct() allows creating instance without required fields
        device = device_class.model_construct()
        # Get the @data_reader property (returns reader instance)
        reader = getattr(device, method_name)
        # Extract kwargs from reader instance
        return _extract_kwargs_from_reader(reader)
    except Exception as e:
        logger.debug(f"Failed to extract kwargs for {device_class.__name__}.{method_name}: {e}")
        return None


def populate_catalog_from_pydantic(experiment_class: type["BaseSchema"]) -> None:
    """Populate catalog tables (StreamType, DeviceType) from Pydantic Experiment class.

    Extracts DeviceType, StreamType, and their relationships from the Pydantic
    class hierarchy. Also extracts reader constructor kwargs for readers that
    require additional arguments (e.g., BitmaskEvent needs value/tag).

    This function is idempotent - safe to call multiple times.

    Args:
        experiment_class: Pydantic Experiment class with rig field
    """
    streams = dj.VirtualModule("streams", get_schema_name("streams"))

    # Get Rig class from Experiment.rig field
    rig_field = experiment_class.model_fields.get("rig")
    if rig_field is None:
        logger.warning(
            f"Experiment class {experiment_class} has no 'rig' field. Skipping catalog population."
        )
        return

    rig_class = rig_field.annotation

    # Track entries to insert
    device_type_entries = []
    stream_type_entries = []
    device_stream_entries = []

    # Iterate over Rig fields to find Device classes
    for _field_name, field_info in rig_class.model_fields.items():
        device_class = get_device_class_from_field(field_info)
        if device_class is None:
            continue

        # Use the class name as the device type (not the inherited device_type field,
        # which gives parent class names like "HarpOutputExpander" for Feeder subclasses)
        device_type = device_class.__name__
        device_type_entries.append({"device_type": device_type})

        # Extract @data_reader methods
        data_reader_methods = get_data_reader_methods(device_class)

        for stream_name, func in data_reader_methods:
            # Get reader class path from return type annotation
            stream_reader_path = get_reader_path_from_annotation(func)
            if stream_reader_path is None:
                logger.debug(f"Could not get reader path for {device_class}.{stream_name}. Skipping.")
                continue

            # Extract reader constructor kwargs by executing @data_reader method
            stream_reader_kwargs = get_reader_kwargs_from_device_class(device_class, stream_name)

            # Convert snake_case to PascalCase
            stream_type = to_pascal_case(stream_name)

            # Compute hash for (stream_type, stream_reader) combination
            stream_hash = dict_to_uuid(
                {
                    "stream_type": stream_type,
                    "stream_reader": stream_reader_path,
                }
            )

            stream_type_entries.append(
                {
                    "stream_hash": stream_hash,
                    "stream_type": stream_type,
                    "stream_reader": stream_reader_path,
                    "stream_reader_kwargs": stream_reader_kwargs,
                }
            )

            device_stream_entries.append(
                {
                    "device_type": device_type,
                    "stream_hash": stream_hash,
                }
            )

    # Insert entries (using skip_duplicates for idempotency)
    # Deduplicate before inserting
    seen_device_types = set()
    unique_device_types = []
    for entry in device_type_entries:
        if entry["device_type"] not in seen_device_types:
            seen_device_types.add(entry["device_type"])
            unique_device_types.append(entry)

    seen_stream_hashes = set()
    unique_stream_types = []
    for entry in stream_type_entries:
        if entry["stream_hash"] not in seen_stream_hashes:
            seen_stream_hashes.add(entry["stream_hash"])
            unique_stream_types.append(entry)

    seen_device_streams = set()
    unique_device_streams = []
    for entry in device_stream_entries:
        key = (entry["device_type"], entry["stream_hash"])
        if key not in seen_device_streams:
            seen_device_streams.add(key)
            unique_device_streams.append(entry)

    # Insert in correct order (StreamType before DeviceType.Stream due to FK)
    # Wrap in transaction to prevent race conditions with multiple workers
    with dj.conn().transaction:
        for entry in unique_stream_types:
            streams.StreamType.insert1(entry, skip_duplicates=True)

        for entry in unique_device_types:
            streams.DeviceType.insert1(entry, skip_duplicates=True)

        for entry in unique_device_streams:
            streams.DeviceType.Stream.insert1(entry, skip_duplicates=True)

    if unique_device_types or unique_stream_types:
        logger.debug(
            f"Catalog population: {len(unique_device_types)} device types, "
            f"{len(unique_stream_types)} stream types, {len(unique_device_streams)} device-stream links"
        )


# endregion


def get_experiment_pydantic(schema_name: str) -> type["BaseSchema"]:
    """Get Pydantic Experiment class from schema name.

    Schema name is a class path like 'swc.aeon.exp.foragingABC.experiment:Experiment'.
    The format is 'module.path:ClassName' where ':' separates module from class.

    Args:
        schema_name: Full class path (module.path:ClassName)

    Returns:
        Pydantic Experiment class type
    """
    module_path, class_name = schema_name.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def to_snake_case(pascal_str: str) -> str:
    """Convert PascalCase to snake_case.

    This function assumes that every capital letter marks a word
    boundary, except when it appears at the start of the string.

    Args:
        pascal_str: PascalCase string (e.g., "BeamBreak", "Video")

    Returns:
        snake_case string (e.g., "beam_break", "video")
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", pascal_str).lower()


def _find_device_in_rig(rig, device_name: str):
    """Find a device by name in the Rig hierarchy.

    Searches through all device collections (cameras, feeders, nest, etc.)
    to find the device with the matching name.

    Args:
        rig: Rig instance
        device_name: Name of the device to find

    Returns:
        Device instance

    Raises:
        ValueError: If device not found in Rig
    """
    for field_name in type(rig).model_fields:
        field_value = getattr(rig, field_name)

        # Handle Dict[Name, Device] collections
        if isinstance(field_value, dict) and device_name in field_value:
            return field_value[device_name]

        # Handle single device fields
        if _has_data_readers(type(field_value)) and (
            field_name == device_name or getattr(field_value, "_container_prefix", None) == device_name
        ):
            return field_value

    raise ValueError(f"Device '{device_name}' not found in Rig")


def get_stream_reader_for_epoch(
    experiment_name: str,
    device_name: str,
    stream_type: str,
    epoch_start,
):
    """Get stream reader for a specific epoch.

    Reconstructs the Rig from metadata stored in EpochConfig.Meta (avoiding file I/O)
    and accesses the device's @data_reader property.

    Args:
        experiment_name: Name of the experiment
        device_name: Name of device instance (e.g., "CameraTop")
        stream_type: Type of stream in PascalCase (e.g., "Video")
        epoch_start: Start time of the epoch

    Returns:
        Reader instance configured for the device/stream
    """
    from aeon.dj_pipeline import acquisition

    # Get Experiment class path (e.g., "swc.aeon.exp.foragingABC.experiment:Experiment")
    schema_name = (acquisition.Experiment.DevicesSchema & {"experiment_name": experiment_name}).fetch1(
        "devices_schema_name"
    )

    # Get rig metadata for the specific epoch
    epoch_key = {"experiment_name": experiment_name, "epoch_start": epoch_start}
    rig_metadata = (acquisition.EpochConfig.Meta & epoch_key).fetch1("metadata")

    # MariaDB 10.3 aliases `json` columns to `longtext`, so DataJoint's auto
    # json.loads() doesn't fire. Deserialize manually if needed.
    if isinstance(rig_metadata, str):
        rig_metadata = json.loads(rig_metadata)

    # Get Rig class from Experiment class and reconstruct directly
    experiment_class = get_experiment_pydantic(schema_name)
    rig_class = experiment_class.model_fields["rig"].annotation
    rig = rig_class.model_validate(rig_metadata)

    # Find device in Rig and access stream reader
    device = _find_device_in_rig(rig, device_name)
    stream_method = to_snake_case(stream_type)  # "Video" → "video"
    return getattr(device, stream_method)


def insert_stream_types(rig: "BaseSchema") -> None:
    """Insert stream types into streams.StreamType from Rig.

    Extracts all unique stream types from device @data_reader methods and inserts
    them into StreamType catalog table. Also extracts reader constructor kwargs
    for readers that require additional arguments.

    Called automatically by insert_device_types() when FK constraint failures
    indicate missing StreamType entries.

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections
    """
    streams = dj.VirtualModule("streams", get_schema_name("streams"))
    stream_entries = get_stream_entries(rig)

    # Deduplicate by stream_hash (same stream type may appear on multiple devices)
    seen_hashes = set()
    for entry in stream_entries:
        if entry["stream_hash"] in seen_hashes:
            continue
        seen_hashes.add(entry["stream_hash"])

        stream_type_entry = {
            "stream_hash": entry["stream_hash"],
            "stream_type": entry["stream_type"],
            "stream_reader": entry["stream_reader"],
            "stream_reader_kwargs": entry.get("stream_reader_kwargs"),
        }
        # Use skip_duplicates to handle race conditions
        streams.StreamType.insert1(stream_type_entry, skip_duplicates=True)

    if seen_hashes:
        logger.debug(f"Processed {len(seen_hashes)} unique StreamType entries")


def insert_device_types(rig: "BaseSchema", metadata_filepath: Path) -> None:
    """Insert device types, device names, and devices into streams schema.

    Populates:
    - streams.DeviceType: Catalog of device types (e.g., "SpinnakerCamera")
    - streams.DeviceType.Stream: Links device types to stream types
    - streams.DeviceName: Catalog of device instance names (e.g., "CameraTop")
    - streams.Device: Physical devices with serial numbers (optional)

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections
        metadata_filepath: Path to metadata file
    """
    streams = dj.VirtualModule("streams", get_schema_name("streams"))

    device_info: dict[str, dict] = get_device_info(rig)
    device_type_mapper, device_sn = get_device_mapper_from_rig(rig, metadata_filepath)

    # Add device type to device_info. Only include devices that:
    # 1. Have a device_type defined in metadata
    # 2. Have at least one stream (skip infrastructure devices like clock_synchronizer)
    # Note: serial number is NOT required - DeviceName is the primary key now
    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name),
            **device_info[device_name],
        }
        for device_name in device_info
        if device_type_mapper.get(device_name) and device_info[device_name].get("stream_type")
    }

    # Create a map of device_type to (stream_type, stream_hash) pairs
    device_stream_map: dict[str, list[tuple[str, str]]] = {}

    for device_config in device_info.values():
        device_type = device_config["device_type"]
        stream_types = device_config["stream_type"]
        stream_hashes = device_config["stream_hash"]

        if device_type not in device_stream_map:
            device_stream_map[device_type] = []

        for stream_type, stream_hash in zip(stream_types, stream_hashes, strict=True):
            pair = (stream_type, stream_hash)
            if pair not in device_stream_map[device_type]:
                device_stream_map[device_type].append(pair)

    # List only new device & stream types that need to be inserted & created.
    new_device_types = [
        {"device_type": device_type}
        for device_type in device_stream_map
        if not streams.DeviceType & {"device_type": device_type}
    ]

    new_device_stream_types = [
        {"device_type": device_type, "stream_hash": stream_hash}
        for device_type, stream_list in device_stream_map.items()
        for _stream_type, stream_hash in stream_list
        if not streams.DeviceType.Stream & {"device_type": device_type, "stream_hash": stream_hash}
    ]

    # DeviceName entries - device_name is now the primary key for ExperimentDevice tables
    new_device_names = [
        {"device_name": device_name, "device_type": device_config["device_type"]}
        for device_name, device_config in device_info.items()
        if not streams.DeviceName & {"device_name": device_name}
    ]

    # Device entries - only for devices with serial numbers (optional hardware tracking)
    new_devices = [
        {
            "device_serial_number": device_sn[device_name],
            "device_type": device_config["device_type"],
        }
        for device_name, device_config in device_info.items()
        if device_sn.get(device_name)
        and not streams.Device & {"device_serial_number": device_sn[device_name]}
    ]

    # Insert new entries.
    if new_device_types:
        streams.DeviceType.insert(new_device_types)

    # Note: DeviceType.Stream may need insertion even without new DeviceTypes
    # (when existing device types get new stream types)
    if new_device_stream_types:
        try:
            streams.DeviceType.Stream.insert(new_device_stream_types)
        except dj.DataJointError as e:
            # Only handle FK constraint violations (MySQL error 1452)
            msg = str(e).lower()
            if "foreign key constraint fails" in msg or "1452" in msg:
                logger.info("FK constraint on DeviceType.Stream, inserting missing StreamType")
                insert_stream_types(rig)
                streams.DeviceType.Stream.insert(new_device_stream_types)
            else:
                raise  # Re-raise unexpected errors

    # Insert DeviceName entries (must be after DeviceType due to FK)
    if new_device_names:
        streams.DeviceName.insert(new_device_names)

    # Insert Device entries (optional, for hardware tracking)
    if new_devices:
        streams.Device.insert(new_devices)


def flatten_rig_devices(rig_config: dict) -> dict[str, dict]:
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
    for camera_name, camera_config in rig_config.get("cameras", {}).items():
        blob_tracking = camera_config.get("cameraTracking", {}).get("blobTracking", {})
        for region_name, region_config in blob_tracking.items():
            if region_name != "threshold":
                active_regions[f"{camera_name}_{region_name}"] = region_config

    # Extract activity center regions if present
    activity_center = rig_config.get("activityCenter", {})
    if "regions" in activity_center:
        active_regions["ActivityCenter"] = activity_center

    return active_regions


def _extract_device_mapper_from_rig(rig_config: dict) -> tuple[dict[str, str], dict[str, str | None]]:  # pyright: ignore[reportUnusedFunction]
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
    device_sn: dict[str, str | None] = {}

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

    streams = dj.VirtualModule("streams", get_schema_name("streams"))

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
    rig_config = epoch_config["metadata"]

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

    for device_name, device_config in epoch_config["devices"].items():
        if table := getattr(streams, device_type_mapper.get(device_name) or "", None):
            # device_name is now the primary key
            device_key = {"device_name": device_name}

            # Check DeviceName exists (should have been inserted by insert_device_types)
            if not (streams.DeviceName & device_key):
                logger.warning(
                    f"Device name '{device_name}' not registered in streams.DeviceName. "
                    "This should not happen - check if metadata file and Rig schema are consistent. "
                    "Skipping..."
                )
                continue

            device_list.append(device_key)
            epoch_device_types.append(table.__name__)

            # Serial number is now optional attribute (for hardware tracking)
            device_sn = device_config.get("serialNumber") or device_config.get("portName")

            table_entry = {
                "experiment_name": experiment_name,
                **device_key,
                f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config["epoch_start"],
                "device_serial_number": device_sn,  # Optional, can be None
            }

            # Convert device config to attribute entries
            table_attribute_entry = [
                {
                    **table_entry,
                    "attribute_name": attr_name,
                    "attribute_value": attr_value,
                }
                for attr_name, attr_value in device_config.items()
                if not isinstance(attr_value, (dict, list))
            ]

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
            # If the same device_name is currently installed, check for changes in configuration.
            current_device_query = table - table.RemovalTime & experiment_key & device_key

            if current_device_query:
                current_device_config: list[dict] = (table.Attribute & current_device_query).fetch(
                    "experiment_name",
                    "device_name",
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

        device_removal_list.extend((table - table.RemovalTime - device_list & experiment_key).fetch("KEY"))

        for device_entry in device_removal_list:
            if device_removal(device_type, device_entry):
                table.RemovalTime.insert1(device_entry)

    return set(epoch_device_types)


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
    components = snake_str.split("_")
    return "".join(word.capitalize() for word in components)


def get_device_info(rig: "BaseSchema") -> dict[str, dict]:
    """Parse Rig to extract device/stream info.

    Streams are defined as @data_reader methods directly on Device classes.
    Uses actual device instances from Rig to ensure pattern resolution works correctly.
    Also extracts reader constructor kwargs for readers that need additional arguments.

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections

    Returns:
        dict[str, dict]: Device info keyed by device_name with stream_type, stream_reader,
            stream_hash, and stream_reader_kwargs lists.
    """
    device_info: dict[str, dict] = {}

    # Collect all (device_name, device) pairs from Rig fields
    devices: list[tuple[str, Any]] = []
    for field_name in type(rig).model_fields:
        field_value = getattr(rig, field_name)
        if isinstance(field_value, dict):
            # Dict[Name, Device] collection (cameras, feeders, nest)
            devices.extend((name, dev) for name, dev in field_value.items() if _has_data_readers(type(dev)))
        elif _has_data_readers(type(field_value)):
            # Single Device field
            devices.append((field_name, field_value))

    # Extract stream info from each device
    for device_name, device in devices:
        device_info.setdefault(device_name, defaultdict(list))

        for method_name, _ in get_data_reader_methods(device.__class__):
            try:
                reader = getattr(device, method_name)
            except Exception as e:
                logger.warning(f"Failed to access {method_name} on {device_name}: {e}. Skipping...")
                continue

            stream_type = to_pascal_case(method_name)
            stream_reader_class = f"{reader.__class__.__module__}.{reader.__class__.__name__}"
            stream_reader_kwargs = _extract_kwargs_from_reader(reader)

            device_info[device_name]["stream_type"].append(stream_type)
            device_info[device_name]["stream_reader"].append(stream_reader_class)
            device_info[device_name]["stream_reader_kwargs"].append(stream_reader_kwargs)
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({"stream_type": stream_type, "stream_reader": stream_reader_class})
            )

    return device_info


def get_stream_entries(rig: "BaseSchema") -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for StreamType table.

    Args:
        rig: Rig instance (Pydantic BaseSchema) containing device collections

    Returns:
        list[dict]: List of dictionaries with stream_hash, stream_type, stream_reader, stream_reader_kwargs.
    """
    device_info = get_device_info(rig)
    return [
        {
            "stream_hash": stream_hash,
            "stream_type": stream_type,
            "stream_reader": stream_reader,
            "stream_reader_kwargs": stream_reader_kwargs,
        }
        for stream_info in device_info.values()
        for stream_type, stream_reader, stream_hash, stream_reader_kwargs in zip(
            stream_info["stream_type"],
            stream_info["stream_reader"],
            stream_info["stream_hash"],
            stream_info["stream_reader_kwargs"],
            strict=True,
        )
    ]


def get_device_mapper_from_rig(
    rig: "BaseSchema", metadata_filepath: Path
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Extract device type mapper and serial numbers from Pydantic Rig.

    Device types are derived from the class name (type(device).__name__).
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
    device_sn: dict[str, str | None] = {}

    # Iterate over Rig fields to find Device collections
    for field_name in type(rig).model_fields:
        field_value = getattr(rig, field_name)

        # Handle Dict[Name, Device] - e.g., cameras, feeders, nest
        if isinstance(field_value, dict):
            for device_name, device in field_value.items():
                if not _has_data_readers(type(device)):
                    continue

                device_type_mapper[device_name] = type(device).__name__
                # Extract serial number or port name
                device_sn[device_name] = (
                    getattr(device, "serial_number", None) or getattr(device, "port_name", None) or None
                )

        # Handle single Device instance (if any)
        elif _has_data_readers(type(field_value)):
            device_name = field_name
            device = field_value
            device_type_mapper[device_name] = type(device).__name__
            device_sn[device_name] = (
                getattr(device, "serial_number", None) or getattr(device, "port_name", None) or None
            )

    return device_type_mapper, device_sn
