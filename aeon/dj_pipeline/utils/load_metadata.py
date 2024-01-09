import datetime
import inspect
import json
import pathlib
from collections import defaultdict
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
from dotmap import DotMap

from aeon.dj_pipeline import acquisition, dict_to_uuid, subject
from aeon.dj_pipeline.utils import streams_maker
from aeon.io import api as io_api

logger = dj.logger
_weight_scale_rate = 100
_weight_scale_nest = 1
_colony_csv_path = pathlib.Path("/ceph/aeon/aeon/colony/colony.csv")
_aeon_schemas = ["social01"]


def ingest_subject(colony_csv_path: pathlib.Path = _colony_csv_path) -> None:
    """Ingest subject information from the colony.csv file."""
    logger.warning("The use of 'colony.csv' is deprecated starting Nov 2023", DeprecationWarning)

    colony_df = pd.read_csv(colony_csv_path, skiprows=[1, 2])
    colony_df.rename(columns={"Id": "subject"}, inplace=True)
    colony_df["sex"] = "U"
    colony_df["subject_birth_date"] = "2021-01-01"
    colony_df["subject_description"] = ""
    subject.Subject.insert(colony_df, skip_duplicates=True, ignore_extra_fields=True)
    acquisition.Experiment.Subject.insert(
        (subject.Subject * acquisition.Experiment).proj(), skip_duplicates=True
    )


def insert_stream_types():
    """Insert into streams.streamType table all streams in the aeon schemas."""
    from aeon.schema import schemas as aeon_schemas

    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    schemas = [getattr(aeon_schemas, aeon_schema) for aeon_schema in _aeon_schemas]
    for schema in schemas:
        stream_entries = get_stream_entries(schema)

        for entry in stream_entries:
            q_param = streams.StreamType & {"stream_hash": entry["stream_hash"]}
            if q_param:  # If the specified stream type already exists
                pname = q_param.fetch1("stream_type")
                if pname != entry["stream_type"]:
                    # If the existed stream type does not have the same name:
                    # human error, trying to add the same content with different name
                    raise dj.DataJointError(f"The specified stream type already exists - name: {pname}")

        streams.StreamType.insert(stream_entries, skip_duplicates=True)


def insert_device_types(device_schema: DotMap, metadata_yml_filepath: Path):
    """
    Use aeon.schema.schemas and metadata.yml to insert into streams.DeviceType and streams.Device.
    Only insert device types that were defined both in the device schema (e.g., exp02) and Metadata.yml.
    It then creates new device tables under streams schema.
    """
    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    device_info: dict[dict] = get_device_info(device_schema)
    device_type_mapper, device_sn = get_device_mapper(device_schema, metadata_yml_filepath)

    # Add device type to device_info. Only add if device types that are defined in Metadata.yml
    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name),
            **device_info[device_name],
        }
        for device_name in device_info
        if device_type_mapper.get(device_name) and device_sn.get(device_name)
    }

    # Create a map of device_type to stream_type.
    device_stream_map: dict[list] = {}

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


def extract_epoch_config(experiment_name: str, metadata_yml_filepath: str) -> dict:
    """Parse experiment metadata YAML file and extract epoch configuration.

    Args:
        experiment_name (str): Name of the experiment.
        metadata_yml_filepath (str): path to the metadata YAML file.

    Returns:
        dict: epoch_config [dict]
    """
    metadata_yml_filepath = pathlib.Path(metadata_yml_filepath)
    epoch_start = datetime.datetime.strptime(metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S")
    epoch_config: dict = (
        io_api.load(
            str(metadata_yml_filepath.parent),
            acquisition._device_schema_mapping[experiment_name].Metadata,
        )
        .reset_index()
        .to_dict("records")[0]
    )

    commit = epoch_config.get("commit")
    if isinstance(commit, float) and np.isnan(commit):
        commit = epoch_config["metadata"]["Revision"]

    assert commit, f'Neither "Commit" nor "Revision" found in {metadata_yml_filepath}'

    devices: list[dict] = json.loads(
        json.dumps(epoch_config["metadata"]["Devices"], default=lambda x: x.__dict__, indent=4)
    )

    if isinstance(devices, list):  # In exp02, it is a list of dict. In presocial. It's a dict of dict.
        devices: dict = {d.pop("Name"): d for d in devices}  # {deivce_name: device_config}

    return {
        "experiment_name": experiment_name,
        "epoch_start": epoch_start,
        "bonsai_workflow": epoch_config["workflow"],
        "commit": commit,
        "metadata": devices,  #! this format might have changed since using aeon metadata reader
        "metadata_file_path": metadata_yml_filepath,
    }


def ingest_epoch_metadata(experiment_name, metadata_yml_filepath):
    """Make entries into device tables."""
    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    if experiment_name.startswith("oct"):
        ingest_epoch_metadata_octagon(experiment_name, metadata_yml_filepath)
        return

    experiment_key = {"experiment_name": experiment_name}
    metadata_yml_filepath = pathlib.Path(metadata_yml_filepath)
    epoch_config = extract_epoch_config(experiment_name, metadata_yml_filepath)

    previous_epoch = (acquisition.Experiment & experiment_key).aggr(
        acquisition.Epoch & f'epoch_start < "{epoch_config["epoch_start"]}"',
        epoch_start="MAX(epoch_start)",
    )
    if len(acquisition.Epoch.Config & previous_epoch) and epoch_config["commit"] == (
        acquisition.Epoch.Config & previous_epoch
    ).fetch1("commit"):
        # if identical commit -> no changes
        return

    device_schema = acquisition._device_schema_mapping[experiment_name]
    device_type_mapper, _ = get_device_mapper(device_schema, metadata_yml_filepath)

    # Insert into each device table
    epoch_device_types = []
    device_list = []
    device_removal_list = []

    for device_name, device_config in epoch_config["metadata"].items():
        if table := getattr(streams, device_type_mapper.get(device_name) or "", None):
            device_sn = device_config.get("SerialNumber", device_config.get("PortName"))
            device_key = {"device_serial_number": device_sn}

            if not (streams.Device & device_key):
                logger.warning(
                    f"Device {device_name} (serial number: {device_sn}) is not yet registered in streams.Device. Skipping..."
                )
                # skip if this device (with a serial number) is not yet inserted in streams.Device
                continue

            device_list.append(device_key)
            epoch_device_types.append(table.__name__)

            table_entry = {
                "experiment_name": experiment_name,
                **device_key,
                f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config["epoch_start"],
                f"{dj.utils.from_camel_case(table.__name__)}_name": device_name,
            }

            table_attribute_entry = [
                {
                    **table_entry,
                    "attribute_name": attribute_name,
                    "attribute_value": attribute_value,
                }
                for attribute_name, attribute_value in device_config.items()
            ]

            """Check if this device is currently installed. If the same device serial number is currently installed check for any changes in configuration. If not, skip this"""
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
        )  # could be VideoSource or Patch

        for device_entry in device_removal_list:
            if device_removal(device_type, device_entry):
                table.RemovalTime.insert1(device_entry)

    return set(epoch_device_types)


# region Get stream & device information
def get_stream_entries(schema: DotMap) -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for a given device.

    Args:
        schema (DotMap): DotMap object (e.g., exp02, octagon01)

    Returns:
    stream_info (list[dict]): list of dictionaries containing the stream entries for a given device,

        e.g. {'stream_type': 'EnvironmentState',
        'stream_reader': aeon.io.reader.Csv,
        'stream_reader_kwargs': {'pattern': '{pattern}_EnvironmentState',
        'columns': ['state'],
        'extension': 'csv',
        'dtype': None}
    """
    device_info = get_device_info(schema)
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


def get_device_info(schema: DotMap) -> dict[dict]:
    """Read from the above DotMap object and returns a device dictionary as the following.

    Args:
        schema (DotMap): DotMap object (e.g., exp02, octagon01)

    Returns:
        device_info (dict[dict]): A dictionary of device information

        e.g.   {'CameraTop':
                    {'stream_type': ['Video', 'Position', 'Region'],
                        'stream_reader': [
                                    aeon.io.reader.Video,
                                    aeon.io.reader.Position,
                                    aeon.schema.foraging._RegionReader
                                ],
                        'pattern': ['{pattern}', '{pattern}_200', '{pattern}_201']
                    }
                }
    """

    def _get_class_path(obj):
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"

    schema_json = json.dumps(schema, default=lambda x: x.__dict__, indent=4)
    schema_dict = json.loads(schema_json)
    device_info = {}

    for device_name, device in schema.items():
        if device_name.startswith("_"):
            continue

        device_info[device_name] = defaultdict(list)

        if isinstance(device, DotMap):
            for stream_type, stream_obj in device.items():
                if stream_obj.__class__.__module__ in [
                    "aeon.io.reader",
                    "aeon.schema.foraging",
                    "aeon.schema.octagon",
                ]:
                    device_info[device_name]["stream_type"].append(stream_type)
                    device_info[device_name]["stream_reader"].append(_get_class_path(stream_obj))

                    required_args = [
                        k for k in inspect.signature(stream_obj.__init__).parameters if k != "self"
                    ]
                    pattern = schema_dict[device_name][stream_type].get("pattern")
                    schema_dict[device_name][stream_type]["pattern"] = pattern.replace(
                        device_name, "{pattern}"
                    )

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

            required_args = {k: None for k in inspect.signature(device.__init__).parameters if k != "self"}
            pattern = schema_dict[device_name].get("pattern")
            schema_dict[device_name]["pattern"] = pattern.replace(device_name, "{pattern}")

            kwargs = {k: v for k, v in schema_dict[device_name].items() if k in required_args}
            device_info[device_name]["stream_reader_kwargs"].append(kwargs)
            # Add hash
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({**kwargs, "stream_reader": _get_class_path(device)})
            )
    return device_info


def get_device_mapper(schema: DotMap, metadata_yml_filepath: Path):
    """Returns a mapping dictionary between device name and device type based on the dataset schema and metadata.yml from the experiment. Store the mapper dictionary and read from it if the type info doesn't exist in Metadata.yml.

    Args:
        schema (DotMap): DotMap object (e.g., exp02)
        metadata_yml_filepath (Path): Path to metadata.yml.

    Returns:
        device_type_mapper (dict): {"device_name", "device_type"}
         e.g. {'CameraTop': 'VideoSource', 'Patch1': 'Patch'}
        device_sn (dict): {"device_name", "serial_number"}
         e.g. {'CameraTop': '21053810'}
    """
    from aeon.io import api

    metadata_yml_filepath = Path(metadata_yml_filepath)
    meta_data = (
        api.load(
            str(metadata_yml_filepath.parent),
            schema.Metadata,
        )
        .reset_index()
        .to_dict("records")[0]["metadata"]
    )

    # Store the mapper dictionary here
    filename = Path(__file__).parent.parent / "utils/device_type_mapper.json"

    device_type_mapper = {}  # {device_name: device_type}
    device_sn = {}  # {device_name: device_sn}

    if filename.is_file():
        with filename.open("r") as f:
            device_type_mapper = json.load(f)

    try:  # if the device type is not in the mapper, add it
        for item in meta_data.Devices:
            if isinstance(item, DotMap):
                device_type_mapper[item.Name] = item.Type
                device_sn[item.Name] = (
                    item.SerialNumber or item.PortName or None
                )  # assign either the serial number (if it exists) or port name. If neither exists, assign None
            elif isinstance(item, str):  # presocial
                if meta_data.Devices[item].get("Type"):
                    device_type_mapper[item] = meta_data.Devices[item].get("Type")
                device_sn[item] = (
                    meta_data.Devices[item].get("SerialNumber")
                    or meta_data.Devices[item].get("PortName")
                    or None
                )

        with filename.open("w") as f:
            json.dump(device_type_mapper, f)
    except AttributeError:
        pass

    return device_type_mapper, device_sn


def ingest_epoch_metadata_octagon(experiment_name, metadata_yml_filepath):
    """Temporary ingestion routine to load devices' meta information for Octagon arena experiments."""
    streams = dj.VirtualModule("streams", streams_maker.schema_name)

    oct01_devices = [
        ("Metadata", "Metadata"),
        ("CameraTop", "Camera"),
        ("CameraColorTop", "Camera"),
        ("ExperimentalMetadata", "ExperimentalMetadata"),
        ("Photodiode", "Photodiode"),
        ("OSC", "OSC"),
        ("TaskLogic", "TaskLogic"),
        ("Wall1", "Wall"),
        ("Wall2", "Wall"),
        ("Wall3", "Wall"),
        ("Wall4", "Wall"),
        ("Wall5", "Wall"),
        ("Wall6", "Wall"),
        ("Wall7", "Wall"),
        ("Wall8", "Wall"),
    ]

    epoch_start = datetime.datetime.strptime(metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S")

    for device_idx, (device_name, device_type) in enumerate(oct01_devices):
        device_sn = f"oct01_{device_idx}"
        streams.Device.insert1(
            {"device_serial_number": device_sn, "device_type": device_type},
            skip_duplicates=True,
        )
        experiment_table = getattr(streams, f"Experiment{device_type}")
        if not (experiment_table & {"experiment_name": experiment_name, "device_serial_number": device_sn}):
            experiment_table.insert1((experiment_name, device_sn, epoch_start, device_name))


# endregion
