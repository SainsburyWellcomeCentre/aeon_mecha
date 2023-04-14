import datetime
import inspect
import json
import pathlib
import re
from collections import defaultdict
from pathlib import Path

import datajoint as dj
import numpy as np
import pandas as pd
from dotmap import DotMap

from aeon.dj_pipeline import (
    acquisition,
    dict_to_uuid,
    get_schema_name,
    lab,
    streams,
    subject,
)
from aeon.io import api as io_api

_weight_scale_rate = 100
_weight_scale_nest = 1
_colony_csv_path = pathlib.Path("/ceph/aeon/aeon/colony/colony.csv")


def ingest_subject(colony_csv_path: pathlib.Path = _colony_csv_path) -> None:
    """Ingest subject information from the colony.csv file"""
    colony_df = pd.read_csv(colony_csv_path, skiprows=[1, 2])
    colony_df.rename(columns={"Id": "subject"}, inplace=True)
    colony_df["sex"] = "U"
    colony_df["subject_birth_date"] = "2021-01-01"
    colony_df["subject_description"] = ""
    subject.Subject.insert(colony_df, skip_duplicates=True, ignore_extra_fields=True)
    acquisition.Experiment.Subject.insert(
        (subject.Subject * acquisition.Experiment).proj(), skip_duplicates=True
    )


def ingest_streams():
    """Insert into streams.streamType table all streams in the dataset schema."""
    from aeon.schema import dataset

    schemas = [v for v in dataset.__dict__.values() if isinstance(v, DotMap)]
    for schema in schemas:

        stream_entries = get_stream_entries(schema)

        for entry in stream_entries:
            q_param = streams.StreamType & {"stream_hash": entry["stream_hash"]}
            if q_param:  # If the specified stream type already exists
                pname = q_param.fetch1("stream_type")
                if pname != entry["stream_type"]:
                    # If the existed stream type does not have the same name:
                    # human error, trying to add the same content with different name
                    raise dj.DataJointError(
                        f"The specified stream type already exists - name: {pname}"
                    )

        streams.StreamType.insert(stream_entries, skip_duplicates=True)


def ingest_devices(schema: DotMap, metadata_yml_filepath: Path):
    """Use dataset.schema and metadata.yml to insert into streams.DeviceType and streams.Device. Only insert device types that were defined both in the device schema (e.g., exp02) and Metadata.yml. It then creates new device tables under streams schema."""
    device_info: dict[dict] = get_device_info(schema)
    device_type_mapper, device_sn = get_device_mapper(schema, metadata_yml_filepath)

    # Add device type to device_info. Only add if device types that are defined in Metadata.yml
    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name),
            **device_info[device_name],
        }
        for device_name in device_info
        if device_type_mapper.get(device_name)
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
        for device_type in device_stream_map.keys()
        if not streams.DeviceType & {"device_type": device_type}
    ]

    new_device_stream_types = [
        {"device_type": device_type, "stream_type": stream_type}
        for device_type, stream_list in device_stream_map.items()
        for stream_type in stream_list
        if not streams.DeviceType.Stream
        & {"device_type": device_type, "stream_type": stream_type}
    ]

    new_devices = [
        {
            "device_serial_number": device_sn[device_name],
            "device_type": device_config["device_type"],
        }
        for device_name, device_config in device_info.items()
        if device_sn[device_name]
        and not streams.Device & {"device_serial_number": device_sn[device_name]}
    ]

    # Insert new entries.
    if new_device_types:
        streams.DeviceType.insert(new_device_types)

    if new_device_stream_types:
        streams.DeviceType.Stream.insert(new_device_stream_types)

    if new_devices:
        streams.Device.insert(new_devices)

    # Create tables.
    context = inspect.currentframe().f_back.f_locals

    for device_info in new_device_types:
        table_class = streams.get_device_template(device_info["device_type"])
        context[table_class.__name__] = table_class
        streams.schema(table_class, context=context)

    # Create device_type tables
    for device_info in new_device_stream_types:
        table_class = streams.get_device_stream_template(
            device_info["device_type"], device_info["stream_type"]
        )
        context[table_class.__name__] = table_class
        streams.schema(table_class, context=context)

    streams.schema.activate(streams.schema_name, add_objects=context)
    vm = dj.VirtualModule(streams.schema_name, streams.schema_name)
    for k, v in vm.__dict__.items():
        if "Table" in str(v.__class__):
            streams.__dict__[k] = v


def extract_epoch_config(experiment_name: str, metadata_yml_filepath: str) -> dict:
    """Parse experiment metadata YAML file and extract epoch configuration.

    Args:
        experiment_name (str)
        metadata_yml_filepath (str)

    Returns:
        dict: epoch_config [dict]
    """
    metadata_yml_filepath = pathlib.Path(metadata_yml_filepath)
    epoch_start = datetime.datetime.strptime(
        metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S"
    )
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
        json.dumps(
            epoch_config["metadata"]["Devices"], default=lambda x: x.__dict__, indent=4
        )
    )

    devices: dict = {
        d.pop("Name"): d for d in devices
    }  # {deivce_name: device_config}  #! may not work for presocial

    return {
        "experiment_name": experiment_name,
        "epoch_start": epoch_start,
        "bonsai_workflow": epoch_config["workflow"],
        "commit": commit,
        "metadata": devices,  #! this format might have changed since using aeon metadata reader
        "metadata_file_path": metadata_yml_filepath,
    }


def ingest_epoch_metadata(experiment_name, metadata_yml_filepath):
    """
    work-in-progress
    Missing:
    + camera/patch location
    + patch, weightscale serial number
    """
    from aeon.dj_pipeline import streams

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

    device_frequency_mapper = {
        name: float(value)
        for name, value in epoch_config["metadata"]["VideoController"].items()
        if name.endswith("Frequency")
    }  # May not be needed?

    # Insert into each device table
    for device_name, device_config in epoch_config["metadata"].items():
        if table := getattr(streams, device_config["Type"], None):
            device_sn = device_config.get("SerialNumber", device_config.get("PortName"))
            device_key = {"device_serial_number": device_sn}

            if not (
                table
                & {
                    "experiment_name": experiment_name,
                    "device_serial_number": device_sn,
                }
            ):

                table_entry = {
                    "experiment_name": experiment_name,
                    "device_serial_number": device_sn,
                    f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config[
                        "epoch_start"
                    ],
                    f"{dj.utils.from_camel_case(table.__name__)}_name": device_name,
                }

                table_attribute_entry = [
                    {
                        "experiment_name": experiment_name,
                        "device_serial_number": device_sn,
                        f"{dj.utils.from_camel_case(table.__name__)}_install_time": epoch_config[
                            "epoch_start"
                        ],
                        f"{dj.utils.from_camel_case(table.__name__)}_name": device_name,
                        "attribute_name": attribute_name,
                        "attribute_value": attribute_value,
                    }
                    for attribute_name, attribute_value in device_config.items()
                ]

                """Check if this camera is currently installed. If the same camera serial number is currently installed check for any changes in configuration. If not, skip this"""
                current_device_query = (
                    table.Attribute - table.RemovalTime & experiment_key & device_key
                )

                if current_device_query:
                    current_device_config: list[dict] = current_device_query.fetch(
                        "experiment_name",
                        "device_serial_number",
                        "attribute_name",
                        "attribute_value",
                        as_dict=True,
                    )
                    new_device_config: list[dict] = [
                        {
                            k: v
                            for k, v in entry.items()
                            if k
                            != f"{dj.utils.from_camel_case(table.__name__)}_install_time"
                        }
                        for entry in table_attribute_entry
                    ]

                    if dict_to_uuid(current_device_config) == dict_to_uuid(
                        new_device_config
                    ):  # Skip if none of the configuration has changed.
                        continue

                    # Remove old device
                    table_removal_entry = [
                        {
                            **entry,
                            f"{dj.utils.from_camel_case(table.__name__)}_removal_time": epoch_config[
                                "epoch_start"
                            ],
                        }
                        for entry in current_device_config
                    ]

                # Insert into table.
                with table.connection.in_transaction:
                    table.insert1(table_entry)
                    table.Attribute.insert(table_attribute_entry)
                    table.RemovalTime.insert(table_removal_entry)

    # Remove the currently installed devices that are absent in this config
    device_removal_list.extend(
        (table - table.RemovalTime - device_list & experiment_key).fetch("KEY")
    )

    # Insert
    # def insert():
    #     lab.Camera.insert(camera_list, skip_duplicates=True)
    #     acquisition.ExperimentCamera.RemovalTime.insert(camera_removal_list)
    #     acquisition.ExperimentCamera.insert(camera_installation_list)
    #     acquisition.ExperimentCamera.Position.insert(camera_position_list)
    #     lab.FoodPatch.insert(patch_list, skip_duplicates=True)
    #     acquisition.ExperimentFoodPatch.RemovalTime.insert(patch_removal_list)
    #     acquisition.ExperimentFoodPatch.insert(patch_installation_list)
    #     acquisition.ExperimentFoodPatch.Position.insert(patch_position_list)
    #     lab.WeightScale.insert(weight_scale_list, skip_duplicates=True)
    #     acquisition.ExperimentWeightScale.RemovalTime.insert(weight_scale_removal_list)
    #     acquisition.ExperimentWeightScale.insert(weight_scale_installation_list)

    # if acquisition.Experiment.connection.in_transaction:
    #     insert()
    # else:
    #     with acquisition.Experiment.connection.transaction:
    #         insert()


# region Get stream & device information
def get_stream_entries(schema: DotMap) -> list[dict]:
    """Returns a list of dictionaries containing the stream entries for a given device,

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
        )
    ]


def get_device_info(schema: DotMap) -> dict[dict]:
    """
    Read from the above DotMap object and returns a device dictionary as the following.

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
                    device_info[device_name]["stream_reader"].append(
                        _get_class_path(stream_obj)
                    )

                    required_args = [
                        k
                        for k in inspect.signature(stream_obj.__init__).parameters
                        if k != "self"
                    ]
                    pattern = schema_dict[device_name][stream_type].get("pattern")
                    schema_dict[device_name][stream_type]["pattern"] = pattern.replace(
                        device_name, "{pattern}"
                    )

                    kwargs = {
                        k: v
                        for k, v in schema_dict[device_name][stream_type].items()
                        if k in required_args
                    }
                    device_info[device_name]["stream_reader_kwargs"].append(kwargs)
                    # Add hash
                    device_info[device_name]["stream_hash"].append(
                        dict_to_uuid(
                            {**kwargs, "stream_reader": _get_class_path(stream_obj)}
                        )
                    )
        else:
            stream_type = device.__class__.__name__
            device_info[device_name]["stream_type"].append(stream_type)
            device_info[device_name]["stream_reader"].append(_get_class_path(device))

            required_args = {
                k: None
                for k in inspect.signature(device.__init__).parameters
                if k != "self"
            }
            pattern = schema_dict[device_name].get("pattern")
            schema_dict[device_name]["pattern"] = pattern.replace(
                device_name, "{pattern}"
            )

            kwargs = {
                k: v for k, v in schema_dict[device_name].items() if k in required_args
            }
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
    import os

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
    repository_root = (
        os.popen("git rev-parse --show-toplevel").read().strip()
    )  # repo root path
    filename = Path(
        repository_root + "/aeon/dj_pipeline/create_experiments/device_type_mapper.json"
    )

    device_type_mapper = {}  # {device_name: device_type}
    device_sn = {}  # device serial number

    if filename.is_file():
        with filename.open("r") as f:
            device_type_mapper = json.load(f)

    try:  # if the device type is not in the mapper, add it
        for item in meta_data.Devices:
            device_type_mapper[item.Name] = item.Type
            device_sn[item.Name] = (
                item.SerialNumber if not isinstance(item.SerialNumber, DotMap) else None
            )
        with filename.open("w") as f:
            json.dump(device_type_mapper, f)
    except AttributeError:
        pass

    return device_type_mapper, device_sn


def ingest_epoch_metadata_octagon(experiment_name, metadata_yml_filepath):
    """
    Temporary ingestion routine to load devices' meta information for Octagon arena experiments
    """
    from aeon.dj_pipeline import streams

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

    epoch_start = datetime.datetime.strptime(
        metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S"
    )

    for device_idx, (device_name, device_type) in enumerate(oct01_devices):
        device_sn = f"oct01_{device_idx}"
        streams.Device.insert1(
            {"device_serial_number": device_sn, "device_type": device_type},
            skip_duplicates=True,
        )
        experiment_table = getattr(streams, f"Experiment{device_type}")
        if not (
            experiment_table
            & {"experiment_name": experiment_name, "device_serial_number": device_sn}
        ):
            experiment_table.insert1(
                (experiment_name, device_sn, epoch_start, device_name)
            )


# endregion
