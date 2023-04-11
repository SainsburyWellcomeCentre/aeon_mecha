import datetime
import json
import pathlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotmap import DotMap

from aeon.dj_pipeline import acquisition, dict_to_uuid, lab, streams, subject
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


def insert_devices(schema: DotMap, metadata_yml_filepath: Path):
    """Use dataset.schema and metadata.yml to insert into streams.DeviceType and streams.Device. Only insert device types that were defined both in the device schema (e.g., exp02) and Metadata.yml."""
    device_info: dict[dict] = get_device_info(schema)
    device_type_mapper, device_sn = get_device_mapper(schema, metadata_yml_filepath)

    # Add device type to device_info.
    device_info = {
        device_name: {
            "device_type": device_type_mapper.get(device_name, None),
            **device_info[device_name],
        }
        for device_name in device_info
    }
    # Return only a list of device types that have been inserted.
    for device_name, info in device_info.items():

        if info["device_type"]:

            streams.DeviceType.insert1(
                {
                    "device_type": info["device_type"],
                    "device_description": "",
                },
                skip_duplicates=True,
            )
            streams.DeviceType.Stream.insert(
                [
                    {
                        "device_type": info["device_type"],
                        "stream_type": e,
                    }
                    for e in info["stream_type"]
                ],
                skip_duplicates=True,
            )

            if device_sn[device_name]:
                if streams.Device & {"device_serial_number": device_sn[device_name]}:
                    continue
                streams.Device.insert1(
                    {
                        "device_serial_number": device_sn[device_name],
                        "device_type": info["device_type"],
                    },
                    skip_duplicates=True,
                )


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

    devices: dict[str, dict] = json.loads(
        json.dumps(
            epoch_config["metadata"]["Devices"], default=lambda x: x.__dict__, indent=4
        )
    )

    # devices: dict = {d.pop("Name"): d for d in devices}  # {deivce_name: device_config}

    return {
        "experiment_name": experiment_name,
        "epoch_start": epoch_start,
        "bonsai_workflow": epoch_config["workflow"],
        "commit": commit,
        "devices": devices,
        "metadata_file_path": metadata_yml_filepath,
    }


def ingest_epoch_metadata(experiment_name, metadata_yml_filepath):
    """
    work-in-progress
    Missing:
    + camera/patch location
    + patch, weightscale serial number
    """

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
        for name, value in epoch_config["devices"]["VideoController"].items()
        if name.endswith("Frequency")
    }

    # ---- Load cameras ----
    camera_list, camera_installation_list, camera_removal_list, camera_position_list = (
        [],
        [],
        [],
        [],
    )
    # Check if this is a new camera, add to lab.Camera if needed
    for device_name, device_config in epoch_config["devices"].items():
        if device_config["Type"] == "VideoSource":
            camera_key = {"camera_serial_number": device_config["SerialNumber"]}
            camera_list.append(camera_key)

            camera_installation = {
                **camera_key,
                "experiment_name": experiment_name,
                "camera_install_time": epoch_config["epoch_start"],
                "camera_description": device_name,
                "camera_sampling_rate": device_frequency_mapper[
                    device_config["TriggerFrequency"]
                ],
                "camera_gain": float(device_config["Gain"]),
                "camera_bin": int(device_config["Binning"]),
            }

            if "position" in device_config:
                camera_position = {
                    **camera_key,
                    "experiment_name": experiment_name,
                    "camera_install_time": epoch_config["epoch_start"],
                    "camera_position_x": device_config["position"]["x"],
                    "camera_position_y": device_config["position"]["y"],
                    "camera_position_z": device_config["position"]["z"],
                }
            else:
                camera_position = {
                    "camera_position_x": None,
                    "camera_position_y": None,
                    "camera_position_z": None,
                    "camera_rotation_x": None,
                    "camera_rotation_y": None,
                    "camera_rotation_z": None,
                }

            """Check if this camera is currently installed. If the same camera serial number is currently installed check for any changes in configuration. If not, skip this"""
            current_camera_query = (
                acquisition.ExperimentCamera - acquisition.ExperimentCamera.RemovalTime
                & experiment_key
                & camera_key
            )

            if current_camera_query:
                current_camera_config = current_camera_query.join(
                    acquisition.ExperimentCamera.Position, left=True
                ).fetch1()

                new_camera_config = {**camera_installation, **camera_position}
                current_camera_config.pop("camera_install_time")
                new_camera_config.pop("camera_install_time")

                if dict_to_uuid(current_camera_config) == dict_to_uuid(
                    new_camera_config
                ):
                    continue
                # Remove old camera
                camera_removal_list.append(
                    {
                        **current_camera_query.fetch1("KEY"),
                        "camera_remove_time": epoch_config["epoch_start"],
                    }
                )
            # Install new camera
            camera_installation_list.append(camera_installation)

            if "position" in device_config:
                camera_position_list.append(camera_position)
    # Remove the currently installed cameras that are absent in this config
    camera_removal_list.extend(
        (
            acquisition.ExperimentCamera
            - acquisition.ExperimentCamera.RemovalTime
            - camera_list
            & experiment_key
        ).fetch("KEY")
    )

    # ---- Load food patches ----
    patch_list, patch_installation_list, patch_removal_list, patch_position_list = (
        [],
        [],
        [],
        [],
    )

    # Check if this is a new food patch, add to lab.FoodPatch if needed
    for device_name, device_config in epoch_config["devices"].items():
        if device_config["Type"] == "Patch":

            patch_key = {
                "food_patch_serial_number": device_config.get(
                    "SerialNumber", device_config.get("PortName")
                )
            }
            patch_list.append(patch_key)
            patch_installation = {
                **patch_key,
                "experiment_name": experiment_name,
                "food_patch_install_time": epoch_config["epoch_start"],
                "food_patch_description": device_config["Name"],
                "wheel_sampling_rate": float(
                    re.search(r"\d+", device_config["SampleRate"]).group()
                ),
                "wheel_radius": float(device_config["Radius"]),
            }
            if "position" in device_config:
                patch_position = {
                    **patch_key,
                    "experiment_name": experiment_name,
                    "food_patch_install_time": epoch_config["epoch_start"],
                    "food_patch_position_x": device_config["position"]["x"],
                    "food_patch_position_y": device_config["position"]["y"],
                    "food_patch_position_z": device_config["position"]["z"],
                }
            else:
                patch_position = {
                    "food_patch_position_x": None,
                    "food_patch_position_y": None,
                    "food_patch_position_z": None,
                }

            """Check if this camera is currently installed. If the same camera serial number is currently installed, check for any changes in configuration, if not, skip this"""
            current_patch_query = (
                acquisition.ExperimentFoodPatch
                - acquisition.ExperimentFoodPatch.RemovalTime
                & experiment_key
                & patch_key
            )
            if current_patch_query:
                current_patch_config = current_patch_query.join(
                    acquisition.ExperimentFoodPatch.Position, left=True
                ).fetch1()
                new_patch_config = {**patch_installation, **patch_position}
                current_patch_config.pop("food_patch_install_time")
                new_patch_config.pop("food_patch_install_time")
                if dict_to_uuid(current_patch_config) == dict_to_uuid(new_patch_config):
                    continue
                # Remove old food patch
                patch_removal_list.append(
                    {
                        **current_patch_query.fetch1("KEY"),
                        "food_patch_remove_time": epoch_config["epoch_start"],
                    }
                )
            # Install new food patch
            patch_installation_list.append(patch_installation)
            if "position" in device_config:
                patch_position_list.append(patch_position)
        # Remove the currently installed patches that are absent in this config
        patch_removal_list.extend(
            (
                acquisition.ExperimentFoodPatch
                - acquisition.ExperimentFoodPatch.RemovalTime
                - patch_list
                & experiment_key
            ).fetch("KEY")
        )

    # ---- Load weight scales ----
    weight_scale_list, weight_scale_installation_list, weight_scale_removal_list = (
        [],
        [],
        [],
    )

    # Check if this is a new weight scale, add to lab.WeightScale if needed
    for device_name, device_config in epoch_config["devices"].items():
        if device_config["Type"] == "WeightScale":

            weight_scale_key = {
                "weight_scale_serial_number": device_config.get(
                    "SerialNumber", device_config.get("PortName")
                )
            }
            weight_scale_list.append(weight_scale_key)
            arena_key = (lab.Arena & acquisition.Experiment & experiment_key).fetch1(
                "KEY"
            )
            weight_scale_installation = {
                **weight_scale_key,
                **arena_key,
                "experiment_name": experiment_name,
                "weight_scale_install_time": epoch_config["epoch_start"],
                "nest": _weight_scale_nest,
                "weight_scale_description": device_name,
                "weight_scale_sampling_rate": float(_weight_scale_rate),
            }

            # Check if this weight scale is currently installed - if so, remove it
            current_weight_scale_query = (
                acquisition.ExperimentWeightScale
                - acquisition.ExperimentWeightScale.RemovalTime
                & experiment_key
                & weight_scale_key
            )
            if current_weight_scale_query:
                current_weight_scale_config = current_weight_scale_query.fetch1()
                new_weight_scale_config = weight_scale_installation.copy()
                current_weight_scale_config.pop("weight_scale_install_time")
                new_weight_scale_config.pop("weight_scale_install_time")
                if dict_to_uuid(current_weight_scale_config) == dict_to_uuid(
                    new_weight_scale_config
                ):
                    continue
                # Remove old weight scale
                weight_scale_removal_list.append(
                    {
                        **current_weight_scale_query.fetch1("KEY"),
                        "weight_scale_remove_time": epoch_config["epoch_start"],
                    }
                )
            # Install new weight scale
            weight_scale_installation_list.append(weight_scale_installation)
        # Remove the currently installed weight scales that are absent in this config
        weight_scale_removal_list.extend(
            (
                acquisition.ExperimentWeightScale
                - acquisition.ExperimentWeightScale.RemovalTime
                - weight_scale_list
                & experiment_key
            ).fetch("KEY")
        )

    # Insert
    def insert():
        lab.Camera.insert(camera_list, skip_duplicates=True)
        acquisition.ExperimentCamera.RemovalTime.insert(camera_removal_list)
        acquisition.ExperimentCamera.insert(camera_installation_list)
        acquisition.ExperimentCamera.Position.insert(camera_position_list)
        lab.FoodPatch.insert(patch_list, skip_duplicates=True)
        acquisition.ExperimentFoodPatch.RemovalTime.insert(patch_removal_list)
        acquisition.ExperimentFoodPatch.insert(patch_installation_list)
        acquisition.ExperimentFoodPatch.Position.insert(patch_position_list)
        lab.WeightScale.insert(weight_scale_list, skip_duplicates=True)
        acquisition.ExperimentWeightScale.RemovalTime.insert(weight_scale_removal_list)
        acquisition.ExperimentWeightScale.insert(weight_scale_installation_list)

    if acquisition.Experiment.connection.in_transaction:
        insert()
    else:
        with acquisition.Experiment.connection.transaction:
            insert()


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


# region Get stream & device information
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
    schema_json = json.dumps(schema, default=lambda x: x.__dict__, indent=4)
    schema_dict = json.loads(schema_json)

    device_info = {}

    for device_name in schema:
        if not device_name.startswith("_"):
            device_info[device_name] = defaultdict(list)
            if isinstance(schema[device_name], DotMap):
                for stream_type in schema[device_name].keys():
                    if schema[device_name][stream_type].__class__.__module__ in [
                        "aeon.io.reader",
                        "aeon.schema.foraging",
                        "aeon.schema.octagon",
                    ]:
                        device_info[device_name]["stream_type"].append(stream_type)
                        device_info[device_name]["stream_reader"].append(
                            schema[device_name][stream_type].__class__
                        )
            else:
                stream_type = schema[device_name].__class__.__name__
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["stream_reader"].append(
                    schema[device_name].__class__
                )

    """Add a kwargs such as pattern, columns, extension, dtype and hash
    e.g., {'pattern': '{pattern}_SubjectState',
            'columns': ['id', 'weight', 'event'],
            'extension': 'csv',
            'dtype': None}"""
    for device_name in device_info:
        if pattern := schema_dict[device_name].get("pattern"):
            schema_dict[device_name]["pattern"] = pattern.replace(
                device_name, "{pattern}"
            )

            # Add stream_reader_kwargs
            kwargs = schema_dict[device_name]
            device_info[device_name]["stream_reader_kwargs"].append(kwargs)
            stream_reader = device_info[device_name]["stream_reader"]
            # Add hash
            device_info[device_name]["stream_hash"].append(
                dict_to_uuid({**kwargs, "stream_reader": stream_reader})
            )

        else:
            for stream_type in device_info[device_name]["stream_type"]:
                pattern = schema_dict[device_name][stream_type]["pattern"]
                schema_dict[device_name][stream_type]["pattern"] = pattern.replace(
                    device_name, "{pattern}"
                )
                # Add stream_reader_kwargs
                kwargs = schema_dict[device_name][stream_type]
                device_info[device_name]["stream_reader_kwargs"].append(kwargs)
                stream_ind = device_info[device_name]["stream_type"].index(stream_type)
                stream_reader = device_info[device_name]["stream_reader"][stream_ind]
                # Add hash
                device_info[device_name]["stream_hash"].append(
                    dict_to_uuid({**kwargs, "stream_reader": stream_reader})
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


# endregion
