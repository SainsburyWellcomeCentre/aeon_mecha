import datetime
import json
import pathlib
import re

import numpy as np
import pandas as pd

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
    """Insert into stream.streamType table all streams in the dataset schema."""
    from dotmap import DotMap

    from aeon.schema import dataset

    schemas = [v for v in dataset.__dict__.values() if isinstance(v, DotMap)]
    for schema in schemas:
        streams.StreamType.insert_streams(schema)


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
