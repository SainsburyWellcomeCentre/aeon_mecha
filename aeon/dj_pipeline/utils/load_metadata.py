import datetime
import pathlib
import re

import numpy as np
import pandas as pd
import yaml

from aeon.dj_pipeline import acquisition, dict_to_uuid, lab, subject
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


def extract_epoch_metadata(experiment_name, metadata_yml_filepath):
    metadata_yml_filepath = pathlib.Path(metadata_yml_filepath)
    epoch_start = datetime.datetime.strptime(
        metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S"
    )
    experiment_setup: dict = (
        io_api.load(
            str(metadata_yml_filepath.parent),
            acquisition._device_schema_mapping[experiment_name].Metadata,
        )
        .reset_index()
        .to_dict("records")[0]
    )

    commit = experiment_setup.get("commit")
    if isinstance(commit, float) and np.isnan(commit):
        commit = experiment_setup["metadata"]["Revision"]
    else:
        commit = None

    assert commit, f'Neither "Commit" nor "Revision" found in {metadata_yml_filepath}'
    return {
        "experiment_name": experiment_name,
        "epoch_start": epoch_start,
        "bonsai_workflow": experiment_setup["workflow"],
        "commit": commit,
        "metadata": experiment_setup,
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

    metadata_yml_filepath = pathlib.Path(metadata_yml_filepath)
    epoch_start = datetime.datetime.strptime(
        metadata_yml_filepath.parent.name, "%Y-%m-%dT%H-%M-%S"
    )

    experiment_setup = io_api.load(
        str(metadata_yml_filepath.parent),
        acquisition._device_schema_mapping[experiment_name].Metadata,
    )

    experiment_key = {"experiment_name": experiment_name}
    # Check if there has been any changes in the arena setup
    # by comparing the "Commit" against the most immediate preceding epoch
    commit = experiment_setup.get("Commit", experiment_setup.get("Revision"))
    assert commit, f'Neither "Commit" nor "Revision" found in {metadata_yml_filepath}'
    previous_epoch = (acquisition.Experiment & experiment_key).aggr(
        acquisition.Epoch & f'epoch_start < "{epoch_start}"',
        epoch_start="MAX(epoch_start)",
    )
    if len(acquisition.Epoch.Config & previous_epoch) and commit == (
        acquisition.Epoch.Config & previous_epoch
    ).fetch1("commit"):
        # if identical commit -> no changes
        return
    if isinstance(experiment_setup["Devices"], list):
        experiment_devices = experiment_setup.pop("Devices")
    elif isinstance(experiment_setup["Devices"], dict):
        experiment_devices = []
        for device_name, device_info in experiment_setup.pop("Devices").items():
            if device_name.startswith("VideoController"):
                device_type = "VideoController"
            elif all(v in device_info for v in ("TriggerFrequency", "FrameEvents")):
                device_type = "VideoSource"
            elif all(v in device_info for v in ("PelletDelivered", "PatchEvents")):
                device_type = "Patch"
            elif all(v in device_info for v in ("TareWeight", "WeightEvents")):
                device_type = "WeightScale"
            elif device_name.startswith("AudioAmbient"):
                device_type = "AudioAmbient"
            elif device_name.startswith("Wall"):
                device_type = "Wall"
            elif device_name.startswith("Photodiode"):
                device_type = "Photodiode"
            else:
                raise ValueError(f"Unrecognized Device Type for {device_name}")
            experiment_devices.append(
                {"Name": device_name, "Type": device_type, **device_info}
            )
    else:
        raise ValueError(
            f"Unexpected devices variable type: {type(experiment_setup['Devices'])}"
        )
    # ---- Video Controller ----
    video_controller = [
        device for device in experiment_devices if device["Type"] == "VideoController"
    ]
    assert (
        len(video_controller) == 1
    ), "Unable to find one unique VideoController device"
    video_controller = video_controller[0]
    device_frequency_mapper = {
        name: float(value)
        for name, value in video_controller.items()
        if name.endswith("Frequency")
    }
    # ---- Load cameras ----
    cameras = [
        device for device in experiment_devices if device["Type"] == "VideoSource"
    ]
    camera_list, camera_installation_list, camera_removal_list, camera_position_list = (
        [],
        [],
        [],
        [],
    )
    for camera in cameras:
        # ---- Check if this is a new camera, add to lab.Camera if needed
        camera_key = {"camera_serial_number": camera["SerialNumber"]}
        camera_list.append(camera_key)
        camera_installation = {
            "experiment_name": experiment_name,
            **camera_key,
            "camera_install_time": epoch_start,
            "camera_description": camera["Name"],
            "camera_sampling_rate": device_frequency_mapper[camera["TriggerFrequency"]],
            "camera_gain": float(camera["Gain"]),
            "camera_bin": int(camera["Binning"]),
        }
        if "position" in camera:
            camera_position = {
                **camera_key,
                "experiment_name": experiment_name,
                "camera_install_time": epoch_start,
                "camera_position_x": camera["position"]["x"],
                "camera_position_y": camera["position"]["y"],
                "camera_position_z": camera["position"]["z"],
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
        # ---- Check if this camera is currently installed
        # If the same camera serial number is currently installed
        # check for any changes in configuration, if not, skip this
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
            if dict_to_uuid(current_camera_config) == dict_to_uuid(new_camera_config):
                continue
            # ---- Remove old camera
            camera_removal_list.append(
                {
                    **current_camera_query.fetch1("KEY"),
                    "camera_remove_time": epoch_start,
                }
            )
        # ---- Install new camera
        camera_installation_list.append(camera_installation)
        if "position" in camera:
            camera_position_list.append(camera_position)
    # remove the currently installed cameras that are absent in this config
    camera_removal_list.extend(
        (
            acquisition.ExperimentCamera
            - acquisition.ExperimentCamera.RemovalTime
            - camera_list
            & experiment_key
        ).fetch("KEY")
    )
    # ---- Load food patches ----
    food_patches = [
        device for device in experiment_devices if device["Type"] == "Patch"
    ]
    patch_list, patch_installation_list, patch_removal_list, patch_position_list = (
        [],
        [],
        [],
        [],
    )
    for patch in food_patches:
        # ---- Check if this is a new food patch, add to lab.FoodPatch if needed
        patch_key = {
            "food_patch_serial_number": patch.get("SerialNumber") or patch["PortName"]
        }
        patch_list.append(patch_key)
        patch_installation = {
            **patch_key,
            "experiment_name": experiment_name,
            "food_patch_install_time": epoch_start,
            "food_patch_description": patch["Name"],
            "wheel_sampling_rate": float(
                re.search(r"\d+", patch["SampleRate"]).group()
            ),
            "wheel_radius": float(patch["Radius"]),
        }
        if "position" in patch:
            patch_position = {
                **patch_key,
                "experiment_name": experiment_name,
                "food_patch_install_time": epoch_start,
                "food_patch_position_x": patch["position"]["x"],
                "food_patch_position_y": patch["position"]["y"],
                "food_patch_position_z": patch["position"]["z"],
            }
        else:
            patch_position = {
                "food_patch_position_x": None,
                "food_patch_position_y": None,
                "food_patch_position_z": None,
            }
        # ---- Check if this camera is currently installed
        # If the same camera serial number is currently installed
        # check for any changes in configuration, if not, skip this
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
            # ---- Remove old food patch
            patch_removal_list.append(
                {
                    **current_patch_query.fetch1("KEY"),
                    "food_patch_remove_time": epoch_start,
                }
            )
        # ---- Install new food patch
        patch_installation_list.append(patch_installation)
        if "position" in patch:
            patch_position_list.append(patch_position)
    # remove the currently installed patches that are absent in this config
    patch_removal_list.extend(
        (
            acquisition.ExperimentFoodPatch
            - acquisition.ExperimentFoodPatch.RemovalTime
            - patch_list
            & experiment_key
        ).fetch("KEY")
    )
    # ---- Load weight scales ----
    weight_scales = [
        device for device in experiment_devices if device["Type"] == "WeightScale"
    ]
    weight_scale_list, weight_scale_installation_list, weight_scale_removal_list = (
        [],
        [],
        [],
    )
    for weight_scale in weight_scales:
        # ---- Check if this is a new weight scale, add to lab.WeightScale if needed
        weight_scale_key = {
            "weight_scale_serial_number": weight_scale.get("SerialNumber")
            or weight_scale["PortName"]
        }
        weight_scale_list.append(weight_scale_key)
        arena_key = (lab.Arena & acquisition.Experiment & experiment_key).fetch1("KEY")
        weight_scale_installation = {
            "experiment_name": experiment_name,
            **weight_scale_key,
            "weight_scale_install_time": epoch_start,
            **arena_key,
            "nest": _weight_scale_nest,
            "weight_scale_description": weight_scale["Name"],
            "weight_scale_sampling_rate": float(_weight_scale_rate),
        }
        # ---- Check if this weight scale is currently installed - if so, remove it
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
            # ---- Remove old weight scale
            weight_scale_removal_list.append(
                {
                    **current_weight_scale_query.fetch1("KEY"),
                    "weight_scale_remove_time": epoch_start,
                }
            )
        # ---- Install new weight scale
        weight_scale_installation_list.append(weight_scale_installation)
    # remove the currently installed weight scales that are absent in this config
    weight_scale_removal_list.extend(
        (
            acquisition.ExperimentWeightScale
            - acquisition.ExperimentWeightScale.RemovalTime
            - weight_scale_list
            & experiment_key
        ).fetch("KEY")
    )
    # ---- insert ----
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
