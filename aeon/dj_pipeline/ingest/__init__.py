import json
import re
from datetime import datetime
from pathlib import Path


def load_experiment_setup(setup_json_filepath, experiment_name):
    # TODO #47 putting imports here to avoid db connection. This function should probably not be in __init__.py. Also overlaps with `exp01_insert_meta.py`. Is that deprecated?
    from aeon.dj_pipeline import acquisition, lab

    """
    work-in-progress
    Missing:
    + camera/patch location
    + patch serial number
    + timestamps of device installation/removal
    """

    setup_json_filepath = Path(setup_json_filepath)
    file_creation_time = datetime.fromtimestamp(setup_json_filepath.stat().st_ctime)

    with open(setup_json_filepath, "r") as f:
        experiment_setup = json.load(f)

    video_controller = [
        device
        for device in experiment_setup["Devices"]
        if device["Type"] == "VideoController"
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

    with acquisition.Experiment.connection.transaction:
        # ---- Load cameras ----
        for device in experiment_setup["Devices"]:
            if device["Type"] == "VideoSource":
                camera = device
                # ---- Check if this is a new camera, add to lab.Camera if needed
                camera_key = {"camera_serial_number": camera["SerialNumber"]}
                if camera_key not in lab.Camera():
                    lab.Camera.insert1(camera_key)
                # ---- Check if this camera is currently installed
                current_camera_query = (
                    acquisition.ExperimentCamera
                    - acquisition.ExperimentCamera.RemovalTime
                    & {"experiment_name": experiment_name}
                    & camera_key
                )
                if current_camera_query:  # If the same camera is currently installed
                    if (
                        current_camera_query.fetch1("camera_install_time")
                        == file_creation_time
                    ):
                        # If it is installed at the same time as that read from this yml file
                        # then it is the same ExperimentCamera instance, no need to do anything
                        continue

                    # ---- Remove old camera
                    acquisition.ExperimentCamera.RemovalTime.insert1(
                        {
                            **current_camera_query.fetch1("KEY"),
                            "camera_remove_time": experiment_setup["start-time"],
                        }
                    )

                # ---- Install new camera
                acquisition.ExperimentCamera.insert1(
                    {
                        **camera_key,
                        "experiment_name": experiment_name,
                        "camera_install_time": file_creation_time,
                        "camera_description": camera["Name"],
                        "camera_sampling_rate": device_frequency_mapper[
                            camera["TriggerFrequency"]
                        ],
                        "camera_gain": camera["Gain"],
                        "camera_bin": camera["Binning"],
                    }
                )
                acquisition.ExperimentCamera.Position.insert1(
                    {
                        **camera_key,
                        "experiment_name": experiment_name,
                        "camera_install_time": file_creation_time,
                        "camera_position_x": camera["position"]["x"],
                        "camera_position_y": camera["position"]["y"],
                        "camera_position_z": camera["position"]["z"],
                    }
                )
            elif device["Type"] == "Patch":
                patch = device
                # ---- Load food patches ----
                # ---- Check if this is a new food patch, add to lab.FoodPatch if needed
                patch_key = {
                    "food_patch_serial_number": patch["SerialNumber"]
                    or patch["PortName"]
                }
                if patch_key not in lab.FoodPatch():
                    lab.FoodPatch.insert1(patch_key)
                # ---- Check if this food patch is currently installed - if so, remove it
                current_patch_query = (
                    acquisition.ExperimentFoodPatch
                    - acquisition.ExperimentFoodPatch.RemovalTime
                    & {"experiment_name": experiment_name}
                    & patch_key
                )
                if current_patch_query:  # If the same food-patch is currently installed
                    if (
                        current_patch_query.fetch1("food_patch_install_time")
                        == file_creation_time
                    ):
                        # If it is installed at the same time as that read from this yml file
                        # then it is the same ExperimentFoodPatch instance, no need to do anything
                        continue

                    # ---- Remove old food patch
                    acquisition.ExperimentFoodPatch.RemovalTime.insert1(
                        {
                            **current_patch_query.fetch1("KEY"),
                            "food_patch_remove_time": file_creation_time,
                        }
                    )

                # ---- Install new food patch
                acquisition.ExperimentFoodPatch.insert1(
                    {
                        **patch_key,
                        "experiment_name": experiment_name,
                        "food_patch_install_time": file_creation_time,
                        "food_patch_description": patch["Name"],
                        "wheel_sampling_rate": float(
                            re.search(r"\d+", patch["SampleRate"]).group()
                        ),
                        "wheel_radius": float(patch["Radius"]),
                    }
                )
                acquisition.ExperimentFoodPatch.Position.insert1(
                    {
                        **patch_key,
                        "experiment_name": experiment_name,
                        "food_patch_install_time": file_creation_time,
                        "food_patch_position_x": patch["position"]["x"],
                        "food_patch_position_y": patch["position"]["y"],
                        "food_patch_position_z": patch["position"]["z"],
                    }
                )


#
