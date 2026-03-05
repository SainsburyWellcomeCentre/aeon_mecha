"""Functions to populate the database with the metadata for experiment 0.1."""

import pathlib

import yaml

from aeon.dj_pipeline import acquisition, lab, subject

_wheel_sampling_rate = 500
_weight_scale_rate = 100


def ingest_exp01_metadata(metadata_yml_filepath, experiment_name):
    """Ingest metadata from a yml file into the database for experiment 0.1."""
    with open(metadata_yml_filepath) as f:
        arena_setup = yaml.full_load(f)

    device_frequency_mapper = {
        name.replace("frequency", "").replace("-", ""): value
        for name, value in arena_setup["video-controller"].items()
        if name.endswith("frequency")
    }

    with acquisition.Experiment.connection.transaction:
        # ---- Load cameras ----
        for camera in arena_setup["cameras"]:
            # ---- Check if this is a new camera, add to lab.Camera if needed
            camera_key = {"camera_serial_number": camera["serial-number"]}
            if camera_key not in lab.Camera():
                lab.Camera.insert1(camera_key)
            # ---- Check if this camera is currently installed
            current_camera_query = (
                acquisition.ExperimentCamera - acquisition.ExperimentCamera.RemovalTime
                & {"experiment_name": experiment_name}
                & camera_key
            )
            if current_camera_query:  # If the same camera is currently installed
                if current_camera_query.fetch1("camera_install_time") == arena_setup["start-time"]:
                    # If it is installed at the same time as that read from this yml file
                    # then it is the same ExperimentCamera instance, no need to do anything
                    continue

                # ---- Remove old camera
                acquisition.ExperimentCamera.RemovalTime.insert1(
                    {
                        **current_camera_query.fetch1("KEY"),
                        "camera_remove_time": arena_setup["start-time"],
                    }
                )

            # ---- Install new camera
            acquisition.ExperimentCamera.insert1(
                {
                    **camera_key,
                    "experiment_name": experiment_name,
                    "camera_install_time": arena_setup["start-time"],
                    "camera_description": camera["description"],
                    "camera_sampling_rate": device_frequency_mapper[camera["trigger-source"].lower()],
                }
            )
            acquisition.ExperimentCamera.Position.insert1(
                {
                    **camera_key,
                    "experiment_name": experiment_name,
                    "camera_install_time": arena_setup["start-time"],
                    "camera_position_x": camera["position"]["x"],
                    "camera_position_y": camera["position"]["y"],
                    "camera_position_z": camera["position"]["z"],
                }
            )
        # ---- Load food patches ----
        for patch in arena_setup["patches"]:
            # ---- Check if this is a new food patch, add to lab.FoodPatch if needed
            patch_key = {"food_patch_serial_number": patch["serial-number"] or patch["port-name"]}
            if patch_key not in lab.FoodPatch():
                lab.FoodPatch.insert1(patch_key)
            # ---- Check if this food patch is currently installed - if so, remove it
            current_patch_query = (
                acquisition.ExperimentFoodPatch - acquisition.ExperimentFoodPatch.RemovalTime
                & {"experiment_name": experiment_name}
                & patch_key
            )
            if current_patch_query:  # If the same food-patch is currently installed
                if current_patch_query.fetch1("food_patch_install_time") == arena_setup["start-time"]:
                    # If it is installed at the same time as that read from this yml file
                    # then it is the same ExperimentFoodPatch instance, no need to do anything
                    continue

                # ---- Remove old food patch
                acquisition.ExperimentFoodPatch.RemovalTime.insert1(
                    {
                        **current_patch_query.fetch1("KEY"),
                        "food_patch_remove_time": arena_setup["start-time"],
                    }
                )

            # ---- Install new food patch
            acquisition.ExperimentFoodPatch.insert1(
                {
                    **patch_key,
                    "experiment_name": experiment_name,
                    "food_patch_install_time": arena_setup["start-time"],
                    "food_patch_description": patch["description"],
                    "wheel_sampling_rate": _wheel_sampling_rate,
                }
            )
            acquisition.ExperimentFoodPatch.Position.insert1(
                {
                    **patch_key,
                    "experiment_name": experiment_name,
                    "food_patch_install_time": arena_setup["start-time"],
                    "food_patch_position_x": patch["position"]["x"],
                    "food_patch_position_y": patch["position"]["y"],
                    "food_patch_position_z": patch["position"]["z"],
                }
            )
        # ---- Load weight scales ----
        for weight_scale in arena_setup["weight-scales"]:
            weight_scale_key = {"weight_scale_serial_number": weight_scale["serial-number"]}
            if weight_scale_key not in lab.WeightScale():
                lab.WeightScale.insert1(weight_scale_key)
            # ---- Check if this weight scale is currently installed - if so, remove it
            current_weight_scale_query = (
                acquisition.ExperimentWeightScale - acquisition.ExperimentWeightScale.RemovalTime
                & {"experiment_name": experiment_name}
                & weight_scale_key
            )
            if current_weight_scale_query:  # If the same weight scale is currently installed
                if (
                    current_weight_scale_query.fetch1("weight_scale_install_time")
                    == arena_setup["start-time"]
                ):
                    # If it is installed at the same time as that read from this yml file
                    # then it is the same ExperimentWeightScale instance, no need to do anything
                    continue

                # ---- Remove old weight scale
                acquisition.ExperimentWeightScale.RemovalTime.insert1(
                    {
                        **current_weight_scale_query.fetch1("KEY"),
                        "weight_scale_remove_time": arena_setup["start-time"],
                    }
                )

            nest_key = (
                lab.ArenaNest
                & (acquisition.Experiment & {"experiment_name": experiment_name})
                & {"nest": weight_scale["nest"]}
            ).fetch1("KEY")

            acquisition.ExperimentWeightScale.insert1(
                {
                    **weight_scale_key,
                    "experiment_name": experiment_name,
                    "weight_scale_install_time": arena_setup["start-time"],
                    "weight_scale_description": weight_scale["description"],
                    "weight_scale_sampling_rate": _weight_scale_rate,
                    **nest_key,
                }
            )


# ============ Manual and automatic steps to for experiment 0.1 populate ============

experiment_name = "exp0.1-r0"


def create_new_experiment():
    """Create a new experiment and add subjects to it."""
    # ---------------- Subject -----------------
    subject.Subject.insert(
        [
            {"subject": "BAA-1099790", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099791", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099792", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099793", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099794", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099795", "sex": "U", "subject_birth_date": "2021-01-01"},
            {"subject": "BAA-1099796", "sex": "U", "subject_birth_date": "2021-01-01"},
        ],
        skip_duplicates=True,
    )

    # ---------------- Experiment -----------------
    if {"experiment_name": experiment_name} not in acquisition.Experiment.proj():
        acquisition.Experiment.insert1(
            {
                "experiment_name": experiment_name,
                "experiment_start_time": "2021-06-03 07-00-00",
                "experiment_description": "experiment 0.1",
                "arena_name": "circle-2m",
                "lab": "SWC",
                "location": "room-0",
                "experiment_type": "foraging",
            }
        )
        acquisition.Experiment.Subject.insert(
            [
                {"experiment_name": experiment_name, "subject": "BAA-1099790"},
                {"experiment_name": experiment_name, "subject": "BAA-1099791"},
                {"experiment_name": experiment_name, "subject": "BAA-1099792"},
                {"experiment_name": experiment_name, "subject": "BAA-1099793"},
                {"experiment_name": experiment_name, "subject": "BAA-1099794"},
                {"experiment_name": experiment_name, "subject": "BAA-1099795"},
            ]
        )
        acquisition.Experiment.Directory.insert(
            [
                {
                    "experiment_name": experiment_name,
                    "repository_name": "ceph_aeon",
                    "directory_type": "raw",
                    "directory_path": "aeon/data/raw/AEON/experiment0.1",
                },
                {
                    "experiment_name": experiment_name,
                    "repository_name": "ceph_aeon",
                    "directory_type": "quality-control",
                    "directory_path": "aeon/data/qc/AEON/experiment0.1",
                },
            ]
        )

    if {"arena_name": "circle-2m", "nest": 1} not in lab.ArenaNest.proj():
        nest_coordinates = [
            (0.3264, 0.864),
            (0.3264, 1.0368),
            (0.4992, 0.864),
            (0.4992, 1.0368),
        ]
        lab.ArenaNest.insert1({"arena_name": "circle-2m", "nest": 1})
        lab.ArenaNest.Vertex.insert(
            (
                {
                    "arena_name": "circle-2m",
                    "nest": 1,
                    "vertex": v_id,
                    "vertex_x": x,
                    "vertex_y": y,
                }
                for v_id, (x, y) in enumerate(nest_coordinates)
            ),
            skip_duplicates=True,
        )


def add_arena_setup():
    """Add arena setup."""
    # Arena Setup - Experiment Devices
    this_file = pathlib.Path(__file__).expanduser().absolute().resolve()
    metadata_yml_filepath = this_file.parent / "setup_yml" / "Experiment0.1.yml"

    ingest_exp01_metadata(metadata_yml_filepath, experiment_name)

    # manually update coordinates of foodpatch and nest
    patch_coordinates = {"Patch1": (1.13, 1.59, 0), "Patch2": (1.19, 0.50, 0)}

    for patch_key in (acquisition.ExperimentFoodPatch & {"experiment_name": experiment_name}).fetch("KEY"):
        patch = (acquisition.ExperimentFoodPatch & patch_key).fetch1("food_patch_description")
        x, y, z = patch_coordinates[patch]
        acquisition.ExperimentFoodPatch.Position.update1(
            {
                **patch_key,
                "food_patch_position_x": x,
                "food_patch_position_y": y,
                "food_patch_position_z": z,
            }
        )


def main():
    """Main function to create a new experiment and set up the arena."""
    create_new_experiment()
    add_arena_setup()


if __name__ == "__main__":
    main()
