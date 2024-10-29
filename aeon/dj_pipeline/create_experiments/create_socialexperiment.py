"""Function to create new social experiments."""

from datetime import datetime

from aeon.dj_pipeline import acquisition
from aeon.dj_pipeline.utils.paths import get_repository_path

# ---- Programmatic creation of a new social experiment ----
# Infer experiment metadata from the experiment name
# User-specified "experiment_name" (everything else should be automatically inferred)
experiment_name = "social0.4-aeon3"


# Find paths
ceph_dir = get_repository_path("ceph_aeon")
ceph_data_dir = ceph_dir / "aeon" / "data"


def create_new_social_experiment(experiment_name):
    """Create new social experiment."""
    exp_name, machine_name = experiment_name.split("-")
    raw_dir = ceph_data_dir / "raw" / machine_name.upper() / exp_name
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    start_time = sorted([f.name for f in raw_dir.glob("*") if f.is_dir()])[0]

    new_experiment_entry = {
        "experiment_name": experiment_name,
        "experiment_start_time": datetime.strptime(start_time, "%Y-%m-%dT%H-%M-%S"),
        "experiment_description": f"{exp_name.capitalize()} experiment on {machine_name.upper()} machine",
        "arena_name": "circle-2m",
        "lab": "SWC",
        "location": machine_name.upper(),
        "experiment_type": "social",
    }
    experiment_directories = [
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": dir_type,
            "directory_path": (
                ceph_data_dir / dir_type / machine_name.upper() / exp_name
            )
            .relative_to(ceph_dir)
            .as_posix(),
            "load_order": load_order,
        }
        for load_order, dir_type in enumerate(["processed", "raw"])
    ]

    with acquisition.Experiment.connection.transaction:
        acquisition.Experiment.insert1(
            new_experiment_entry,
            skip_duplicates=True,
        )
        acquisition.Experiment.Directory.insert(
            experiment_directories, skip_duplicates=True
        )
        acquisition.Experiment.DevicesSchema.insert1(
            {
                "experiment_name": experiment_name,
                "devices_schema_name": exp_name.replace(".", ""),
            },
            skip_duplicates=True,
        )
