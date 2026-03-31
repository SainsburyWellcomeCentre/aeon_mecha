"""Utility functions for creating new AEON experiments.

This module provides a reusable pattern for programmatically creating new experiments
in the DataJoint pipeline.
"""

from datetime import datetime
from pathlib import Path

from aeon.dj_pipeline import acquisition
from aeon.dj_pipeline.utils.paths import get_repository_path


def create_experiment(
    experiment_name: str,
    experiment_type: str,
    devices_schema_name: str,
    arena_name: str = "circle-2m",
    lab: str = "SWC",
    location: str | None = None,
    description: str | None = None,
    raw_dir: Path | None = None,
    repository_name: str = "ceph_aeon",
) -> None:
    """Create a new experiment entry in the database.

    This function creates the core experiment entries:
    - acquisition.Experiment (main experiment record)
    - acquisition.Experiment.Directory (data directories)
    - acquisition.Experiment.DevicesSchema (Pydantic schema class path)

    Args:
        experiment_name: Unique experiment identifier, e.g., "social0.4-aeon3"
        experiment_type: Type of experiment ("foraging" or "social")
        devices_schema_name: Full class path to the Pydantic Experiment class,
            e.g., "swc.aeon.exp.foragingABC.experiment.ForagingABC"
        arena_name: Name of the arena (default: "circle-2m")
        lab: Laboratory name (default: "SWC")
        location: Machine/room location. If None, inferred from experiment_name
        description: Experiment description. If None, auto-generated
        raw_dir: Path to raw data directory. If None, auto-discovered
        repository_name: Name of the data repository (default: "ceph_aeon")

    Raises:
        FileNotFoundError: If the raw data directory cannot be found

    Examples:
        >>> from aeon.dj_pipeline.utils.create_experiment import create_experiment
        >>> create_experiment(
        ...     experiment_name="myexp-aeon3",
        ...     experiment_type="foraging",
        ...     devices_schema_name="swc.aeon.exp.myexp.experiment.MyExperiment",
        ... )
    """
    # Parse experiment name to infer components
    # Convention: "{exp_name}-{machine_name}", e.g., "social0.4-aeon3"
    if "-" in experiment_name:
        exp_name, machine_name = experiment_name.rsplit("-", 1)
    else:
        exp_name = experiment_name
        machine_name = location or "unknown"

    # Auto-generate location if not provided
    if location is None:
        location = machine_name.upper()

    # Auto-generate description if not provided
    if description is None:
        description = f"{exp_name.capitalize()} experiment on {machine_name.upper()} machine"

    # Find raw data directory
    repo_path = get_repository_path(repository_name)
    if raw_dir is None:
        # Standard AEON data organization: {repo}/aeon/data/raw/{MACHINE}/{exp_name}
        raw_dir = repo_path / "aeon" / "data" / "raw" / machine_name.upper() / exp_name

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Find experiment start time from first epoch directory
    epoch_dirs = sorted([f.name for f in raw_dir.glob("*") if f.is_dir()])
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in: {raw_dir}")
    start_time = datetime.strptime(epoch_dirs[0], "%Y-%m-%dT%H-%M-%S")

    # Prepare experiment entry
    experiment_entry = {
        "experiment_name": experiment_name,
        "experiment_start_time": start_time,
        "experiment_description": description,
        "arena_name": arena_name,
        "lab": lab,
        "location": location,
        "experiment_type": experiment_type,
    }

    # Prepare directory entries
    # Standard directory types: raw, processed/ingest
    directory_entries = [
        {
            "experiment_name": experiment_name,
            "repository_name": repository_name,
            "directory_type": dir_type,
            "directory_path": (repo_path / "aeon" / "data" / dir_type / machine_name.upper() / exp_name)
            .relative_to(repo_path)
            .as_posix(),
            "load_order": load_order,
        }
        for load_order, dir_type in enumerate(["processed", "raw"])
        if (repo_path / "aeon" / "data" / dir_type / machine_name.upper() / exp_name).exists()
        or dir_type == "raw"  # Always include raw
    ]

    # Insert into database
    with acquisition.Experiment.connection.transaction:
        acquisition.Experiment.insert1(experiment_entry, skip_duplicates=True)
        acquisition.Experiment.Directory.insert(directory_entries, skip_duplicates=True)
        acquisition.Experiment.DevicesSchema.insert1(
            {"experiment_name": experiment_name, "devices_schema_name": devices_schema_name},
            skip_duplicates=True,
        )
