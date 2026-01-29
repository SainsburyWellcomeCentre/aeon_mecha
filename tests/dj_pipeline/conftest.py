"""Global configurations and fixtures for pytest.

Test commands:
    pytest -m unit                    # Unit tests (no database)
    pytest -m integration             # Integration tests (testcontainers MySQL)
    pytest tests/dj_pipeline/         # All tests
"""

import datetime
import os
from pathlib import Path

import pytest


# ============================================================================
# Golden Dataset Registry
# ============================================================================

GOLDEN_DATASETS = {
    "foraging_abc_2025_11_18": {
        "experiment_name": "abcBehav0-aeon3",
        "experiment_path": "AEON3/abcBehav0",
        "epoch_dir": "2025-11-18T10-13-15",
        "devices_schema": "swc.aeon.exp.foragingABC.experiment:Experiment",
        "arena_name": "arena-aeon3",
        "lab": "SWC",
        "location": "room-0",
        "experiment_type": "foraging",
        "required_files": [
            "Metadata.json",
            "CameraTop/CameraTop_2025-11-18T10-00-00.csv",
            "Feeder1/Feeder1_32_2025-11-18T10-00-00.bin",
        ],
        "expected_camera_count": 13,
        "expected_feeder_count": 6,
    },
    # Future datasets can be added here
}

# Default golden data root (can be overridden via dj.config["custom"]["repository_config"])
DEFAULT_GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"

# NOTE: datajoint is imported lazily inside fixtures to allow unit tests
# (which mock datajoint) to run without triggering DB connections


# ============================================================================
# Integration Test Fixtures (testcontainers MySQL)
# ============================================================================


@pytest.fixture(scope="session")
def dj_config_integration(mysql_container):
    """Configure DataJoint to use testcontainers MySQL.

    Sets up isolated test database with unique prefix.
    Environment variables are set by mysql_container fixture.
    """
    import datajoint as dj

    # Read connection details from environment (set by mysql_container fixture)
    dj.config.update(
        {
            "safemode": False,
            "database.host": os.environ.get("DJ_HOST", "localhost"),
            "database.port": int(os.environ.get("DJ_PORT", "3306")),
            "database.user": os.environ.get("DJ_USER", "root"),
            "database.password": os.environ.get("DJ_PASS", "test_password"),
        }
    )

    # Initialize custom dict if it doesn't exist
    if "custom" not in dj.config:
        dj.config["custom"] = {}

    dj.config["custom"]["database.prefix"] = "test_integration_"

    yield dj.config


@pytest.fixture(scope="session")
def streams_schema(dj_config_integration):
    """Create streams schema tables for integration tests.

    Creates catalog tables:
    - StreamType
    - DeviceType (with DeviceType.Stream part table)
    - DeviceName
    - Device

    Session-scoped to avoid repeated schema creation.
    """
    import datajoint as dj

    from aeon.dj_pipeline.utils.streams_maker import Device, DeviceName, DeviceType, StreamType

    # Get schema name with test prefix
    schema_name = dj_config_integration["custom"]["database.prefix"] + "streams"

    # Drop existing schema if it exists (ensures clean state with new table definitions)
    existing_schema = dj.Schema(schema_name)
    if existing_schema.is_activated():
        existing_schema.drop(force=True)

    schema = dj.Schema(schema_name)

    # Register catalog tables
    schema(StreamType)
    schema(DeviceType)
    schema(DeviceName)
    schema(Device)

    yield {
        "StreamType": StreamType,
        "DeviceType": DeviceType,
        "DeviceName": DeviceName,
        "Device": Device,
        "schema": schema,
        "schema_name": schema_name,
    }

    # Teardown: drop test schema
    schema.drop()


@pytest.fixture(scope="session")
def pipeline_integration(dj_config_integration, streams_schema):
    """Full integration test setup with DB and streams schema.

    Provides access to:
    - DataJoint config (dj_config_integration)
    - Streams catalog tables (streams_schema)
    - Virtual streams module for queries
    """
    import datajoint as dj

    streams = dj.VirtualModule("streams", streams_schema["schema_name"])

    return {
        "config": dj_config_integration,
        "streams": streams,
        **streams_schema,
    }


@pytest.fixture
def clean_streams_tables(pipeline_integration):
    """Clean streams tables before and after test.

    Use for tests that require empty catalog tables (FK handling tests).
    Note: DeviceType.Stream is a Part table - deleting DeviceType deletes it automatically.
    """
    streams = pipeline_integration["streams"]

    # Pre-test cleanup (order matters due to FK constraints)
    # Device references DeviceType, DeviceName references DeviceType
    # DeviceType.Stream is a Part table - deleted automatically with DeviceType
    streams.Device().delete()
    streams.DeviceName().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()

    yield

    # Post-test cleanup (same order)
    streams.Device().delete()
    streams.DeviceName().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()


# ============================================================================
# Golden Dataset Integration Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def golden_dataset_config():
    """Get the active golden dataset configuration."""
    return GOLDEN_DATASETS["foraging_abc_2025_11_18"]


@pytest.fixture(scope="session")
def dj_config_with_golden_data(mysql_container, golden_dataset_config):
    """Configure DataJoint for golden dataset tests."""
    import datajoint as dj

    if not DEFAULT_GOLDEN_DATA_ROOT.exists():
        pytest.skip(f"Golden data root not found: {DEFAULT_GOLDEN_DATA_ROOT}")

    dj.config.update(
        {
            "safemode": False,
            "database.host": os.environ.get("DJ_HOST", "localhost"),
            "database.port": int(os.environ.get("DJ_PORT", "3306")),
            "database.user": os.environ.get("DJ_USER", "root"),
            "database.password": os.environ.get("DJ_PASS", "test_password"),
        }
    )

    if "custom" not in dj.config:
        dj.config["custom"] = {}

    dj.config["custom"]["database.prefix"] = "test_golden_"
    dj.config["custom"]["repository_config"] = {
        "ceph_aeon": str(DEFAULT_GOLDEN_DATA_ROOT)
    }

    yield dj.config


@pytest.fixture(scope="session")
def require_golden_data(dj_config_with_golden_data, golden_dataset_config):
    """Skip tests if golden dataset is unavailable. Returns epoch path."""
    from aeon.dj_pipeline.utils.paths import get_repository_path

    repo_path = get_repository_path("ceph_aeon")
    epoch_path = repo_path / "raw" / golden_dataset_config["experiment_path"] / golden_dataset_config["epoch_dir"]

    if not epoch_path.exists():
        pytest.skip(f"Golden dataset not found: {epoch_path}")

    for f in golden_dataset_config["required_files"]:
        if not (epoch_path / f).exists():
            pytest.skip(f"Missing required file: {f}")

    return epoch_path


@pytest.fixture(scope="session")
def full_pipeline(dj_config_with_golden_data, golden_dataset_config):
    """Full pipeline setup with all schemas for golden dataset tests.

    Follows the "Three Decoupled Steps" architecture:
    1. Create base tables (StreamType, DeviceType, Device) via streams_maker.main()
    2. Populate catalog from Pydantic class
    3. Create ExperimentDevice and DeviceDataStream tables
    4. Tests can then call EpochConfig.make() for DML (Step 3)
    """
    from aeon.dj_pipeline import acquisition, lab, subject
    from aeon.dj_pipeline.utils import streams_maker
    from aeon.dj_pipeline.utils.load_metadata import (
        get_experiment_pydantic,
        populate_catalog_from_pydantic,
    )

    cfg = golden_dataset_config

    # Step 1a: Create base catalog tables (StreamType, DeviceType, Device)
    # This creates streams.py file and registers base tables in database
    streams_maker.main(create_tables=False)

    # Step 1b: Populate catalog from Pydantic class
    experiment_class = get_experiment_pydantic(cfg["devices_schema"])
    populate_catalog_from_pydantic(experiment_class)

    # Step 2: Create ExperimentDevice and DeviceDataStream tables
    streams = streams_maker.main(create_tables=True)

    yield {
        "lab": lab,
        "subject": subject,
        "acquisition": acquisition,
        "streams": streams,
    }

    # Teardown - drop schemas in reverse dependency order
    import datajoint as dj

    prefix = dj.config["custom"]["database.prefix"]
    # Get all schemas to drop, then drop them handling FK constraints
    schemas_to_drop = [s for s in dj.list_schemas() if s.startswith(prefix)]
    # Try multiple passes to handle foreign key dependencies
    max_attempts = len(schemas_to_drop) + 1
    for _ in range(max_attempts):
        if not schemas_to_drop:
            break
        remaining = []
        for schema_name in schemas_to_drop:
            try:
                dj.Schema(schema_name).drop()
            except Exception:
                remaining.append(schema_name)
        schemas_to_drop = remaining


@pytest.fixture(scope="session")
def test_experiment(full_pipeline, require_golden_data, golden_dataset_config):
    """Create experiment from golden dataset config."""
    lab = full_pipeline["lab"]
    acquisition = full_pipeline["acquisition"]
    cfg = golden_dataset_config

    # Insert required lookup entries (Arena and Location are Lookup tables with pre-populated contents)
    # Arena schema: arena_name, arena_description, arena_shape, arena_x_dim, arena_y_dim, arena_z_dim
    lab.Arena.insert1(
        {
            "arena_name": cfg["arena_name"],
            "arena_description": f"Arena for {cfg['experiment_name']}",
            "arena_shape": "circular",
            "arena_x_dim": 2.0,  # (m) diameter
            "arena_y_dim": 2.0,  # (m) diameter
            "arena_z_dim": 0.2,  # (m) wall height
        },
        skip_duplicates=True,
    )

    # Insert DevicesSchema
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": cfg["devices_schema"]},
        skip_duplicates=True,
    )

    # Parse epoch_dir for start time
    epoch_dt = datetime.datetime.strptime(cfg["epoch_dir"], "%Y-%m-%dT%H-%M-%S")

    # Insert experiment
    experiment_key = {
        "experiment_name": cfg["experiment_name"],
        "experiment_start_time": epoch_dt,
        "experiment_description": f"Golden dataset test: {cfg['experiment_name']}",
        "arena_name": cfg["arena_name"],
        "lab": cfg["lab"],
        "location": cfg["location"],
        "experiment_type": cfg["experiment_type"],
    }
    acquisition.Experiment.insert1(experiment_key, skip_duplicates=True)

    # Link DevicesSchema
    acquisition.Experiment.DevicesSchema.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "devices_schema_name": cfg["devices_schema"],
        },
        skip_duplicates=True,
    )

    # Insert Directory
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "directory_type": "raw",
            "repository_name": "ceph_aeon",
            "directory_path": f"raw/{cfg['experiment_path']}",
        },
        skip_duplicates=True,
    )

    return experiment_key


@pytest.fixture(scope="session")
def test_epochs(test_experiment, full_pipeline, golden_dataset_config):
    """Ingest epochs using Epoch.ingest_epochs() to test actual ingestion logic."""
    acquisition = full_pipeline["acquisition"]
    cfg = golden_dataset_config

    # Use actual ingest_epochs function - detects epochs from filesystem
    acquisition.Epoch.ingest_epochs(cfg["experiment_name"])

    # Return list of ingested epoch keys
    epochs = (acquisition.Epoch & {"experiment_name": cfg["experiment_name"]}).fetch("KEY")
    return epochs
