"""Global configurations and fixtures for pytest.

Test commands:
    pytest -m unit                    # Unit tests (no database)
    pytest -m integration             # Integration tests (testcontainers MySQL)
    pytest tests/dj_pipeline/         # All tests
"""

import datetime
import logging
import os
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


# ============================================================================
# Golden Dataset Registry
# ============================================================================

GOLDEN_DATASETS = {
    "foraging_abc_2025_11_18": {
        "experiment_name": "abcBehav0-aeon3",
        "experiment_path": "AEON3/abcBehav0",
        "epoch_dir": "2025-11-18T10-13-15",
        "devices_schema": "swc.aeon_exp.foragingABC.experiment:Experiment",
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
    dj.config.safemode = False
    dj.config.database.host = os.environ.get("DJ_HOST", "localhost")
    dj.config.database.port = int(os.environ.get("DJ_PORT", "3306"))
    dj.config.database.user = os.environ.get("DJ_USER", "root")
    dj.config.database.password = os.environ.get("DJ_PASS", "test_password")
    dj.config.database.database_prefix = "test_integration_"

    return dj.config


@pytest.fixture(scope="session")
def streams_schema(dj_config_integration):
    """Create streams schema tables for integration tests.

    Creates catalog tables:
    - StreamType
    - DeviceType (with DeviceType.Stream part table)
    - DeviceName
    - Device

    Uses a fresh dj.Schema object with the undecorated catalog class definitions
    from streams_maker (not streams.schema, which may be polluted by full_pipeline's
    main() appending dynamic tables with unresolvable FK references).
    Session-scoped to avoid repeated schema creation.
    """
    import datajoint as dj

    import aeon.dj_pipeline as pipeline

    # Save and patch db_prefix so get_schema_name() returns test-prefixed names
    original_db_prefix = pipeline.db_prefix
    target_prefix = dj.config.database.database_prefix
    pipeline.db_prefix = target_prefix

    schema_name = target_prefix + "streams"

    # Create a fresh Schema for catalog tables only.
    # We use the undecorated classes from streams_maker (not streams.schema)
    # because full_pipeline's main() may have polluted streams.schema with
    # dynamic table classes whose FK refs don't exist in this test DB.
    from aeon.dj_pipeline.utils.streams_maker import (
        Device,
        DeviceName,
        DeviceType,
        StreamType,
    )

    test_schema = dj.Schema(schema_name, create_schema=True, create_tables=True)
    test_schema(StreamType)
    test_schema(DeviceType)
    test_schema(DeviceName)
    test_schema(Device)

    yield {
        "StreamType": StreamType,
        "DeviceType": DeviceType,
        "DeviceName": DeviceName,
        "Device": Device,
        "schema": test_schema,
        "schema_name": schema_name,
    }

    # Teardown: drop test schema and restore db_prefix
    dj.Schema(schema_name).drop()
    pipeline.db_prefix = original_db_prefix


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

    dj.config.safemode = False
    dj.config.database.host = os.environ.get("DJ_HOST", "localhost")
    dj.config.database.port = int(os.environ.get("DJ_PORT", "3306"))
    dj.config.database.user = os.environ.get("DJ_USER", "root")
    dj.config.database.password = os.environ.get("DJ_PASS", "test_password")
    dj.config.database.database_prefix = "test_golden_"

    import aeon.dj_pipeline as pipeline

    pipeline.repository_config = {"ceph_aeon": str(DEFAULT_GOLDEN_DATA_ROOT)}

    return dj.config


@pytest.fixture(scope="session")
def require_golden_data(dj_config_with_golden_data, golden_dataset_config):
    """Skip tests if golden dataset is unavailable. Returns epoch path."""
    from aeon.dj_pipeline.utils.paths import get_repository_path

    repo_path = get_repository_path("ceph_aeon")
    epoch_path = (
        repo_path / "raw" / golden_dataset_config["experiment_path"] / golden_dataset_config["epoch_dir"]
    )

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
    1. Activate schemas with golden test prefix
    2. Populate catalog from Pydantic class
    3. Create ExperimentDevice and DeviceDataStream tables via streams_maker.main()
    4. Tests can then call EpochConfig.make() for DML

    When schema integration tests run first (with a different db prefix),
    module-level variables like db_prefix and streams_maker.schema_name become
    stale. This fixture patches them and uses schema.activate() to rebind all
    schemas to the golden test database.
    """
    import datajoint as dj

    import aeon.dj_pipeline as pipeline
    from aeon.dj_pipeline import streams
    from aeon.dj_pipeline.utils import streams_maker

    # Save and patch db_prefix to match golden test config.
    # Other integration tests may have set a different prefix at import time.
    original_db_prefix = pipeline.db_prefix
    target_prefix = dj.config.database.database_prefix
    pipeline.db_prefix = target_prefix
    streams_maker.schema_name = pipeline.get_schema_name("streams")

    # Re-activate pipeline schemas with the correct prefix.
    # Importing streams.py triggers acquisition/lab/subject schema activation
    # at module level (with the default or a previous test prefix).
    #
    # WORKAROUND: DataJoint's Schema.activate() raises DataJointError if the
    # schema is already activated for a different database name (schemas.py:109-116).
    # There is no public API for rebinding a Schema to a different database.
    # We reset schema.database = None to bypass this guard before re-activating.
    # This is an internal implementation detail — if DataJoint adds a rebind() API
    # or changes Schema internals, this pattern should be updated accordingly.
    from aeon.dj_pipeline import acquisition, lab, subject

    for module, name in [(lab, "lab"), (subject, "subject"), (acquisition, "acquisition")]:
        expected_schema = pipeline.get_schema_name(name)
        if hasattr(module, "schema") and module.schema.database != expected_schema:
            module.schema.database = None
            module.schema.activate(expected_schema, create_schema=True, create_tables=True)

    # Rebind streams schema to golden test database (same workaround as above)
    streams_schema_name = pipeline.get_schema_name("streams")
    if streams.schema.database != streams_schema_name:
        streams.schema.database = None
        streams.schema.activate(streams_schema_name, create_schema=True, create_tables=True)

    from aeon.dj_pipeline.utils.load_metadata import (
        get_experiment_pydantic,
        populate_catalog_from_pydantic,
    )

    cfg = golden_dataset_config

    # Step 1: Populate catalog from Pydantic class
    experiment_class = get_experiment_pydantic(cfg["devices_schema"])
    populate_catalog_from_pydantic(experiment_class)

    # Step 2: Create ExperimentDevice and DeviceDataStream tables
    streams_module = streams_maker.main(create_tables=True)

    yield {
        "lab": lab,
        "subject": subject,
        "acquisition": acquisition,
        "streams": streams_module,
    }

    # Teardown - drop schemas in reverse dependency order, then restore db_prefix
    prefix = dj.config.database.database_prefix
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
            except Exception as e:
                logger.warning(f"Failed to drop schema {schema_name}: {e}")
                remaining.append(schema_name)
        schemas_to_drop = remaining
    if schemas_to_drop:
        logger.error(f"Could not drop schemas after {max_attempts} attempts: {schemas_to_drop}")

    pipeline.db_prefix = original_db_prefix


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
    epochs = (acquisition.Epoch & {"experiment_name": cfg["experiment_name"]}).keys()
    return epochs
