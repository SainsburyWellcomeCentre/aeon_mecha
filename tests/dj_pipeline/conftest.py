"""Global configurations and fixtures for pytest.

Test commands:
    pytest -m unit                    # Unit tests (no database)
    pytest -m integration             # Integration tests (testcontainers MySQL)
    pytest tests/dj_pipeline/         # All tests
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def dj_download_to_tmp(request):
    """Redirect DataJoint attach downloads to a per-test tmpdir.

    DataJoint extracts <attach> columns to `dj.config["download_path"]` (cwd by
    default). Without this fixture, fetching sync_model rows leaves .joblib files
    in the repo root. Skipped for unit tests which mock datajoint entirely.
    """
    if request.node.get_closest_marker("unit"):
        yield
        return
    import datajoint as dj

    with tempfile.TemporaryDirectory() as tmpdir, dj.config.override(download_path=tmpdir):
        yield


# Single test prefix for ALL integration tests
TEST_DB_PREFIX = "test_aeon_"

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

# Default golden data root (can be overridden via DJ_REPOSITORY_CONFIG env var)
DEFAULT_GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"

# NOTE: datajoint is imported lazily inside fixtures to allow unit tests
# (which mock datajoint) to run without triggering DB connections


# ============================================================================
# Integration Test Fixtures (testcontainers MySQL)
# ============================================================================


@pytest.fixture(scope="session")
def dj_config_integration(mysql_container):
    """Configure DataJoint and import pipeline with test prefix.

    Sets DJ config (including database_prefix) BEFORE importing pipeline
    modules. This way, module-level schema activation in lab.py, subject.py,
    acquisition.py, and streams.py naturally uses the test prefix — no manual
    re-decoration of table classes needed.
    """
    import datajoint as dj

    # Set config BEFORE any pipeline imports
    dj.config.safemode = False
    dj.config.database.host = os.environ.get("DJ_HOST", "localhost")
    dj.config.database.port = int(os.environ.get("DJ_PORT", "3306"))
    dj.config.database.user = os.environ.get("DJ_USER", "root")
    dj.config.database.password = os.environ.get("DJ_PASS", "test_password")
    dj.config.database.database_prefix = TEST_DB_PREFIX

    # Now import pipeline — all module-level schema activations use test prefix

    return {"database_prefix": TEST_DB_PREFIX}


@pytest.fixture(scope="session")
def streams_schema(dj_config_integration):
    """Provide access to streams catalog tables.

    Calls streams_maker.main() to ensure catalog tables (StreamType, DeviceType, etc.)
    are written to the auto-generated streams.py module.
    Session-scoped — shared by load_metadata integration tests and golden dataset tests.
    """
    import importlib
    import sys

    from aeon.dj_pipeline.utils import streams_maker

    f = streams_maker._STREAMS_MODULE_FILE
    original = f.read_bytes() if f.exists() else None

    # Delete auto-generated file so main() regenerates it with catalog tables
    f.unlink(missing_ok=True)

    # Remove stale module from cache so it gets reimported after regeneration
    sys.modules.pop("aeon.dj_pipeline.streams", None)

    streams_maker.main(create_tables=False)

    streams = importlib.import_module("aeon.dj_pipeline.streams")

    yield {
        "StreamType": streams.StreamType,
        "DeviceType": streams.DeviceType,
        "DeviceName": streams.DeviceName,
        "schema": streams.schema,
        "schema_name": streams.schema.database,
    }

    # Teardown
    # - drop test schema (may already be dropped by full_pipeline)
    # - restore original streams.py file
    import datajoint as dj

    try:
        dj.Schema(streams.schema.database).drop()
    except Exception:
        pass

    if original is not None:
        f.write_bytes(original)
    else:
        f.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pipeline_integration(dj_config_integration, streams_schema):
    """Integration test setup with DB and streams schema.

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
    streams.DeviceName().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()

    yield

    # Post-test cleanup (same order)
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
def require_golden_data(dj_config_integration, golden_dataset_config):
    """Skip tests if golden dataset is unavailable. Returns epoch path."""
    import aeon.dj_pipeline as pipeline

    # If DJ_REPOSITORY_CONFIG env var is not set, use default golden data root
    if "DJ_REPOSITORY_CONFIG" not in os.environ:
        pipeline.repository_config = {"ceph_aeon": str(DEFAULT_GOLDEN_DATA_ROOT)}

    if not Path(pipeline.repository_config["ceph_aeon"]).exists():
        pytest.skip(f"Golden data root not found: {pipeline.repository_config['ceph_aeon']}")

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
def full_pipeline(dj_config_integration, streams_schema, golden_dataset_config):
    """Full pipeline setup with all schemas for golden dataset tests.

    Since dj_config_integration sets database_prefix BEFORE importing pipeline
    modules, all schemas (lab, subject, acquisition, streams) are already
    activated with test_aeon_* databases. No manual re-decoration needed.

    Requires swc.aeon_exp (private package) — skips if not installed.

    Steps:
    1. Populate catalog from Pydantic class (populate_catalog_from_pydantic)
    2. Create ExperimentDevice and DeviceDataStream tables via streams_maker.main()
    3. Tests can then call EpochConfig.make() for DML
    """
    pytest.importorskip("swc.aeon_exp", reason="requires swc.aeon_exp package")
    import datajoint as dj

    from aeon.dj_pipeline import acquisition, lab, subject
    from aeon.dj_pipeline.utils import streams_maker
    from aeon.dj_pipeline.utils.load_metadata import get_experiment_pydantic, populate_catalog_from_pydantic

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

    # Teardown - drop all test schemas
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
    from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

    epoch_dt = parse_epoch_timestamp(cfg["epoch_dir"])

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
