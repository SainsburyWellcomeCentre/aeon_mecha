"""Global configurations and fixtures for pytest.

# run all tests:
# pytest -sv --cov-report term-missing --cov=aeon_mecha -p no:warnings tests/dj_pipeline

# run one test, debug:
# pytest [above options] --pdb tests/dj_pipeline/test_ingestion.py -k <function_name>

# run test on marker:
# pytest -m <marker_name>
"""

import os
import pathlib

import pytest

# NOTE: datajoint is imported lazily inside fixtures to allow unit tests
# (which mock datajoint) to run without triggering DB connections

_tear_down = True  # always set to True since most fixtures are session-scoped
_populate_settings = {"suppress_errors": True}


def data_dir():
    """Returns test data directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


@pytest.fixture(scope="session")
def test_params():
    return {
        "start_ts": "2022-06-22 08:51:10",
        "end_ts": "2022-06-22 14:00:00",
        "experiment_name": "exp0.2-r0",
        "raw_dir": "aeon/data/raw/AEON2/experiment0.2",
        "qc_dir": "aeon/data/qc/AEON2/experiment0.2",
        "test_dir": data_dir(),
        "subject_count": 5,
        "epoch_count": 1,
        "chunk_count": 7,
        "experiment_log_message_count": 0,
        "subject_enter_exit_count": 0,
        "subject_weight_time_count": 0,
        "camera_qc_count": 40,
        "camera_tracking_object_count": 5,
    }


@pytest.fixture(scope="session")
def _dj_config():
    """Configures DataJoint connection and loads custom settings.

    This fixture sets up the DataJoint configuration using the
    'dj_local_conf.json' file. It raises FileNotFoundError if the file
    does not exist, and KeyError if 'custom' is not found in the
    DataJoint configuration.
    """
    import datajoint as dj

    dj_config_fp = pathlib.Path("dj_local_conf.json")
    assert dj_config_fp.exists()
    dj.config.load(dj_config_fp)
    dj.config["safemode"] = False
    assert "custom" in dj.config
    dj.config["custom"][
        "database.prefix"
    ] = f"u_{dj.config['database.user']}_testsuite_"


def load_pipeline():
    from aeon.dj_pipeline import (
        acquisition,
        analysis,
        lab,
        qc,
        report,
        subject,
        tracking,
    )

    return {
        "subject": subject,
        "lab": lab,
        "acquisition": acquisition,
        "qc": qc,
        "tracking": tracking,
        "analysis": analysis,
        "report": report,
    }


def drop_schema():
    _pipeline = load_pipeline()

    _pipeline["report"].schema.drop()
    _pipeline["analysis"].schema.drop()
    _pipeline["tracking"].schema.drop()
    _pipeline["qc"].schema.drop()
    _pipeline["acquisition"].schema.drop()
    _pipeline["subject"].schema.drop()
    _pipeline["lab"].schema.drop()

    print("\n\nAll schemas dropped")


@pytest.fixture(scope="session")
def pipeline(_dj_config):
    _pipeline = load_pipeline()

    yield _pipeline

    if _tear_down:
        drop_schema()


@pytest.fixture(scope="session")
def _experiment_creation(test_params, pipeline):
    from aeon.dj_pipeline.create_experiments import create_experiment_02

    create_experiment_02.main()

    acquisition = pipeline["acquisition"]

    experiment_name = acquisition.Experiment.fetch1("experiment_name")

    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "raw",
            "directory_path": test_params["raw_dir"],
        }
    )
    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "quality-control",
            "directory_path": test_params["qc_dir"],
        }
    )


@pytest.fixture(scope="session")
def _epoch_chunk_ingestion(test_params, pipeline, _experiment_creation):
    acquisition = pipeline["acquisition"]

    test_params["experiment_name"]

    acquisition.Epoch.ingest_epochs(
        experiment_name=test_params["experiment_name"],
        start=test_params["start_ts"],
        end=test_params["end_ts"],
    )

    acquisition.Chunk.ingest_chunks(experiment_name=test_params["experiment_name"])


@pytest.fixture(scope="session")
def _experimentlog_ingestion(pipeline):
    acquisition = pipeline["acquisition"]
    if not len(acquisition.Chunk()):
        raise Exception("Chunk table is empty!")
    acquisition.ExperimentLog.populate(**_populate_settings)
    acquisition.SubjectEnterExit.populate(**_populate_settings)
    acquisition.SubjectWeight.populate(**_populate_settings)


@pytest.fixture(scope="session")
def _camera_qc_ingestion(pipeline, _epoch_chunk_ingestion):
    qc = pipeline["qc"]
    qc.CameraQC.populate(**_populate_settings)


@pytest.fixture(scope="session")
def _camera_tracking_ingestion(pipeline, _camera_qc_ingestion):
    tracking = pipeline["tracking"]
    tracking.CameraTracking.populate(**_populate_settings)


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
    - Device

    Session-scoped to avoid repeated schema creation.
    """
    import datajoint as dj

    from aeon.dj_pipeline.utils.streams_maker import Device, DeviceType, StreamType

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
    schema(Device)

    yield {
        "StreamType": StreamType,
        "DeviceType": DeviceType,
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
    # Device references DeviceType, so delete Device first
    # DeviceType.Stream is a Part table - deleted automatically with DeviceType
    streams.Device().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()

    yield

    # Post-test cleanup (same order)
    streams.Device().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()
