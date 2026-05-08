"""Root pytest configuration.

This conftest.py is loaded FIRST by pytest before any test collection.
It sets up:
- Mocks for unit tests that don't need database connections
- Testcontainers MySQL fixture for integration tests
- Auto-marking hook for integration fixtures
"""

import logging
import os
import sys

import pytest

logger = logging.getLogger(__name__)

# Set default DataJoint connection env vars (for imports to work)
# These will be overridden by mysql_container fixture for integration tests
if "DJ_HOST" not in os.environ:
    os.environ["DJ_HOST"] = "localhost"
if "DJ_USER" not in os.environ:
    os.environ["DJ_USER"] = "root"
if "DJ_PASS" not in os.environ:
    os.environ["DJ_PASS"] = "test_password"
if "DJ_PORT" not in os.environ:
    os.environ["DJ_PORT"] = "3306"


def pytest_configure(config):
    """Called before test collection begins.

    For unit tests, mock datajoint to prevent DB connection attempts
    during import of aeon.dj_pipeline modules.
    """
    # Check if we're running unit tests only
    markers = config.getoption("-m", default="")

    if "unit" in markers:
        from unittest.mock import MagicMock

        # Create a mock datajoint module
        mock_dj = MagicMock()
        mock_dj.logger = MagicMock()
        mock_dj.config = MagicMock()
        mock_dj.config.database.database_prefix = ""
        mock_dj.VirtualModule = MagicMock()
        mock_dj.Schema = MagicMock()
        mock_dj.DataJointError = Exception
        # Codec must be a real class so subclasses (AeonStreamCodec, OnixStreamCodec)
        # can be instantiated without turning into MagicMock objects.
        mock_dj.Codec = type("Codec", (), {})

        sys.modules["datajoint"] = mock_dj


def pytest_collection_modifyitems(items):
    """Auto-mark tests that use integration fixtures."""
    integration_fixtures = {"mysql_container", "pipeline_integration", "dj_config_integration"}
    for item in items:
        if integration_fixtures & set(item.fixturenames):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def mysql_container():
    """Auto-provision MySQL via testcontainers.

    Session-scoped to share container across all integration tests.
    Container is automatically cleaned up when session ends.
    """
    from testcontainers.mysql import MySqlContainer

    container = MySqlContainer(
        image="mysql:8.0",
        username="root",
        password="test_password",
        dbname="test_db",
    )
    container.start()

    # Update environment variables with container details
    host = container.get_container_host_ip()
    port = container.get_exposed_port(3306)
    os.environ["DJ_HOST"] = host
    os.environ["DJ_PORT"] = str(port)
    os.environ["DJ_USER"] = "root"
    os.environ["DJ_PASS"] = "test_password"

    logger.info(f"MySQL container started at {host}:{port}")

    yield container

    container.stop()
    logger.info("MySQL container stopped")
