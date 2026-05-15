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


def _make_mock_dj():
    from unittest.mock import MagicMock

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
    return mock_dj


@pytest.fixture(autouse=True)
def mock_dj_for_unit(request):
    """Auto-mock datajoint for unit tests.

    This prevents DB connection attempts when importing
    aeon.dj_pipeline modules in unit tests. Integration tests
    will use the real DJ with testcontainers MySQL.
    """
    if not request.node.get_closest_marker("unit"):
        yield
        return

    # Save and evict all datajoint + pipeline modules so the test gets a
    # fresh import with the mock; not a cached real-DJ module from a prior
    # integration test in the same session.
    evict_prefixes = ("datajoint", "aeon.dj_pipeline")
    saved = {
        k: v for k, v in sys.modules.items() if any(k == p or k.startswith(p + ".") for p in evict_prefixes)
    }
    for k in saved:
        sys.modules.pop(k)

    sys.modules["datajoint"] = _make_mock_dj()

    yield

    # Restore original modules (real DJ for integration tests that follow)
    for k in list(sys.modules):
        if any(k == p or k.startswith(p + ".") for p in evict_prefixes):
            sys.modules.pop(k)
    sys.modules.update(saved)

    # Clear (or restore) the `dj_pipeline` attribute on the `aeon` package so that
    # subsequent `import aeon.dj_pipeline as _pipeline` (IMPORT_FROM bytecode) reads
    # the correct module rather than the stale mock left on the parent package object.
    aeon_pkg = sys.modules.get("aeon")
    if aeon_pkg is not None:
        if "aeon.dj_pipeline" in saved:
            aeon_pkg.dj_pipeline = saved["aeon.dj_pipeline"]  # type: ignore[attr-defined]
        else:
            try:
                delattr(aeon_pkg, "dj_pipeline")
            except AttributeError:
                pass


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
