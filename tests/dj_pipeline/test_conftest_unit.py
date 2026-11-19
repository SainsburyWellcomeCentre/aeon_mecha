"""Unit tests for tests/dj_pipeline/conftest.py helpers.

Loads the conftest via importlib to avoid `import conftest` ambiguity when
multiple conftest.py files coexist in the tree.
"""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_dj_pipeline_conftest():
    """Import tests/dj_pipeline/conftest.py by explicit path."""
    conftest_path = Path(__file__).parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("dj_pipeline_conftest", conftest_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDropTestSchemasGuard:
    """Safety guard test for _drop_test_schemas — must refuse unsafe prefixes."""

    def test_refuses_empty_prefix(self, monkeypatch):
        import datajoint as dj

        conftest_mod = _load_dj_pipeline_conftest()

        monkeypatch.setattr(dj.config.database, "database_prefix", "")
        with pytest.raises(RuntimeError, match="not a safe test prefix"):
            conftest_mod._drop_test_schemas()


class TestPipelinePrefixGuard:
    """aeon.dj_pipeline's frozen prefix must match the one the tests configure.

    Without this, a module-scope aeon import in any test file freezes the production
    prefix during collection and the session writes schemas under production names.
    """

    def test_refuses_a_prefix_frozen_before_the_fixture_ran(self, monkeypatch):
        import aeon.dj_pipeline as pipeline

        conftest_mod = _load_dj_pipeline_conftest()

        monkeypatch.setattr(pipeline, "db_prefix", "aeon_")
        with pytest.raises(RuntimeError, match="froze database prefix"):
            conftest_mod._assert_pipeline_prefix("test_aeon_")

    def test_accepts_a_matching_prefix(self, monkeypatch):
        import aeon.dj_pipeline as pipeline

        conftest_mod = _load_dj_pipeline_conftest()

        monkeypatch.setattr(pipeline, "db_prefix", "test_aeon_")
        conftest_mod._assert_pipeline_prefix("test_aeon_")
