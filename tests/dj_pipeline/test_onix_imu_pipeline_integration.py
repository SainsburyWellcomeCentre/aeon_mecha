"""Integration tests for the ONIX IMU pipeline (EphysSyncModel, EphysChunk, OnixImuChunk)."""

import logging

import pytest

logger = logging.getLogger(__name__)

# Auto-marked as integration via fixture usage; explicit marker for clarity:
pytestmark = pytest.mark.integration


# ============================================================================
# Task 4: Table existence
# ============================================================================


def test_ephys_sync_model_table_exists(dj_config_integration):
    """EphysSyncModel can be activated against an integration DB."""
    from aeon.dj_pipeline import ephys

    assert hasattr(ephys, "EphysSyncModel")
    table = ephys.EphysSyncModel()
    assert "experiment_name" in table.primary_key
    assert "epoch_start" in table.primary_key
    assert "sync_start" in table.primary_key

    attrs = set(table.heading.attributes.keys())
    expected_attrs = {"sync_end", "onix_ts_start", "onix_ts_end", "sync_model", "r2", "n_samples"}
    assert expected_attrs <= attrs, f"Missing: {expected_attrs - attrs}"
