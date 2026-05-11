"""Unit tests for resolve_harp helper in aeon.dj_pipeline.utils.ephys_utils."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def testresolve_harp_fast_path_start_returns_observed_sync_start():
    """When onix_ts == onix_ts_start exactly, returns observed sync_start (no joblib load)."""
    from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

    sync_row = {
        "onix_ts_start": 1000,
        "onix_ts_end": 9000,
        "sync_start": datetime(2024, 6, 4, 11, 0, 0),
        "sync_end": datetime(2024, 6, 4, 12, 0, 0),
        "sync_model": "/should/never/be/loaded.joblib",
    }
    result = resolve_harp(sync_row, onix_ts=1000)
    assert result == datetime(2024, 6, 4, 11, 0, 0)


def testresolve_harp_fast_path_end_returns_observed_sync_end():
    """When onix_ts == onix_ts_end exactly, returns observed sync_end (no joblib load)."""
    from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

    sync_row = {
        "onix_ts_start": 1000,
        "onix_ts_end": 9000,
        "sync_start": datetime(2024, 6, 4, 11, 0, 0),
        "sync_end": datetime(2024, 6, 4, 12, 0, 0),
        "sync_model": "/should/never/be/loaded.joblib",
    }
    result = resolve_harp(sync_row, onix_ts=9000)
    assert result == datetime(2024, 6, 4, 12, 0, 0)


def testresolve_harp_slow_path_uses_model_cache(monkeypatch):
    """When _model_cache is provided, repeated calls don't reload the model."""
    import joblib

    from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

    sync_row = {
        "onix_ts_start": 1000,
        "onix_ts_end": 9000,
        "sync_start": datetime(2024, 6, 4, 11, 0, 0),
        "sync_end": datetime(2024, 6, 4, 12, 0, 0),
        "sync_model": "/fake/model.joblib",
    }

    load_calls = {"count": 0}

    def fake_load(path):
        load_calls["count"] += 1
        m = MagicMock()
        m.predict.return_value = np.array([[3000.0]])  # arbitrary harp seconds
        return m

    monkeypatch.setattr(joblib, "load", fake_load)

    cache: dict = {}
    # First call: middle of the range, slow path
    resolve_harp(sync_row, onix_ts=5000, _model_cache=cache)
    # Second call: same sync_row, different ts — should reuse cached model
    resolve_harp(sync_row, onix_ts=6000, _model_cache=cache)

    assert load_calls["count"] == 1, "Model should be loaded only once when cache is provided"


def testresolve_harp_slow_path_without_cache_reloads(monkeypatch):
    """Without _model_cache, every call hits joblib.load."""
    import joblib

    from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

    sync_row = {
        "onix_ts_start": 1000,
        "onix_ts_end": 9000,
        "sync_start": datetime(2024, 6, 4, 11, 0, 0),
        "sync_end": datetime(2024, 6, 4, 12, 0, 0),
        "sync_model": "/fake/model.joblib",
    }

    load_calls = {"count": 0}

    def fake_load(path):
        load_calls["count"] += 1
        m = MagicMock()
        m.predict.return_value = np.array([[3000.0]])
        return m

    monkeypatch.setattr(joblib, "load", fake_load)

    resolve_harp(sync_row, onix_ts=5000)
    resolve_harp(sync_row, onix_ts=6000)

    assert load_calls["count"] == 2, "Without cache, each call reloads the model"
