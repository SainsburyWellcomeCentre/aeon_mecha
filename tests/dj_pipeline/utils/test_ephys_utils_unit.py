"""Unit tests for ephys_utils pure helpers: probe assignment parsing and resolve_harp.

Tests target pure functions with no database dependencies. Carry-forward logic
(which queries EphysEpoch.Insertion) is tested during HPC validation.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection -- before any fixtures run --
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestParseProbeAssignmentsFile:
    """Test the JSON file parsing helper (pure function, no DB)."""

    def test_single_probe(self, tmp_path):
        """Single probe serial maps to correct label."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(
            json.dumps({"version": 1, "probe_assignments": {"23299108854": {"subject": "BAA-1104545"}}})
        )
        probe_info = {"ProbeB": "23299108854"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        result = _parse_probe_assignments_file(json_path, probe_info)

        assert result == {"ProbeB": {"subject": "BAA-1104545"}}

    def test_multiple_probes(self, tmp_path):
        """Multiple probes each map to their label."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "probe_assignments": {
                        "11111": {"subject": "BAA-1104545"},
                        "22222": {"subject": "BAA-1104545"},
                    },
                }
            )
        )
        probe_info = {"ProbeA": "11111", "ProbeB": "22222"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        result = _parse_probe_assignments_file(json_path, probe_info)

        assert result == {
            "ProbeA": {"subject": "BAA-1104545"},
            "ProbeB": {"subject": "BAA-1104545"},
        }

    def test_extra_serial_in_json_ignored(self, tmp_path):
        """Serials in JSON but not in probe_info are silently ignored."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "probe_assignments": {
                        "11111": {"subject": "BAA-1104545"},
                        "99999": {"subject": "BAA-9999999"},
                    },
                }
            )
        )
        probe_info = {"ProbeB": "11111"}  # only one probe active

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        result = _parse_probe_assignments_file(json_path, probe_info)

        assert result == {"ProbeB": {"subject": "BAA-1104545"}}

    def test_missing_serial_raises(self, tmp_path):
        """Serial in probe_info but not in JSON raises ValueError."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "probe_assignments": {
                        "11111": {"subject": "BAA-1104545"},
                    },
                }
            )
        )
        probe_info = {"ProbeA": "11111", "ProbeB": "99999"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        with pytest.raises(ValueError, match="99999"):
            _parse_probe_assignments_file(json_path, probe_info)

    def test_missing_version_raises(self, tmp_path):
        """JSON without version field raises ValueError."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({"probe_assignments": {"11111": {"subject": "X"}}}))
        probe_info = {"ProbeA": "11111"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        with pytest.raises(ValueError, match="version"):
            _parse_probe_assignments_file(json_path, probe_info)

    def test_missing_probe_assignments_key_raises(self, tmp_path):
        """JSON without probe_assignments key raises ValueError."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({"version": 1}))
        probe_info = {"ProbeA": "11111"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file

        with pytest.raises(ValueError, match="probe_assignments"):
            _parse_probe_assignments_file(json_path, probe_info)


class TestResolveHarp:
    """Test resolve_harp fast-path boundary checks and model-cache behaviour."""

    _SYNC_ROW = {
        "onix_ts_start": 1000,
        "onix_ts_end": 9000,
        "sync_start": datetime(2024, 6, 4, 11, 0, 0),
        "sync_end": datetime(2024, 6, 4, 12, 0, 0),
        "sync_model": "/should/never/be/loaded.joblib",
    }

    @pytest.mark.parametrize(
        ("onix_ts", "expected"),
        [
            (1000, datetime(2024, 6, 4, 11, 0, 0)),
            (9000, datetime(2024, 6, 4, 12, 0, 0)),
        ],
        ids=["start_boundary", "end_boundary"],
    )
    def test_fast_path(self, onix_ts, expected):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

        assert resolve_harp(self._SYNC_ROW, onix_ts=onix_ts) == expected

    @pytest.mark.parametrize(
        ("use_cache", "expected_loads"),
        [
            (True, 1),
            (False, 2),
        ],
        ids=["with_cache", "without_cache"],
    )
    def test_slow_path_cache_behavior(self, monkeypatch, use_cache, expected_loads):
        import joblib

        from aeon.dj_pipeline.utils.ephys_utils import resolve_harp

        sync_row = {**self._SYNC_ROW, "sync_model": "/fake/model.joblib"}
        load_calls = {"count": 0}

        def fake_load(path):
            load_calls["count"] += 1
            m = MagicMock()
            m.predict.return_value = np.array([[3000.0]])
            return m

        monkeypatch.setattr(joblib, "load", fake_load)

        cache: dict | None = {} if use_cache else None
        resolve_harp(sync_row, onix_ts=5000, _model_cache=cache)
        resolve_harp(sync_row, onix_ts=6000, _model_cache=cache)

        assert load_calls["count"] == expected_loads
