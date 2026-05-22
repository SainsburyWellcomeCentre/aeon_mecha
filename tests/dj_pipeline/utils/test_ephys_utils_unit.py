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


class TestGetProbeId:
    """Regression tests for get_probe_id — covers bugs 1, 2, 3 from bugs_found.md."""

    def test_v2_extracts_serial_from_windows_path(self):
        """V2 hardware: parses the Bonsai-written Windows path and returns the serial directory name.

        Covers bug 1 (correct metadata path: Devices.NeuropixelsV2e.ConfigurationA/B) AND
        bug 3 (PureWindowsPath correctly parses backslash separators on Linux).
        """
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        metadata = {
            "Devices": {
                "ProbeA": "true",
                "NeuropixelsV2e": {
                    "ConfigurationA": {
                        "GainCalibrationFileName": r"Z:\aeon\code\23299108854\file.csv",
                    },
                },
            },
        }
        assert get_probe_id(metadata, "NeuropixelsV2", "ProbeA") == "23299108854"

    def test_v2_returns_none_for_disabled_probe(self):
        """V2 hardware: Devices.ProbeA = "false" signals a disabled/dummy probe → return None.

        Covers bug 2 (downstream EphysEpoch.make filters None-returning probes).
        """
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        metadata = {
            "Devices": {
                "ProbeA": "false",
                "NeuropixelsV2e": {
                    "ConfigurationA": {
                        "GainCalibrationFileName": r"Z:\aeon\code\12345678\file.csv",
                    },
                },
            },
        }
        assert get_probe_id(metadata, "NeuropixelsV2", "ProbeA") is None

    def test_v2_falls_back_to_default_when_metadata_missing(self):
        """V2 hardware with metadata=None: fall back to "{device_name}_{probe_label}"."""
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        assert get_probe_id(None, "NeuropixelsV2", "ProbeA") == "NeuropixelsV2_ProbeA"

    def test_v2_falls_back_to_default_when_calibration_missing(self):
        """V2 hardware where GainCalibrationFileName is absent: fall back to default ID."""
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        metadata = {
            "Devices": {
                "ProbeA": "true",
                "NeuropixelsV2e": {
                    "ConfigurationA": {},  # no GainCalibrationFileName
                },
            },
        }
        assert get_probe_id(metadata, "NeuropixelsV2", "ProbeA") == "NeuropixelsV2_ProbeA"

    def test_v2beta_uses_default_id(self):
        """V2Beta hardware has no per-probe serial; always returns the default ID."""
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        # Even with metadata present, V2Beta path is not exercised — default returned.
        metadata = {
            "Devices": {
                "ProbeA": "true",
                "NeuropixelsV2e": {
                    "ConfigurationA": {
                        "GainCalibrationFileName": r"Z:\aeon\code\99999999\file.csv",
                    },
                },
            },
        }
        assert get_probe_id(metadata, "NeuropixelsV2Beta", "ProbeA") == "NeuropixelsV2Beta_ProbeA"


class TestDiscoverEpochProbes:
    """Tests for discover_epoch_probes(epoch_path)."""

    def test_single_probe(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        device_dir = tmp_path / "NeuropixelsV2Beta"
        device_dir.mkdir()
        (device_dir / "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin").touch()

        device_name, device_path, labels = discover_epoch_probes(tmp_path)

        assert device_name == "NeuropixelsV2Beta"
        assert device_path == device_dir
        assert labels == ["ProbeA"]

    def test_two_probes(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        device_dir = tmp_path / "NeuropixelsV2Beta"
        device_dir.mkdir()
        (device_dir / "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin").touch()
        (device_dir / "NeuropixelsV2Beta_ProbeB_AmplifierData_0.bin").touch()

        device_name, _, labels = discover_epoch_probes(tmp_path)

        assert device_name == "NeuropixelsV2Beta"
        assert labels == ["ProbeA", "ProbeB"]

    def test_no_device_directory(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        device_name, device_path, labels = discover_epoch_probes(tmp_path)

        assert device_name is None
        assert device_path is None
        assert labels == []

    def test_v2beta_preferred_over_v2(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        (tmp_path / "NeuropixelsV2Beta").mkdir()
        (tmp_path / "NeuropixelsV2Beta" / "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin").touch()
        (tmp_path / "NeuropixelsV2").mkdir()
        (tmp_path / "NeuropixelsV2" / "NeuropixelsV2_ProbeA_AmplifierData_0.bin").touch()

        device_name, _, _ = discover_epoch_probes(tmp_path)

        assert device_name == "NeuropixelsV2Beta"

    def test_no_amplifier_files(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        (tmp_path / "NeuropixelsV2Beta").mkdir()

        device_name, _, labels = discover_epoch_probes(tmp_path)

        assert device_name == "NeuropixelsV2Beta"
        assert labels == []

    def test_unexpected_filenames_ignored(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import discover_epoch_probes

        device_dir = tmp_path / "NeuropixelsV2Beta"
        device_dir.mkdir()
        (device_dir / "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin").touch()
        (device_dir / "NeuropixelsV2Beta_Clock_0.bin").touch()
        (device_dir / "random_file.txt").touch()

        _, _, labels = discover_epoch_probes(tmp_path)

        assert labels == ["ProbeA"]


class TestParseEpochMetadata:
    """Tests for parse_epoch_metadata(epoch_path)."""

    def test_valid_metadata(self, tmp_path):
        import json

        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        metadata = {"Devices": {"NeuropixelsV2e": {"ConfigurationA": {}}}}
        (tmp_path / "Metadata.yml").write_text(json.dumps(metadata))
        result = parse_epoch_metadata(tmp_path)

        assert result is not None
        assert "Devices" in result

    def test_missing_metadata(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        result = parse_epoch_metadata(tmp_path)

        assert result is None

    def test_malformed_metadata(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        (tmp_path / "Metadata.yml").write_text("not: valid: json: {{}")
        result = parse_epoch_metadata(tmp_path)

        assert result is None
