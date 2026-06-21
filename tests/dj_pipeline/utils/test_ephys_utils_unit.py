"""Unit tests for ephys_utils pure helpers: probe assignment parsing and resolve_harp.

Tests target pure functions with no database dependencies. Carry-forward logic
(which queries EphysEpochConfig.Insertion) is tested during HPC validation.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection -- before any fixtures run --
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

import json
from datetime import datetime
from pathlib import Path
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


def _mock_table_with_transaction():
    """Build a MagicMock table whose .connection.transaction is a no-op context manager."""
    table = MagicMock()
    table.connection.transaction.__enter__ = MagicMock(return_value=None)
    table.connection.transaction.__exit__ = MagicMock(return_value=False)
    return table


class TestCreateElectrodeConfig:
    """Unit tests for create_electrode_config — uses the fixture JSON, mocks tables.

    The fixture is a synthetic 4-shank NP2.0 probeinterface JSON with 16 total
    contacts and 8 active (odd contact_ids are active). The end-to-end check
    against a real probe lives in the ephys golden-dataset integration suite.
    """

    FIXTURE_JSON = (
        Path(__file__).parent.parent.parent / "fixtures" / "ephys" / "synthetic_np2_multishank.json"
    )
    N_TOTAL_CONTACTS = 16
    N_ACTIVE_CONTACTS = 8

    def test_returns_canonical_keys(self):
        from aeon.dj_pipeline.utils.ephys_utils import create_electrode_config

        probe_type_table = _mock_table_with_transaction()
        probe_type_table.Electrode = MagicMock()
        electrode_config_table = _mock_table_with_transaction()
        electrode_config_table.Electrode = MagicMock()

        probe_type, config_name = create_electrode_config(
            json_path=self.FIXTURE_JSON,
            probe_type_table=probe_type_table,
            electrode_config_table=electrode_config_table,
        )
        assert probe_type == "neuropixels2.0-multishank"
        assert config_name == "synthetic_np2_multishank"

    def test_inserts_full_geometry_and_active_subset(self):
        from aeon.dj_pipeline.utils.ephys_utils import create_electrode_config

        probe_type_table = _mock_table_with_transaction()
        probe_type_table.Electrode = MagicMock()
        electrode_config_table = _mock_table_with_transaction()
        electrode_config_table.Electrode = MagicMock()

        create_electrode_config(
            json_path=self.FIXTURE_JSON,
            probe_type_table=probe_type_table,
            electrode_config_table=electrode_config_table,
        )
        pt_call = probe_type_table.Electrode.insert.call_args
        pt_df = pt_call.args[0]
        assert len(pt_df) == self.N_TOTAL_CONTACTS

        ec_call = electrode_config_table.Electrode.insert.call_args
        ec_rows = list(ec_call.args[0])
        assert len(ec_rows) == self.N_ACTIVE_CONTACTS
        assert all({"probe_type", "electrode_config_name", "electrode"} <= set(r.keys()) for r in ec_rows)

    def test_config_name_override(self):
        from aeon.dj_pipeline.utils.ephys_utils import create_electrode_config

        probe_type_table = _mock_table_with_transaction()
        probe_type_table.Electrode = MagicMock()
        electrode_config_table = _mock_table_with_transaction()
        electrode_config_table.Electrode = MagicMock()

        _, config_name = create_electrode_config(
            json_path=self.FIXTURE_JSON,
            probe_type_table=probe_type_table,
            electrode_config_table=electrode_config_table,
            config_name="custom-name",
        )
        assert config_name == "custom-name"

    def test_does_not_set_config_file_name_on_electrode_config(self):
        """ElectrodeConfig has no JSON-provenance column.

        The JSON basename belongs on EphysEpochConfig.Insertion (per-(epoch,
        probe) record), recorded by the caller — not on the dedup'd
        ElectrodeConfig row.
        """
        from aeon.dj_pipeline.utils.ephys_utils import create_electrode_config

        probe_type_table = _mock_table_with_transaction()
        probe_type_table.Electrode = MagicMock()
        electrode_config_table = _mock_table_with_transaction()
        electrode_config_table.Electrode = MagicMock()

        create_electrode_config(
            json_path=self.FIXTURE_JSON,
            probe_type_table=probe_type_table,
            electrode_config_table=electrode_config_table,
        )
        ec_call = electrode_config_table.insert1.call_args
        inserted_row = ec_call.args[0]
        assert "config_file_name" not in inserted_row


class TestParseMetadataProbeConfigs:
    """Tests for parse_metadata_probe_configs — Metadata.yml → {probe_label: basename}."""

    def _write_metadata(self, tmp_path, npx_configs):
        """Build a minimal Metadata.yml with given Configuration* mapping."""
        data = {
            "Devices": {
                "NeuropixelsV2e": {
                    "DeviceName": "test",
                    **npx_configs,
                },
            },
        }
        path = tmp_path / "Metadata.yml"
        path.write_text(json.dumps(data))
        return tmp_path

    def test_probe_a_and_b_both_active(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        ep = self._write_metadata(
            tmp_path,
            {
                "ConfigurationA": {"ProbeInterfaceFileName": r"Z:\dir\probeA.json"},
                "ConfigurationB": {"ProbeInterfaceFileName": r"Z:\dir\probeB.json"},
            },
        )
        result = parse_metadata_probe_configs(ep)
        assert result == {"ProbeA": "probeA.json", "ProbeB": "probeB.json"}

    def test_probe_a_disabled(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        ep = self._write_metadata(
            tmp_path,
            {
                "ConfigurationA": {"ProbeInterfaceFileName": None},
                "ConfigurationB": {"ProbeInterfaceFileName": r"Z:\dir\probeB.json"},
            },
        )
        result = parse_metadata_probe_configs(ep)
        assert result == {"ProbeA": None, "ProbeB": "probeB.json"}

    def test_missing_metadata_raises(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        with pytest.raises(FileNotFoundError):
            parse_metadata_probe_configs(tmp_path)

    def test_real_golden_metadata(self):
        """Verify against the actual golden epoch's Metadata.yml if available."""
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        epoch_path = (
            Path.home()
            / "sciops-data"
            / "project_aeon"
            / "aeon"
            / "data"
            / "raw"
            / "AEONX1"
            / "abcGolden01"
            / "2026-05-11T07-50-11"
        )
        if not epoch_path.exists():
            pytest.skip(f"Golden epoch not on disk: {epoch_path}")
        result = parse_metadata_probe_configs(epoch_path)
        assert result["ProbeA"] is None
        assert result["ProbeB"] == "M81_ProbeB_4Shanks_1000_to_1700_um.json"


class TestResolveEpochProbeJson:
    """Tests for resolve_epoch_probe_json — central → epoch-local fallback."""

    def test_prefers_central_when_both_exist(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_epoch_probe_json

        raw = tmp_path / "AEONX1"
        epoch = raw / "2024-01-01T00-00-00"
        (raw / "recording_configurations").mkdir(parents=True)
        epoch.mkdir(parents=True)
        central = raw / "recording_configurations" / "probe.json"
        local = epoch / "probe.json"
        central.write_text("{}")
        local.write_text("{}")

        assert resolve_epoch_probe_json(raw, epoch, "probe.json") == central

    def test_falls_back_to_epoch_local(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_epoch_probe_json

        raw = tmp_path / "AEONX1"
        epoch = raw / "2024-01-01T00-00-00"
        epoch.mkdir(parents=True)
        local = epoch / "probe.json"
        local.write_text("{}")
        # No recording_configurations/ dir at all

        assert resolve_epoch_probe_json(raw, epoch, "probe.json") == local

    def test_raises_when_neither_exists(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_epoch_probe_json

        raw = tmp_path / "AEONX1"
        epoch = raw / "2024-01-01T00-00-00"
        epoch.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="not found at"):
            resolve_epoch_probe_json(raw, epoch, "probe.json")

    def test_accepts_str_args(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_epoch_probe_json

        raw = tmp_path / "AEONX1"
        epoch = raw / "2024-01-01T00-00-00"
        epoch.mkdir(parents=True)
        (epoch / "probe.json").write_text("{}")

        # Both args as strings (not Path) — common from older callers
        p = resolve_epoch_probe_json(str(raw), str(epoch), "probe.json")
        assert p == epoch / "probe.json"


class TestLoadDeviceChannelMap:
    """Tests for load_device_channel_map — takes JSON path directly.

    The fixture is a synthetic 4-shank NP2.0 probeinterface JSON: 16 contacts,
    odd contact_ids active, mapped sequentially to channels 0..7. So contact
    1 → channel 0, contact 3 → channel 1, ..., contact 15 → channel 7.
    """

    FIXTURE_JSON = (
        Path(__file__).parent.parent.parent / "fixtures" / "ephys" / "synthetic_np2_multishank.json"
    )

    def test_returns_active_contacts_mapping(self):
        from aeon.dj_pipeline.utils.ephys_utils import load_device_channel_map

        channel_map = load_device_channel_map(self.FIXTURE_JSON)
        assert len(channel_map) == 8
        assert channel_map[1] == 0
        assert channel_map[15] == 7

    def test_raises_on_missing_file(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import load_device_channel_map

        with pytest.raises(FileNotFoundError):
            load_device_channel_map(tmp_path / "does_not_exist.json")

    def test_raises_on_no_device_channel_indices(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import load_device_channel_map

        bad = tmp_path / "bad.json"
        bad.write_text('{"probes": [{"contact_ids": ["0"]}]}')
        with pytest.raises(ValueError, match="No device_channel_indices"):
            load_device_channel_map(bad)

    def test_raises_on_no_active_contacts(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import load_device_channel_map

        all_inactive = tmp_path / "inactive.json"
        all_inactive.write_text(
            '{"probes": [{"contact_ids": ["0", "1"], "device_channel_indices": [-1, -1]}]}'
        )
        with pytest.raises(ValueError, match="No active contacts"):
            load_device_channel_map(all_inactive)
