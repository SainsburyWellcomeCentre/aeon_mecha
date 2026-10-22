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


class TestFindNearestWindow:
    """Test find_nearest_window index selection, including out-of-range ts."""

    _STARTS = np.array([100, 200, 300], dtype=np.int64)

    @pytest.mark.parametrize(
        ("ts", "expected"),
        [
            (50, 0),  # before every window -> clipped to first
            (100, 0),  # exact match on first window's start
            (150, 0),  # inside first window's span
            (250, 1),  # inside second window's span
            (300, 2),  # exact match on last window's start
            (999, 2),  # after every window -> last window (extrapolated forward)
        ],
    )
    def test_scalar(self, ts, expected):
        """Test scalar behaviour: used in EphysChunk.ingest_chunks."""
        from aeon.dj_pipeline.utils.ephys_utils import find_nearest_window

        assert find_nearest_window(self._STARTS, ts) == expected

    def test_array(self):
        """Test vectorised behaviour: used in SyncedSpikes.indices2syncedtimes."""
        from aeon.dj_pipeline.utils.ephys_utils import find_nearest_window

        ts = np.array([50, 150, 250, 999])
        result = find_nearest_window(self._STARTS, ts)

        np.testing.assert_array_equal(result, [0, 0, 1, 2])


def _gain_cal(serial):
    """GainCalibrationFileName entry as Bonsai writes it: a Windows path with the serial as parent dir."""
    return {"GainCalibrationFileName": f"Z:\\aeon\\code\\{serial}\\file.csv"}


def _old_layout(probe_a="true", config_a=None):
    """Pre-2026 metadata: enable flags at Devices.ProbeA/B, configs at Devices.NeuropixelsV2e."""
    return {
        "Devices": {
            "ProbeA": probe_a,
            "ProbeB": "true",
            "NeuropixelsV2e": {
                "ConfigurationA": _gain_cal("11111111") if config_a is None else config_a,
                "ConfigurationB": _gain_cal("22222222"),
            },
        },
    }


def _new_layout(block):
    """2026+ metadata: a top-level NeuropixelsV2B block holding its own enable flag and config."""
    return {"NeuropixelsV2B": block}


class TestGetProbeId:
    """Regression tests for get_probe_id — covers bugs 1, 2, 3 from bugs_found.md."""

    @pytest.mark.parametrize(
        ("metadata", "probe_label", "expected"),
        [
            pytest.param(None, "ProbeA", "NeuropixelsV2_ProbeA", id="no metadata"),
            # Bug 1: serial comes from Devices.NeuropixelsV2e.ConfigurationA/B.
            # Bug 3: PureWindowsPath parses the backslash separators on Linux.
            pytest.param(_old_layout(), "ProbeA", "11111111", id="old layout, serial"),
            pytest.param(_old_layout(), "ProbeB", "22222222", id="old layout, ProbeB reads ConfigurationB"),
            # Bug 2: a disabled/dummy probe returns None, which EphysEpoch.make filters out.
            pytest.param(_old_layout(probe_a="false"), "ProbeA", None, id="old layout, disabled"),
            pytest.param(
                _old_layout(config_a={}), "ProbeA", "NeuropixelsV2_ProbeA", id="old layout, no serial"
            ),
            pytest.param(
                _new_layout({"Enable": "true", "ProbeConfiguration": _gain_cal("33333333")}),
                "ProbeB",
                "33333333",
                id="new layout, serial",
            ),
            pytest.param(
                _new_layout({"ProbeConfiguration": _gain_cal("33333333")}),
                "ProbeB",
                "33333333",
                id="new layout, enable flag absent defaults to enabled",
            ),
            pytest.param(
                _new_layout({"Enable": "False", "ProbeConfiguration": _gain_cal("33333333")}),
                "ProbeB",
                None,
                id="new layout, disabled",
            ),
            pytest.param(
                _new_layout({"Enable": "true", "ProbeConfiguration": {}}),
                "ProbeB",
                "NeuropixelsV2_ProbeB",
                id="new layout, no serial",
            ),
            # Devices.NeuropixelsV2e.ConfigurationB holds a different serial (22222222),
            # so this fails if the old layout is read before the new block.
            pytest.param(
                {**_old_layout(), **_new_layout({"ProbeConfiguration": _gain_cal("33333333")})},
                "ProbeB",
                "33333333",
                id="new layout takes precedence over Devices",
            ),
        ],
    )
    def test_v2(self, metadata, probe_label, expected):
        """V2 hardware: probe ID is the serial from GainCalibrationFileName, or None if disabled."""
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        assert get_probe_id(metadata, "NeuropixelsV2", probe_label) == expected

    def test_v2beta_uses_default_id(self):
        """V2Beta hardware has no per-probe serial; always returns the default ID."""
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id

        # Even with V2 metadata present, the V2 path is not taken — default returned.
        assert get_probe_id(_old_layout(), "NeuropixelsV2Beta", "ProbeA") == "NeuropixelsV2Beta_ProbeA"


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

    def test_malformed_metadata_names_the_file_rather_than_denying_it(self, tmp_path):
        """A corrupt file must not be reported as an absent one.

        Returning None here reaches parse_metadata_probe_configs as
        FileNotFoundError("No Metadata.json or Metadata.yml in ..."), sending an
        operator after a file that is sitting right there, truncated.
        """
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        (tmp_path / "Metadata.yml").write_text("not: valid: json: {{}")
        with pytest.raises(ValueError, match="Metadata.yml"):
            parse_epoch_metadata(tmp_path)


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

    def test_falls_back_to_device_subdirectory(self, tmp_path):
        """Arrow-era epochs keep the probe config inside <epoch>/<device>/."""
        from aeon.dj_pipeline.utils.ephys_utils import resolve_epoch_probe_json

        raw = tmp_path / "AEONX1"
        device_dir = raw / "2026-10-13T094255Z" / "NeuropixelsV2"
        device_dir.mkdir(parents=True)
        (device_dir / "probe.json").write_text("{}")

        assert resolve_epoch_probe_json(raw, device_dir.parent, "probe.json") == device_dir / "probe.json"

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
        with pytest.raises(FileNotFoundError) as exc:
            resolve_epoch_probe_json(raw, epoch, "probe.json")
        # The message must name every candidate path, so a miss is diagnosable.
        # Asserting on the bare device name would be satisfied by 'NeuropixelsV2Beta'
        # alone, and would not notice a candidate dropped from the search list.
        from aeon.dj_pipeline.utils.ephys_utils import DEVICE_PROBE_TYPE_MAP

        message = str(exc.value)
        assert str(raw / "recording_configurations" / "probe.json") in message
        assert str(epoch / "probe.json") in message
        for device in DEVICE_PROBE_TYPE_MAP:
            assert str(epoch / device / "probe.json") in message, device

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


class TestResolveEphysFile:
    """Pure path resolver: prefer the .zarr twin under the processed store, else the raw .bin, else error.

    No DB. See SPEC_READ_COMPRESSED.md.
    """

    @pytest.fixture
    def raw_bin_path(self, tmp_path):
        """Return a raw .bin path under tmp_path with a 'raw' component."""
        # tmp_path has no "raw" component, so the one we add is unambiguous.
        bin_path = tmp_path / "raw" / "AEONX1" / "exp" / "dev" / "Probe_AmplifierData_0.bin"
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        return bin_path

    @pytest.fixture
    def zarr_path(self, tmp_path):
        """Return a processed .zarr path under tmp_path."""
        return tmp_path / "processed" / "AEONX1" / "exp" / "dev" / "Probe_AmplifierData_0.zarr"

    def test_prefers_zarr_when_both_present(self, raw_bin_path, zarr_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        raw_bin_path.write_bytes(b"\x00\x00")
        zarr_path.mkdir(parents=True)  # zarr is a directory on disk
        # Pins the raw->processed mapping AND the zarr preference.
        assert resolve_ephys_file(raw_bin_path) == zarr_path

    def test_falls_back_to_bin_when_only_bin(self, raw_bin_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        raw_bin_path.write_bytes(b"\x00\x00")
        assert resolve_ephys_file(raw_bin_path) == raw_bin_path

    def test_returns_zarr_when_only_zarr(self, raw_bin_path, zarr_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        # raw_bin_path's parent dir exists, but no .bin file
        zarr_path.mkdir(parents=True)
        assert resolve_ephys_file(raw_bin_path) == zarr_path

    def test_raises_when_neither_exists(self, raw_bin_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        # nothing created beyond raw_bin_path's parent dir
        with pytest.raises(FileNotFoundError, match=r"processed.*\.zarr.*\.bin"):
            resolve_ephys_file(raw_bin_path)

    def test_raises_on_multiple_raw_components(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        bin_path = tmp_path / "raw" / "sub" / "raw" / "f.bin"
        bin_path.parent.mkdir(parents=True)
        bin_path.write_bytes(b"\x00")
        with pytest.raises(ValueError, match="multiple 'raw' components"):
            resolve_ephys_file(bin_path)

    def test_no_raw_component_falls_back_to_bin(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        bin_path = tmp_path / "data" / "f.bin"  # no "raw" component
        bin_path.parent.mkdir(parents=True)
        bin_path.write_bytes(b"\x00")
        assert resolve_ephys_file(bin_path) == bin_path

    def test_raw_ephys_component_not_treated_as_raw(self, tmp_path):
        # "raw-ephys" is a DataJoint directory_type label, never a physical path
        # component. Exact-component matching must not treat it as the "raw" store
        # root, so no .zarr twin is inferred and we fall back to the .bin.
        from aeon.dj_pipeline.utils.ephys_utils import resolve_ephys_file

        bin_path = tmp_path / "raw-ephys" / "AEONX1" / "f.bin"
        bin_path.parent.mkdir(parents=True)
        bin_path.write_bytes(b"\x00")
        assert resolve_ephys_file(bin_path) == bin_path


FIXTURES = Path(__file__).parents[2] / "fixtures" / "metadata"


def _epoch_with(tmp_path, fixture_name, as_name):
    """Copy a real metadata fixture into an epoch dir under its production filename."""
    epoch = tmp_path / "2026-10-13T094255Z"
    epoch.mkdir(parents=True, exist_ok=True)
    (epoch / as_name).write_text((FIXTURES / fixture_name).read_text())
    return epoch


class TestImplantsEdgeCases:
    """Two states the Implants parser must not conflate with "no block at all"."""

    def test_entries_that_are_all_non_mappings_raise(self):
        """A list of strings is a malformed block, not an absent one.

        Returning None reads as "metadata predates Implants" and silently falls back
        to probe_assignments.json, so the subject-probe mapping comes from the wrong
        source with nothing to notice.
        """
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = {"subject": {"Subject": "AAA-1", "Implants": ["ProbeA", "ProbeB"]}}
        with pytest.raises(ValueError, match="none is a mapping"):
            parse_metadata_implants(meta)

    def test_explicitly_empty_implants_falls_back(self):
        """An empty list declares nothing, which is the documented None case."""
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        assert parse_metadata_implants({"subject": {"Subject": "AAA-1", "Implants": []}}) is None

    def test_subject_missing_beside_implants_raises(self):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = {"subject": {"Implants": [{"Probe": "ProbeA", "InsertionNumber": 1}]}}
        with pytest.raises(ValueError, match="subject.Subject is missing"):
            parse_metadata_implants(meta)


class TestMetadataLayouts:
    """Three metadata layouts must all resolve probe identity and config."""

    def test_probe_configs_headstage_layout(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        epoch = _epoch_with(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json")
        assert parse_metadata_probe_configs(epoch) == {
            "ProbeA": "M01_ProbeA_4Shanks_1500_to_2200um.json",
            "ProbeB": "M01_ProbeB_4Shanks_2000_to_2700um_LFP.json",
        }

    def test_probe_configs_devices_layout_unchanged(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_probe_configs

        epoch = _epoch_with(tmp_path, "ephys_devices_Metadata.yml", "Metadata.yml")
        assert parse_metadata_probe_configs(epoch) == {
            "ProbeA": None,
            "ProbeB": "M81_ProbeB_4Shanks_1000_to_1700_um.json",
        }

    def test_probe_id_headstage_layout(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import get_probe_id, parse_epoch_metadata

        meta = parse_epoch_metadata(_epoch_with(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json"))
        assert get_probe_id(meta, "NeuropixelsV2", "ProbeA") == "23107805942"
        assert get_probe_id(meta, "NeuropixelsV2", "ProbeB") == "23409404271"

    def test_headstage_wins_over_both_older_layouts(self, tmp_path):
        """A transitional file carrying all three blocks must resolve by the headstage.

        parse_metadata_probe_configs returns early on the headstage branch and
        get_probe_id checks it first; without this, a file written during the
        changeover resolves to a different serial and config with no error.
        """
        import json

        from aeon.dj_pipeline.utils.ephys_utils import (
            get_probe_id,
            parse_epoch_metadata,
            parse_metadata_probe_configs,
        )

        epoch = _epoch_with(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json")
        data = json.loads((epoch / "Metadata.json").read_text())
        data["NeuropixelsV2A"] = {
            "ProbeConfiguration": {
                "ProbeInterfaceFileName": r"C:\\configs\\stale_A.json",
                "ProbeSerialNumber": "00000000001",
            }
        }
        data["Devices"] = {
            "NeuropixelsV2e": {
                "ConfigurationA": {
                    "ProbeInterfaceFileName": r"C:\\configs\\older_A.json",
                    "ProbeSerialNumber": "00000000002",
                }
            }
        }
        (epoch / "Metadata.json").write_text(json.dumps(data))

        assert parse_metadata_probe_configs(epoch)["ProbeA"] == "M01_ProbeA_4Shanks_1500_to_2200um.json"
        assert get_probe_id(parse_epoch_metadata(epoch), "NeuropixelsV2", "ProbeA") == "23107805942"

    def test_disabled_probe_is_recognised_as_bool_and_as_string(self):
        """Enable may be a JSON bool or the string "false", depending on the writer.

        enabled_probe_labels and get_probe_id must agree: if one reads a probe as
        enabled and the other returns None, EphysEpochConfig.make aborts the epoch
        with "partially transferred" on a perfectly ingestible recording.
        """
        import json

        from aeon.dj_pipeline.utils.ephys_utils import enabled_probe_labels, get_probe_id

        base = json.loads((FIXTURES / "ephys_headstage_Metadata.json").read_text())
        for flag in (False, "false", "False"):
            data = json.loads(json.dumps(base))
            data["ephys"]["Headstage"]["ProbeA"]["Enable"] = flag
            assert get_probe_id(data, "NeuropixelsV2", "ProbeA") is None, flag
            assert "ProbeA" not in enabled_probe_labels(data), flag
            # ProbeB is untouched and must stay enabled by both readers
            assert get_probe_id(data, "NeuropixelsV2", "ProbeB") is not None, flag
            assert "ProbeB" in enabled_probe_labels(data), flag

    def test_parse_epoch_metadata_prefers_json(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        epoch = _epoch_with(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json")
        (epoch / "Metadata.yml").write_text('{"Workflow": "stale"}')
        assert parse_epoch_metadata(epoch)["workflow"] == "Ephys.bonsai"

    def test_parse_epoch_metadata_falls_back_to_yml(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        epoch = _epoch_with(tmp_path, "ephys_devices_Metadata.yml", "Metadata.yml")
        assert parse_epoch_metadata(epoch)["Workflow"] == "Ephys-AEONX1.bonsai"


class TestParseMetadataImplants:
    """subject.Implants[] carries subject, insertion number and target area per probe."""

    def _meta(self, tmp_path, fixture, as_name):
        from aeon.dj_pipeline.utils.ephys_utils import parse_epoch_metadata

        return parse_epoch_metadata(_epoch_with(tmp_path, fixture, as_name))

    def test_returns_none_for_metadata_without_implants(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = self._meta(tmp_path, "ephys_devices_Metadata.yml", "Metadata.yml")
        assert parse_metadata_implants(meta) is None

    def test_maps_each_probe_to_subject_and_insertion(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = self._meta(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json")
        assert parse_metadata_implants(meta) == {
            "ProbeA": {
                "subject": "AAA-1140901",
                "insertion_number": 1,
                "target_area": "mPFC",
                "implantation_date": "2026-05-20",
            },
            "ProbeB": {
                "subject": "AAA-1140901",
                "insertion_number": 2,
                "target_area": "HPC",
                "implantation_date": "2026-05-20",
            },
        }

    def test_target_area_whitespace_stripped_case_preserved(self):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = {
            "subject": {
                "Subject": "S1",
                "Implants": [{"Probe": "ProbeA", "InsertionNumber": 1, "TargetArea": "  mPFC "}],
            }
        }
        assert parse_metadata_implants(meta)["ProbeA"]["target_area"] == "mPFC"

    def test_missing_optional_fields_become_none(self):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = {"subject": {"Subject": "S1", "Implants": [{"Probe": "ProbeA", "InsertionNumber": 3}]}}
        entry = parse_metadata_implants(meta)["ProbeA"]
        assert entry["target_area"] is None and entry["implantation_date"] is None

    def test_implant_without_insertion_number_raises(self):
        from aeon.dj_pipeline.utils.ephys_utils import parse_metadata_implants

        meta = {"subject": {"Subject": "S1", "Implants": [{"Probe": "ProbeA"}]}}
        with pytest.raises(ValueError, match="InsertionNumber"):
            parse_metadata_implants(meta)


class TestEnabledProbeLabels:
    """Probes the metadata declares enabled, for cross-checking against disk."""

    def test_headstage_layout_lists_enabled_only(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import enabled_probe_labels, parse_epoch_metadata

        meta = parse_epoch_metadata(_epoch_with(tmp_path, "ephys_headstage_Metadata.json", "Metadata.json"))
        assert enabled_probe_labels(meta) == {"ProbeA", "ProbeB"}

    def test_disabled_probe_excluded(self):
        from aeon.dj_pipeline.utils.ephys_utils import enabled_probe_labels

        meta = {
            "ephys": {
                "Headstage": {
                    "ProbeA": {"Enable": True, "ProbeConfiguration": {}},
                    "ProbeB": {"Enable": False, "ProbeConfiguration": {}},
                }
            }
        }
        assert enabled_probe_labels(meta) == {"ProbeA"}

    def test_returns_none_for_layouts_without_an_enable_flag(self, tmp_path):
        """Older layouts express disablement differently; no cross-check is possible."""
        from aeon.dj_pipeline.utils.ephys_utils import enabled_probe_labels, parse_epoch_metadata

        meta = parse_epoch_metadata(_epoch_with(tmp_path, "ephys_devices_Metadata.yml", "Metadata.yml"))
        assert enabled_probe_labels(meta) is None
