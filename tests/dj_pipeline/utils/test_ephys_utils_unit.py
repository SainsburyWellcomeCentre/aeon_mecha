"""Unit tests for ephys_utils probe assignment parsing.

Tests target the internal JSON parsing helper (_parse_probe_assignments_file),
which is a pure function with no database dependencies. Carry-forward logic
(which queries EphysEpoch.Insertion) is tested during HPC validation.

Note: imports from aeon.dj_pipeline are done inside test methods, not at module
level. pytest imports test modules during collection -- before any fixtures run --
so a module-level import would trigger aeon/dj_pipeline/__init__.py, which
activates the streams schema and attempts a DB connection.
"""

import json

import pytest

pytestmark = pytest.mark.unit


class TestParseProbeAssignmentsFile:
    """Test the JSON file parsing helper (pure function, no DB)."""

    def test_single_probe(self, tmp_path):
        """Single probe serial maps to correct label."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({
            "version": 1,
            "probe_assignments": {
                "23299108854": {"subject": "BAA-1104545"}
            }
        }))
        probe_info = {"ProbeB": "23299108854"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file
        result = _parse_probe_assignments_file(json_path, probe_info)

        assert result == {"ProbeB": {"subject": "BAA-1104545"}}

    def test_multiple_probes(self, tmp_path):
        """Multiple probes each map to their label."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({
            "version": 1,
            "probe_assignments": {
                "11111": {"subject": "BAA-1104545"},
                "22222": {"subject": "BAA-1104545"},
            }
        }))
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
        json_path.write_text(json.dumps({
            "version": 1,
            "probe_assignments": {
                "11111": {"subject": "BAA-1104545"},
                "99999": {"subject": "BAA-9999999"},
            }
        }))
        probe_info = {"ProbeB": "11111"}  # only one probe active

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file
        result = _parse_probe_assignments_file(json_path, probe_info)

        assert result == {"ProbeB": {"subject": "BAA-1104545"}}

    def test_missing_serial_raises(self, tmp_path):
        """Serial in probe_info but not in JSON raises ValueError."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({
            "version": 1,
            "probe_assignments": {
                "11111": {"subject": "BAA-1104545"},
            }
        }))
        probe_info = {"ProbeA": "11111", "ProbeB": "99999"}

        from aeon.dj_pipeline.utils.ephys_utils import _parse_probe_assignments_file
        with pytest.raises(ValueError, match="99999"):
            _parse_probe_assignments_file(json_path, probe_info)

    def test_missing_version_raises(self, tmp_path):
        """JSON without version field raises ValueError."""
        json_path = tmp_path / "probe_assignments.json"
        json_path.write_text(json.dumps({
            "probe_assignments": {"11111": {"subject": "X"}}
        }))
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
