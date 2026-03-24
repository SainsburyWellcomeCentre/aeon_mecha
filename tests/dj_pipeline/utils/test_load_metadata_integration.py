"""Integration tests for load_metadata.py - requires MySQL via testcontainers.

These tests verify database operations against real DataJoint tables:
- StreamType catalog population
- DeviceType and DeviceType.Stream insertion
- Device registration
- FK constraint handling

Uses real Pydantic classes and real @data_reader decorator from swc.aeon.schema (swc-aeon).

Note: All imports from aeon.dj_pipeline are done inside test methods to avoid
triggering DataJoint connections before testcontainers fixtures are ready.
"""

import importlib.util
import uuid

import pytest

pytestmark = pytest.mark.integration


class TestGetDataReaderMethods:
    """Test @data_reader method extraction with real Pydantic classes."""

    def test_extracts_video_stream(self, mock_device_class, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_data_reader_methods

        methods = get_data_reader_methods(mock_device_class)
        method_names = [name for name, _ in methods]
        assert "video" in method_names

    def test_extracts_position_stream(self, mock_device_class, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_data_reader_methods

        methods = get_data_reader_methods(mock_device_class)
        method_names = [name for name, _ in methods]
        assert "position" in method_names

    def test_uses_real_spinnaker_base(self, mock_device_class, pipeline_integration):
        from swc.aeon.schema.video import SpinnakerCamera

        assert issubclass(mock_device_class, SpinnakerCamera)

    def test_returns_snake_case_names(self, mock_device_class, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_data_reader_methods

        methods = get_data_reader_methods(mock_device_class)
        # All method names should be snake_case (no uppercase)
        for name, _ in methods:
            assert name == name.lower(), f"Expected snake_case, got: {name}"


class TestGetDeviceInfo:
    """Test Rig parsing for device/stream info extraction."""

    def test_extracts_all_cameras(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        # Keys are device names
        assert "CameraTop" in device_info
        assert "CameraSide" in device_info

    def test_extracts_all_feeders(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        # Keys are device names
        assert "Feeder1" in device_info
        assert "Feeder2" in device_info

    def test_device_info_structure(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        camera_info = device_info["CameraTop"]
        # Structure: flat lists of stream_type, stream_reader, stream_hash, stream_reader_kwargs
        assert "stream_type" in camera_info
        assert "stream_reader" in camera_info
        assert "stream_hash" in camera_info
        assert "stream_reader_kwargs" in camera_info

    def test_stream_type_is_pascal_case(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        camera_info = device_info["CameraTop"]
        # "video" method -> "Video" stream_type
        assert "Video" in camera_info["stream_type"]

    def test_stream_reader_is_class_path(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        camera_info = device_info["CameraTop"]
        # Should be fully qualified class path
        for reader_path in camera_info["stream_reader"]:
            assert "." in reader_path

    def test_stream_hash_is_uuid(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_device_info

        device_info = get_device_info(test_rig)
        camera_info = device_info["CameraTop"]
        for hash_val in camera_info["stream_hash"]:
            assert isinstance(hash_val, uuid.UUID)


class TestGetStreamEntries:
    """Test stream entry generation from Rig."""

    def test_returns_list_of_dicts(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_stream_entries

        entries = get_stream_entries(test_rig)
        assert isinstance(entries, list)
        assert all(isinstance(e, dict) for e in entries)

    def test_entry_has_required_keys(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_stream_entries

        entries = get_stream_entries(test_rig)
        # New schema: no stream_reader_kwargs
        required_keys = {"stream_type", "stream_reader", "stream_hash"}
        for entry in entries:
            assert required_keys <= set(entry.keys())

    def test_stream_reader_is_valid_class_path(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_stream_entries

        entries = get_stream_entries(test_rig)
        for entry in entries:
            # stream_reader should be a fully qualified class path
            assert "." in entry["stream_reader"]
            assert entry["stream_reader"].startswith("swc.aeon.io.reader.")

    def test_multiple_devices_produce_entries(self, test_rig, pipeline_integration):
        from aeon.dj_pipeline.utils.load_metadata import get_stream_entries

        entries = get_stream_entries(test_rig)
        # 2 cameras x 2 streams + 2 feeders x 2 streams = 8 entries
        assert len(entries) >= 4  # At minimum from cameras


class TestInsertStreamTypes:
    """Test StreamType catalog population."""

    def test_inserts_stream_types(self, pipeline_integration, test_rig, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_stream_types

        # Monkeypatch schema_name to use test schema
        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        initial_count = len(streams.StreamType())

        insert_stream_types(test_rig)

        assert len(streams.StreamType()) > initial_count

    def test_handles_duplicates(self, pipeline_integration, test_rig, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]

        insert_stream_types(test_rig)
        count_after_first = len(streams.StreamType())

        # Second call should not raise or create duplicates
        insert_stream_types(test_rig)

        assert len(streams.StreamType()) == count_after_first

    def test_stream_type_queryable(self, pipeline_integration, test_rig, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        insert_stream_types(test_rig)

        # Should be able to fetch by stream_type name
        video_entries = (streams.StreamType & {"stream_type": "Video"}).fetch(as_dict=True)
        assert len(video_entries) >= 1


class TestInsertDeviceTypes:
    """Test DeviceType and Device catalog population."""

    def test_inserts_device_types(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types, insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        # Pre-insert StreamTypes to avoid FK cascade
        insert_stream_types(test_rig)

        initial_count = len(streams.DeviceType())
        insert_device_types(test_rig, metadata_filepath)

        assert len(streams.DeviceType()) > initial_count

    def test_inserts_device_type_streams(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types, insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        insert_stream_types(test_rig)
        insert_device_types(test_rig, metadata_filepath)

        # Should have associations between DeviceType and StreamType
        assert len(streams.DeviceType.Stream()) > 0

    def test_inserts_device_names(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types, insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        insert_stream_types(test_rig)
        insert_device_types(test_rig, metadata_filepath)

        # DeviceName should be populated with device instance names
        device_names = streams.DeviceName().fetch(as_dict=True)
        names = [d["device_name"] for d in device_names]
        assert "CameraTop" in names
        assert "CameraSide" in names
        assert "Feeder1" in names
        assert "Feeder2" in names

    def test_inserts_devices(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types, insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        insert_stream_types(test_rig)
        insert_device_types(test_rig, metadata_filepath)

        # Cameras have serial_number, feeders have port_name (optional hardware tracking)
        devices = streams.Device().fetch(as_dict=True)
        serial_numbers = [d["device_serial_number"] for d in devices]
        assert "21053810" in serial_numbers  # CameraTop serial

    def test_handles_existing_device_types(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types, insert_stream_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        insert_stream_types(test_rig)
        insert_device_types(test_rig, metadata_filepath)
        count_after_first = len(streams.DeviceType())

        # Second call should not raise error
        insert_device_types(test_rig, metadata_filepath)

        assert len(streams.DeviceType()) == count_after_first


class TestInsertDeviceTypesFKHandling:
    """Test FK constraint handling in insert_device_types()."""

    def test_fk_constraint_triggers_stream_type_insert(
        self, pipeline_integration, test_rig, tmp_path, monkeypatch, clean_streams_tables
    ):
        """Verify FK failure triggers automatic insert_stream_types()."""
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        # Tables are already clean from clean_streams_tables fixture
        # This should succeed by calling insert_stream_types() on FK failure
        insert_device_types(test_rig, metadata_filepath)

        # Verify both StreamType and DeviceType.Stream are populated
        assert len(streams.StreamType()) > 0
        assert len(streams.DeviceType.Stream()) > 0

    def test_non_fk_errors_are_reraised(self, pipeline_integration, test_rig, tmp_path, monkeypatch):
        """Verify non-FK DataJointErrors are re-raised."""
        import datajoint as dj

        from aeon.dj_pipeline.utils import load_metadata, streams_maker
        from aeon.dj_pipeline.utils.load_metadata import (
            get_device_mapper_from_rig,
            insert_device_types,
            insert_stream_types,
        )

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        # Pre-insert StreamTypes to reach DeviceType.Stream insertion
        insert_stream_types(test_rig)

        # Also insert DeviceType entries first
        device_mapper, _ = get_device_mapper_from_rig(test_rig, metadata_filepath)
        device_types = [{"device_type": dt} for dt in set(device_mapper.values())]
        streams.DeviceType.insert(device_types, skip_duplicates=True)

        def mock_insert(*args, **kwargs):
            """Mock insert that raises non-FK error regardless of arguments."""
            raise dj.DataJointError("Connection refused")

        monkeypatch.setattr(streams.DeviceType.Stream, "insert", mock_insert)

        # Patch dj.VirtualModule in load_metadata to return our pre-patched streams
        monkeypatch.setattr(
            load_metadata,
            "dj",
            type(
                "MockDJ",
                (),
                {
                    "VirtualModule": lambda *args, **kwargs: streams,
                    "DataJointError": dj.DataJointError,
                    "logger": dj.logger,
                },
            )(),
        )

        with pytest.raises(dj.DataJointError, match="Connection refused"):
            insert_device_types(test_rig, metadata_filepath)

    def test_recovers_from_fk_failure(
        self, pipeline_integration, test_rig, tmp_path, monkeypatch, clean_streams_tables
    ):
        """Verify DeviceType.Stream insertion succeeds after FK recovery."""
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import insert_device_types

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])

        streams = pipeline_integration["streams"]
        metadata_filepath = tmp_path / "Metadata.json"
        metadata_filepath.write_text("{}")

        # Start with empty tables (clean_streams_tables fixture)
        # Insert device types without stream types - should trigger FK handling
        insert_device_types(test_rig, metadata_filepath)

        # Verify recovery was successful
        device_type_streams = streams.DeviceType.Stream().fetch(as_dict=True)
        assert len(device_type_streams) > 0

        # Verify the structure is correct (device_type + stream_hash pairs)
        for entry in device_type_streams:
            assert "device_type" in entry
            assert "stream_hash" in entry


@pytest.mark.skipif(
    not importlib.util.find_spec("swc.aeon_exp"),
    reason="swc-aeon-rigs-foragingabc not installed (install with --extra test-golden)",
)
class TestPopulateCatalogFromPydantic:
    """Test Step 1 catalog population from Pydantic class hierarchy."""

    def test_populates_catalog_tables(self, pipeline_integration, monkeypatch, clean_streams_tables):
        """Should populate StreamType, DeviceType, and DeviceType.Stream."""
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import (
            get_experiment_pydantic,
            populate_catalog_from_pydantic,
        )

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])
        streams = pipeline_integration["streams"]

        # Use real ForagingABC Experiment class
        experiment_class = get_experiment_pydantic("swc.aeon_exp.foragingABC.experiment:Experiment")
        populate_catalog_from_pydantic(experiment_class)

        # All three catalog tables should be populated
        assert len(streams.StreamType()) > 0
        assert len(streams.DeviceType()) > 0
        assert len(streams.DeviceType.Stream()) > 0

    def test_idempotent(self, pipeline_integration, monkeypatch, clean_streams_tables):
        """Calling twice should not create duplicates."""
        from aeon.dj_pipeline.utils import streams_maker
        from aeon.dj_pipeline.utils.load_metadata import (
            get_experiment_pydantic,
            populate_catalog_from_pydantic,
        )

        monkeypatch.setattr(streams_maker, "schema_name", pipeline_integration["schema_name"])
        streams = pipeline_integration["streams"]

        experiment_class = get_experiment_pydantic("swc.aeon_exp.foragingABC.experiment:Experiment")

        populate_catalog_from_pydantic(experiment_class)
        first_count = len(streams.StreamType())

        populate_catalog_from_pydantic(experiment_class)
        second_count = len(streams.StreamType())

        assert second_count == first_count
