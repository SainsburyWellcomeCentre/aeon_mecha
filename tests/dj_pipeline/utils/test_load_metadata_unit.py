"""Unit tests for load_metadata.py - pure functions, no database required.

Note: All imports from aeon.dj_pipeline are done inside test methods to avoid
triggering DataJoint connections before testcontainers fixtures are ready.
"""

import pytest


@pytest.mark.unit
class TestToPascalCase:
    """Test snake_case to PascalCase conversion."""

    def test_single_word(self):
        from aeon.dj_pipeline.utils.load_metadata import to_pascal_case

        assert to_pascal_case("video") == "Video"

    def test_two_words(self):
        from aeon.dj_pipeline.utils.load_metadata import to_pascal_case

        assert to_pascal_case("beam_break") == "BeamBreak"

    def test_three_words(self):
        from aeon.dj_pipeline.utils.load_metadata import to_pascal_case

        assert to_pascal_case("weight_raw_data") == "WeightRawData"

    def test_empty_string(self):
        from aeon.dj_pipeline.utils.load_metadata import to_pascal_case

        assert to_pascal_case("") == ""


@pytest.mark.unit
class TestFlattenRigDevices:
    """Test nested rig config flattening."""

    def test_extracts_cameras(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(sample_rig_config)
        assert "CameraTop" in result
        assert result["CameraTop"]["serialNumber"] == "21053810"

    def test_extracts_feeders(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(sample_rig_config)
        assert "Feeder1" in result
        assert result["Feeder1"]["portName"] == "COM3"

    def test_extracts_nest(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(sample_rig_config)
        assert "Nest" in result

    def test_extracts_synchronizers(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(sample_rig_config)
        assert "CameraSynchronizer" in result
        assert "ClockSynchronizer" in result


@pytest.mark.unit
class TestExtractDeviceMapperFromRig:
    """Test device type mapper extraction."""

    def test_extracts_device_types(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(sample_rig_config)
        assert device_type_mapper["CameraTop"] == "SpinnakerVideoSource"
        assert device_type_mapper["Feeder1"] == "UndergroundFeeder"

    def test_extracts_serial_numbers(self, sample_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        _, device_sn = _extract_device_mapper_from_rig(sample_rig_config)
        assert device_sn["CameraTop"] == "21053810"
        assert device_sn["Feeder1"] == "COM3"


@pytest.mark.unit
class TestGetDataReaderMethods:
    """Test @data_reader method extraction from Device class."""

    def test_extracts_data_reader_methods(self, mock_device_class):
        from aeon.dj_pipeline.utils.load_metadata import get_data_reader_methods

        methods = get_data_reader_methods(mock_device_class)
        method_names = [name for name, _ in methods]
        assert "video" in method_names
        assert "position" in method_names

    def test_returns_list_of_tuples(self, mock_device_class):
        from aeon.dj_pipeline.utils.load_metadata import get_data_reader_methods

        result = get_data_reader_methods(mock_device_class)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)

    def test_uses_real_pydantic_base(self, mock_device_class):
        """Verify fixture uses real SpinnakerCamera base class."""
        from swc.aeon.schema.video import SpinnakerCamera

        assert issubclass(mock_device_class, SpinnakerCamera)


# -----------------------------------------------------------------------------
# Tests using sample ForagingABC metadata fixture
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestFlattenRigDevicesWithForagingABC:
    """Test rig flattening with real ForagingABC metadata structure."""

    def test_extracts_all_cameras(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        # ForagingABC has 13 cameras
        expected_cameras = [
            "CameraTop", "CameraNest", "CameraNorth", "CameraEast",
            "CameraSouth", "CameraWest", "CameraLightMonitor",
            "CameraPatch1", "CameraPatch2", "CameraPatch3",
            "CameraPatch4", "CameraPatch5", "CameraPatch6",
        ]
        for camera in expected_cameras:
            assert camera in result, f"Missing camera: {camera}"

    def test_extracts_all_feeders(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        # ForagingABC has 6 feeders
        for i in range(1, 7):
            assert f"Feeder{i}" in result

    def test_extracts_nest(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        assert "Nest" in result
        assert result["Nest"]["portName"] == "COM7"

    def test_extracts_synchronizers(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        assert "CameraSynchronizer" in result
        assert "ClockSynchronizer" in result

    def test_camera_serial_numbers_preserved(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        assert result["CameraTop"]["serialNumber"] == "23032909"
        assert result["CameraNest"]["serialNumber"] == "23031407"

    def test_feeder_port_names_preserved(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig_config)
        assert result["Feeder1"]["portName"] == "COM13"
        assert result["Feeder4"]["portName"] == "COM3"


@pytest.mark.unit
class TestExtractDeviceMapperWithForagingABC:
    """Test device mapper extraction with real ForagingABC metadata."""

    def test_camera_device_types(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        # All cameras should map to SpinnakerVideoSource
        for camera in ["CameraTop", "CameraNest", "CameraPatch1"]:
            assert device_type_mapper[camera] == "SpinnakerVideoSource"

    def test_feeder_device_types(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        # All feeders should map to UndergroundFeeder
        for feeder in ["Feeder1", "Feeder2", "Feeder6"]:
            assert device_type_mapper[feeder] == "UndergroundFeeder"

    def test_nest_device_type(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        assert device_type_mapper["Nest"] == "WeightScale"

    def test_synchronizer_device_types(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        assert device_type_mapper["CameraSynchronizer"] == "CameraController"
        assert device_type_mapper["ClockSynchronizer"] == "TimestampGenerator"

    def test_camera_serial_numbers(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        _, device_sn = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        assert device_sn["CameraTop"] == "23032909"
        assert device_sn["CameraPatch2"] == "21053811"

    def test_feeder_port_names_as_serial(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        _, device_sn = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        # Feeders use portName as device serial
        assert device_sn["Feeder1"] == "COM13"
        assert device_sn["Feeder6"] == "COM4"

    def test_device_count(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import _extract_device_mapper_from_rig

        device_type_mapper, _ = _extract_device_mapper_from_rig(foraging_abc_rig_config)
        # 13 cameras + 6 feeders + 1 nest + 2 synchronizers + 1 light cycle = 23
        assert len(device_type_mapper) == 23


@pytest.mark.unit
class TestExtractActiveRegionsWithForagingABC:
    """Test active region extraction with real ForagingABC metadata."""

    def test_extracts_camera_tracking_regions(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import extract_active_regions

        regions = extract_active_regions(foraging_abc_rig_config)
        # CameraTop has Arena and Nest tracking regions
        assert "CameraTop_Arena" in regions
        assert "CameraTop_Nest" in regions

    def test_extracts_activity_center(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import extract_active_regions

        regions = extract_active_regions(foraging_abc_rig_config)
        assert "ActivityCenter" in regions
        assert regions["ActivityCenter"]["camera"] == "CameraTop"

    def test_region_structure(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import extract_active_regions

        regions = extract_active_regions(foraging_abc_rig_config)
        arena = regions["CameraTop_Arena"]
        assert "regions" in arena
        assert "threshold" in arena
        assert arena["threshold"] == 100

    def test_cameras_without_tracking_excluded(self, foraging_abc_rig_config):
        from aeon.dj_pipeline.utils.load_metadata import extract_active_regions

        regions = extract_active_regions(foraging_abc_rig_config)
        # CameraNest has cameraTracking: null, should not appear
        assert "CameraNest_Arena" not in regions
