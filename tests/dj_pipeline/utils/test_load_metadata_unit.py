"""Unit tests for load_metadata.py - pure functions, no database required.

Note: All imports from aeon.dj_pipeline are done inside test methods to avoid
triggering DataJoint connections before testcontainers fixtures are ready.
"""

import pytest

pytestmark = pytest.mark.unit


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


class TestFlattenRigDevices:
    """Test typed Rig flattening into device_name -> config map."""

    def test_extracts_cameras(self, test_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(test_rig)
        assert "CameraTop" in result
        assert result["CameraTop"]["serialNumber"] == "21053810"

    def test_extracts_feeders(self, test_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(test_rig)
        assert "Feeder1" in result
        assert result["Feeder1"]["portName"] == "COM3"

    def test_excludes_non_device_fields(self, test_rig):
        """Fields whose classes have no @data_reader methods must be skipped."""
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(test_rig)
        # TestRig only declares cameras + feeders, both with @data_reader.
        # Anything else (synchronizers, light cycles) would be excluded by design.
        assert set(result.keys()) == {"CameraTop", "CameraSide", "Feeder1", "Feeder2"}


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
# Tests using sample ForagingABC metadata fixture (typed Rig)
# -----------------------------------------------------------------------------


class TestFlattenRigDevicesWithForagingABC:
    """Test rig flattening with real ForagingABC typed Rig."""

    def test_extracts_all_cameras(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        # ForagingABC has 13 cameras
        expected_cameras = [
            "CameraTop",
            "CameraNest",
            "CameraNorth",
            "CameraEast",
            "CameraSouth",
            "CameraWest",
            "CameraLightMonitor",
            "CameraPatch1",
            "CameraPatch2",
            "CameraPatch3",
            "CameraPatch4",
            "CameraPatch5",
            "CameraPatch6",
        ]
        for camera in expected_cameras:
            assert camera in result, f"Missing camera: {camera}"

    def test_extracts_all_feeders(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        # ForagingABC has 6 feeders
        for i in range(1, 7):
            assert f"Feeder{i}" in result

    def test_extracts_nest(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        assert "Nest" in result
        assert result["Nest"]["portName"] == "COM7"

    def test_excludes_synchronizers_and_light_cycle(self, foraging_abc_rig):
        """Synchronizers and LightCycle have no @data_reader methods, so excluded."""
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        assert "CameraSynchronizer" not in result
        assert "ClockSynchronizer" not in result
        assert "LightCycle" not in result

    def test_camera_serial_numbers_preserved(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        assert result["CameraTop"]["serialNumber"] == "23032909"
        assert result["CameraNest"]["serialNumber"] == "23031407"

    def test_feeder_port_names_preserved(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import flatten_rig_devices

        result = flatten_rig_devices(foraging_abc_rig)
        assert result["Feeder1"]["portName"] == "COM13"
        assert result["Feeder4"]["portName"] == "COM3"


class TestGetDeviceMapperFromRigWithForagingABC:
    """Test typed device mapper extraction with real ForagingABC Rig."""

    def test_camera_device_types(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        device_type_mapper, _ = get_device_mapper_from_rig(foraging_abc_rig)
        # ForagingABC Camera class is named "Camera" (subclass of SpinnakerCamera)
        for camera in ["CameraTop", "CameraNest", "CameraPatch1"]:
            assert device_type_mapper[camera] == "Camera"

    def test_feeder_device_types(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        device_type_mapper, _ = get_device_mapper_from_rig(foraging_abc_rig)
        # ForagingABC Feeder class is named "Feeder"
        for feeder in ["Feeder1", "Feeder2", "Feeder6"]:
            assert device_type_mapper[feeder] == "Feeder"

    def test_nest_device_type(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        device_type_mapper, _ = get_device_mapper_from_rig(foraging_abc_rig)
        # ForagingABC nest class is "ActivityWeightScale"
        assert device_type_mapper["Nest"] == "ActivityWeightScale"

    def test_camera_serial_numbers(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        _, device_sn = get_device_mapper_from_rig(foraging_abc_rig)
        assert device_sn["CameraTop"] == "23032909"
        assert device_sn["CameraPatch2"] == "21053811"

    def test_feeder_port_names_as_serial(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        _, device_sn = get_device_mapper_from_rig(foraging_abc_rig)
        # Feeders use portName as device serial
        assert device_sn["Feeder1"] == "COM13"
        assert device_sn["Feeder6"] == "COM4"

    def test_synchronizers_and_light_cycle_excluded(self, foraging_abc_rig):
        """Only Device classes with @data_reader methods are mapped."""
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        device_type_mapper, _ = get_device_mapper_from_rig(foraging_abc_rig)
        assert "CameraSynchronizer" not in device_type_mapper
        assert "ClockSynchronizer" not in device_type_mapper
        assert "LightCycle" not in device_type_mapper

    def test_device_count(self, foraging_abc_rig):
        from aeon.dj_pipeline.utils.load_metadata import get_device_mapper_from_rig

        device_type_mapper, _ = get_device_mapper_from_rig(foraging_abc_rig)
        # 13 cameras + 6 feeders + 1 nest + 1 environment (light_events @data_reader) = 21
        # Excluded: synchronizers and lightCycle (no @data_reader methods)
        assert len(device_type_mapper) == 21


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
