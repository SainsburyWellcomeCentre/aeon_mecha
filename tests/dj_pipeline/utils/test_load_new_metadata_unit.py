"""Unit tests for load_new_metadata.py - pure functions, no database required."""

import pytest

from aeon.dj_pipeline.utils.load_new_metadata import (
    to_pascal_case,
    _flatten_rig_devices,
    _infer_device_type_from_rig,
    _extract_device_mapper_from_rig,
    extract_stream_types_from_device,
)


@pytest.mark.unit
class TestToPascalCase:
    """Test snake_case to PascalCase conversion."""

    def test_single_word(self):
        assert to_pascal_case("video") == "Video"

    def test_two_words(self):
        assert to_pascal_case("beam_break") == "BeamBreak"

    def test_three_words(self):
        assert to_pascal_case("weight_raw_data") == "WeightRawData"

    def test_empty_string(self):
        assert to_pascal_case("") == ""


@pytest.mark.unit
class TestFlattenRigDevices:
    """Test nested rig config flattening."""

    def test_extracts_cameras(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "CameraTop" in result
        assert result["CameraTop"]["serialNumber"] == "21053810"

    def test_extracts_feeders(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "Feeder1" in result
        assert result["Feeder1"]["portName"] == "COM3"

    def test_extracts_nest(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "Nest" in result

    def test_extracts_synchronizers(self, sample_rig_config):
        result = _flatten_rig_devices(sample_rig_config)
        assert "CameraSynchronizer" in result
        assert "ClockSynchronizer" in result


@pytest.mark.unit
class TestInferDeviceTypeFromRig:
    """Test device type inference from rig structure."""

    def test_camera_type(self, sample_rig_config):
        assert _infer_device_type_from_rig("CameraTop", sample_rig_config) == "SpinnakerVideoSource"

    def test_feeder_type(self, sample_rig_config):
        assert _infer_device_type_from_rig("Feeder1", sample_rig_config) == "UndergroundFeeder"

    def test_nest_type(self, sample_rig_config):
        assert _infer_device_type_from_rig("Nest", sample_rig_config) == "WeightScale"

    def test_unknown_device(self, sample_rig_config):
        assert _infer_device_type_from_rig("UnknownDevice", sample_rig_config) is None


@pytest.mark.unit
class TestExtractDeviceMapperFromRig:
    """Test device type mapper extraction."""

    def test_extracts_device_types(self, sample_rig_config):
        device_type_mapper, _ = _extract_device_mapper_from_rig(sample_rig_config)
        assert device_type_mapper["CameraTop"] == "SpinnakerVideoSource"
        assert device_type_mapper["Feeder1"] == "UndergroundFeeder"

    def test_extracts_serial_numbers(self, sample_rig_config):
        _, device_sn = _extract_device_mapper_from_rig(sample_rig_config)
        assert device_sn["CameraTop"] == "21053810"
        assert device_sn["Feeder1"] == "COM3"


@pytest.mark.unit
class TestExtractStreamTypesFromDevice:
    """Test @data_reader method extraction from Device class."""

    def test_extracts_data_reader_methods(self, mock_device_class):
        stream_types = extract_stream_types_from_device(mock_device_class)
        assert "video" in stream_types
        assert "position" in stream_types

    def test_returns_list(self, mock_device_class):
        result = extract_stream_types_from_device(mock_device_class)
        assert isinstance(result, list)

    def test_uses_real_pydantic_base(self, mock_device_class):
        """Verify fixture uses real SpinnakerCamera base class."""
        # Import from swc.aeon.schema (swc-aeon) which has @data_reader support
        from swc.aeon.schema.video import SpinnakerCamera

        assert issubclass(mock_device_class, SpinnakerCamera)
