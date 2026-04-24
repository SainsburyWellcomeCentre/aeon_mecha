"""Tests for AeonStreamCodec.

Codec encode/decode tests require real datajoint and are marked integration.
"""

import pytest

pytestmark = pytest.mark.integration


class TestAeonStreamCodecEncode:
    def test_valid_encode(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        value = {
            "stream_type": "Encoder",
            "experiment_name": "test",
            "device_name": "Feeder1",
            "chunk_start": "2025-01-01 00:00:00",
            "chunk_end": "2025-01-01 01:00:00",
            "epoch_start": "2025-01-01 00:00:00",
        }
        result = codec.encode(value)
        assert result == value

    def test_missing_keys_raises(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"stream_type": "Encoder"})

    def test_non_dict_raises(self, dj_config_integration):
        from aeon.dj_pipeline.utils.codec import AeonStreamCodec

        codec = AeonStreamCodec()
        with pytest.raises(TypeError, match="expects a dict"):
            codec.encode("not a dict")
