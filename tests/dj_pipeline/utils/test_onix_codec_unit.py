"""Unit tests for OnixStreamCodec encode path."""

import pytest

pytestmark = pytest.mark.unit


class TestOnixStreamCodecEncode:
    def test_encodes_valid_dict(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        ref = {
            "experiment_name": "exp01",
            "epoch_start": "2024-06-04 10:24:07",
            "sync_start": "2024-06-04 11:00:00",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055",
        }
        encoded = codec.encode(ref)
        assert encoded == ref

    def test_rejects_non_dict(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(TypeError, match="OnixStreamCodec expects a dict"):
            codec.encode("not-a-dict")

    def test_rejects_dict_missing_keys(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        codec = OnixStreamCodec()
        with pytest.raises(ValueError, match="missing required keys"):
            codec.encode({"experiment_name": "exp01"})

    def test_codec_name(self):
        from aeon.dj_pipeline.utils.codec import OnixStreamCodec

        assert OnixStreamCodec.name == "aeon_onix_stream"
