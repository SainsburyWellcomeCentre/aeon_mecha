"""Tests for AeonStreamCodec and OnixStreamCodec.

Codec encode/decode tests require real datajoint and are marked integration.
"""

import numpy as np
import pytest
from ephys_factories import (
    make_synthetic_bno055_data,
    make_synthetic_ephys_epoch,
    register_synthetic_experiment,
)

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


class TestOnixStreamCodecRoundTrip:
    def test_round_trip_returns_onix_indexed_dataframe(self, dj_config_integration, tmp_path):
        """Insert a row with stream_df ref, fetch back; codec returns merged ONIX-indexed DF."""
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        experiment_name = "test_codec_round_trip"
        epoch_dir_name = "2024-06-11T10-24-07"
        device_name = "NeuropixelsV2Beta"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

        key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        # Codec returns ONIX-indexed (uint64), NOT HARP datetimes
        assert df.index.dtype == np.uint64
        # Bno055 chunks are staggered against HarpSync windows in the synthetic
        # factory; the codec filters to the sync window's range, so we get a
        # strict subset of the chunk's 100 samples.
        assert 0 < len(df) < 100

    def test_round_trip_no_data_returns_empty_dataframe(self, dj_config_integration, tmp_path):
        """For no-IMU rigs, codec decode returns an empty 13-column DataFrame."""
        from aeon.dj_pipeline import ephys
        from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

        experiment_name = "test_codec_no_data"
        epoch_dir_name = "2024-06-12T10-24-07"
        device_name = "NeuropixelsV2Beta"

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
        # NO make_synthetic_bno055_data — no IMU files
        register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

        ephys.EphysSyncModel.ingest(experiment_name)
        ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

        key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
        df = (ephys.OnixImuChunk & key).fetch1("stream_df")

        assert tuple(df.columns) == IMU_COLUMNS
        assert len(df) == 0
