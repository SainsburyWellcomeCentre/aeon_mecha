"""Integration tests for OnixStreamCodec decode round-trip.

These tests verify that ``<aeon_onix_stream>`` returns ONIX-clock-indexed DataFrames
on fetch (not HARP-indexed), catching any regression where the codec accidentally
applies the sync regression.
"""

import numpy as np
import pytest
from _synthetic_ephys_fixtures import (
    _make_synthetic_bno055_data,
    _make_synthetic_ephys_epoch,
    _register_synthetic_experiment,
)

pytestmark = pytest.mark.integration


def test_round_trip_returns_onix_indexed_dataframe(dj_config_integration, tmp_path):
    """Insert a row with stream_df ref, fetch back; codec returns merged ONIX-indexed DF."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    experiment_name = "test_codec_round_trip"
    epoch_dir_name = "2024-06-11T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
    _make_synthetic_bno055_data(raw_dir, epoch_dir_name, device_name, n_chunks=1)
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

    key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
    df = (ephys.OnixImuChunk & key).fetch1("stream_df")

    assert tuple(df.columns) == IMU_COLUMNS
    # Codec returns ONIX-indexed (uint64), NOT HARP datetimes
    assert df.index.dtype == np.uint64
    assert len(df) == 100


def test_round_trip_no_data_returns_empty_dataframe(dj_config_integration, tmp_path):
    """For no-IMU rigs, codec decode returns an empty 13-column DataFrame."""
    from aeon.dj_pipeline import ephys
    from aeon.dj_pipeline.utils.onix_imu import IMU_COLUMNS

    experiment_name = "test_codec_no_data"
    epoch_dir_name = "2024-06-12T10-24-07"
    device_name = "NeuropixelsV2Beta"

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _make_synthetic_ephys_epoch(raw_dir, epoch_dir_name, device_name, n_chunks=1)
    # NO _make_synthetic_bno055_data — no IMU files
    _register_synthetic_experiment(tmp_path, raw_dir, experiment_name, epoch_dir_name)

    ephys.EphysSyncModel.ingest(experiment_name)
    ephys.OnixImuChunk.populate({"experiment_name": experiment_name})

    key = (ephys.OnixImuChunk & {"experiment_name": experiment_name}).fetch1("KEY")
    df = (ephys.OnixImuChunk & key).fetch1("stream_df")

    assert tuple(df.columns) == IMU_COLUMNS
    assert len(df) == 0
