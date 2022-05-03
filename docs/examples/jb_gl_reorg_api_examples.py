"""
Examples of how to use aeon.api for both exp01 and exp02 data:
Gets position data, pellet data, patch wheel data, mouse weight data, and raw vid data.
"""
from pathlib import Path
from datetime import datetime, time
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
from dotmap import DotMap

from aeon.io import api, stream
from aeon.io.stream import Device
from aeon import preprocessing as pp


# Set exp root paths and time periods of interest
exp01_root = Path('/ceph/aeon/test2/experiment0.1')
exp02_root = Path('/ceph/aeon/test2/experiment0.2')
start_ts1 = pd.Timestamp('2021-11-26')
end_ts1 = pd.Timestamp('2021-11-27')
start_ts2 = pd.Timestamp('2022-02-24')
end_ts2 = pd.Timestamp('2022-02-26')
spec_ts2 = pd.date_range("2022-02-24 09:00:00", periods=4*60*60*1000,
                         freq=pd.Timedelta('0.001s'))  # 4 hours worth of data

# Set schemas
exp01 = DotMap([
    Device("SessionData", stream.session),
    Device("FrameTop", stream.video, stream.position),
    Device("FrameEast", stream.video),
    Device("FrameGate", stream.video),
    Device("FrameNorth", stream.video),
    Device("FramePatch1", stream.video),
    Device("FramePatch2", stream.video),
    Device("FrameSouth", stream.video),
    Device("FrameWest", stream.video),
    Device("Patch1", stream.depletionFunction, stream.encoder, stream.feeder),
    Device("Patch2", stream.depletionFunction, stream.encoder, stream.feeder)
])
exp02 = DotMap([
    Device("Metadata", stream.metadata),
    Device("ExperimentalMetadata", stream.environment, stream.messageLog),
    Device("CameraTop", stream.video, stream.position, stream.region),
    Device("CameraEast", stream.video),
    Device("CameraNest", stream.video),
    Device("CameraNorth", stream.video),
    Device("CameraPatch1", stream.video),
    Device("CameraPatch2", stream.video),
    Device("CameraSouth", stream.video),
    Device("CameraWest", stream.video),
    Device("Nest", stream.weight),
    Device("Patch1", stream.patch),
    Device("Patch2", stream.patch)
])

# Set column names & harp bitmasks for returned data
position_cols = ['x', 'y', 'angle', 'major', 'minor', 'area']
pellet_trig_cols = ['trig_event_harp_bitmask']
trig_bitmask = pp.HARP_EVENT_BITMASK.pellet_trigger
pellet_det_cols = ['det_event_harp_bitmask']
det_bitmask = pp.HARP_EVENT_BITMASK.pellet_detected_in
wheel_enc_cols = ['angle', 'intensity']
weight_cols = ['weight', 'stable?']

# example of data_dict: each key represents a datastream, and each corresponding
# value contains the corresponding datastream files':
# 1) prefix; 2) extension; 3) "read_data" function
exp02_data_dict = DotMap({
    'position': (
        'CameraTop_200', 'bin', lambda file: api.read_harp(file, cols=position_cols)),
    'pellet_triggered_patch1': (
        'Patch1_35', 'bin', lambda file: api.read_harp(file, cols=pellet_trig_cols)),
    'pellet_triggered_patch2': (
        'Patch2_35', 'bin', lambda file: api.read_harp(file, cols=pellet_trig_cols)),
    'pellet_detected_patch1': (
        'Patch1_32', 'bin', lambda file: api.read_harp(file, cols=pellet_det_cols)),
    'pellet_detected_patch2': (
        'Patch2_32', 'bin', lambda file: api.read_harp(file, cols=pellet_det_cols)),
    'wheel_encoder_patch1': (
        'Patch1_90', 'bin', lambda file: api.read_harp(file, cols=wheel_enc_cols)),
    'wheel_encoder_patch2': (
        'Patch2_90', 'bin', lambda file: api.read_harp(file, cols=wheel_enc_cols)),
    'weight': (
        'Nest_200', 'bin', lambda file: api.read_harp(file, cols=weight_cols)),
})
exp01_data_dict = exp02_data_dict.copy()
exp01_data_dict.weight = (
    'WeightData', 'csv', lambda file: api.read_csv(file, cols=weight_cols))
# @todo we could have a config file in the exp root that specifies the "read data"
# function (as a lambda) for each datastream, and pass this into a function
# `gen_data_dict()` in the api that generates the data dict for the given dataset.

# Load data:
# position data from a specified set of timestamps and timestamp tolerance
position_data1 = api.load(exp_root=exp02_root, datastream=exp02_data_dict.position,
                          spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.1s'))

# position data from a start and end timestamp
position_data2 = api.load(exp_root=exp02_root, datastream=exp02_data_dict.position,
                          start_ts=start_ts2, end_ts=end_ts2)

# patch 1 pellet triggered data
pellet_trig_data_p1 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.pellet_triggered_patch1,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
# since `load()` reindexes at `spec_ts`, we get nans for `spec_ts` times when there
# was no event: this is a feature, and we can simply ignore these in the returned data:
pellet_trig_ts_p1 = pp.apply_bitmask(pellet_trig_data_p1, trig_bitmask).index

# patch 2 pellet triggered data
pellet_trig_data_p2 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.pellet_triggered_patch2,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
pellet_trig_ts_p2 = pp.apply_bitmask(pellet_trig_data_p2, trig_bitmask).index

# patch 1 pellet detected data
pellet_det_data_p1 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.pellet_detected_patch1,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
pellet_det_ts_p1 = pp.apply_bitmask(pellet_det_data_p1, det_bitmask).index

# patch 2 pellet detected data
pellet_det_data_p2 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.pellet_detected_patch2,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
pellet_det_ts_p2 = pp.apply_bitmask(pellet_det_data_p2, det_bitmask).index

# patch 1 wheel data
wheel_enc_data_p1 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.wheel_encoder_patch1,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
# remove nans
enc_angle = wheel_enc_data_p1.angle
enc_angle = enc_angle[np.where(~np.isnan(enc_angle))[0]]
wheel_cum_dist_p1 = pp.calc_wheel_cum_dist(enc_angle)

# patch 2 wheel data
wheel_enc_data_p2 = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.wheel_encoder_patch2,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
enc_angle = wheel_enc_data_p2.angle
enc_angle = enc_angle[np.where(~np.isnan(enc_angle))[0]]
wheel_cum_dist_p2 = pp.calc_wheel_cum_dist(enc_angle)

# weight data
weight_data = (
    api.load(exp_root=exp02_root, datastream=exp02_data_dict.weight,
             spec_ts=spec_ts2, ts_tol=pd.Timedelta('0.001s')))
weight_data = weight_data.iloc[np.where(~np.isnan(weight_data))[0]]

# old weight data
old_weight_data = (
    api.load(exp_root=exp01_root, datastream=exp01_data_dict.weight,
             start_ts=pd.Timestamp('2022-02-08'), end_ts=pd.Timestamp('2022-02-09')))

# all_data = api.load(path=exp02_root, start_ts=start_ts, end_ts=end_ts, data='all',
#                     pos=None)  # `'all'` returns dict of dataframes
