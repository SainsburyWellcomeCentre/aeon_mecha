'''
Gets wheel encoder timeseries, pellet triggered timeseries, pellet delivered
timeseries, and mouse position timeseries for select sessions from exp 0.

Raw video file details:
"Frame Top" (acquired at 50 Hz) and "Frame Side" (acquired at 125 Hz) avi
files (they all contain 3 hours of video) in:
/ceph/aeon/test2/data/2021-03-25T15-05-34/
/ceph/aeon/test2/data/2021-03-28T11-52-29/
/ceph/aeon/test2/data/2021-03-31T09-14-43/
'''

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aeon.query import exp0_api

# <s Get good sessions
# Get all session metadata from all `SessionData*` csv files (these are
# 'start' and 'end files) within exp0 root.
root = '/ceph/aeon/test2/data'
metadata = exp0_api.sessiondata(root)  # pandas df
# Filter to only animal sessions (the others were test sessions).
metadata = metadata[metadata.id.str.startswith('BAA')]
# Drop bad sessions.
metadata = metadata.drop([metadata.index[16], metadata.index[17],
                          metadata.index[18]])
# Match each 'start' with its 'end' to get more coherent sessions dataframe.
metadata = exp0_api.sessionduration(metadata)
# /s>

# <s Get data
# The data is organized in a dict, `data`, which can be indexed as
# `data['<session>']['<feature>']` to return a pandas dataframe or numpy
# array (depending on the `<feature>`) for `<feature>` in `<session>`.

data = {}
for i, session in enumerate(metadata.itertuples()):  # per session
    # Get start and end times
    start = session.Index
    end = start + session.duration
    # Get running wheel encoder angle and computed traveled distance vals.
    wheel_encoder_vals = exp0_api.encoderdata(root, start=start, end=end)
    data[start] = {}
    data[start]['wheel_angle'] = \
        wheel_encoder_vals.drop("intensity", axis=1)
    data[start]['wheel_dist'] = \
        exp0_api.distancetravelled(wheel_encoder_vals.angle)
    pellet_vals = exp0_api.pelletdata(root, start=start, end=end)
    data[start]['pellet_triggered'] = \
        pellet_vals.query("event == 'TriggerPellet'").index.to_numpy()
    data[start]['pellet_delivered'] = \
        pellet_vals.query("event == 'PelletDetected'").index.to_numpy()

# Sample position data (the associated video files are the 'FrameTop.avi'
# files in the same dirs).
pos_data = \
    pd.read_csv(Path('/ceph/aeon/aeon/preprocessing/experiment0/BAA'
                     '-1099590/2021-03-25T15-16-18/FrameTop.csv'),
                names=['X', 'Y', 'Orientation', 'MajorAxisLength',
                       'MinoxAxisLength', 'Area'])
pos_data2 = \
    pd.read_csv(Path('/ceph/aeon/aeon/preprocessing/experiment0/BAA'
                     '-1099590/2021-03-28T12-46-02/FrameTop.csv'),
                names=['X', 'Y', 'Orientation', 'MajorAxisLength',
                       'MinoxAxisLength', 'Area'])
