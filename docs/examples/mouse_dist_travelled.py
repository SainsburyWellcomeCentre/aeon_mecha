# Look at position traveled values

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aeon.query.api as aeon

# Get metadata
root = '/ceph/aeon/test2/experiment0.1'
metadata = aeon.sessiondata(root)
annotations = aeon.annotations(root)
metadata = aeon.sessionduration(metadata)
metadata = metadata[metadata.id.str.startswith('BAA')]
metadata = metadata[metadata.start >= pd.Timestamp('20210614')]
metadata = metadata[(metadata.duration > pd.Timedelta('1h'))]

# Look at a particular session
session = metadata[metadata['start'] == '2021-06-28 12:43:44.099679947']
# (on DJ 'SessionSummary' table, this session has a value of 2993.43 m traveled)

# Get position values
PIXEL_SCALE = 0.00192      # 1 px = 1.92 mm
pos = aeon.positiondata(root, start=session.loc[44, 'start'],
                        end=session.loc[44, 'end'])
valid_pos = (pos.area > 0) & (pos.area < 1000)
pos = pos[valid_pos]
x = pos.x.to_numpy()
y = pos.y.to_numpy()
x = x[~np.isnan(x)] * PIXEL_SCALE
y = y[~np.isnan(y)] * PIXEL_SCALE

# Assume max speed of mouse is 20 km/h = 20 * 1000 / 3600 = 5.55 m/s
max_pos_diff_mouse = 5.55 / 50  # 50 hz sampling frequency
x_diff = np.abs(np.diff(x))
y_diff = np.abs(np.diff(y))
# Filter out large jumps in position tracking data; replace with
# `max_pos_diff_mouse`
x_diff[x_diff > max_pos_diff_mouse] = 0 # max_pos_diff_mouse
y_diff[y_diff > max_pos_diff_mouse] = 0 # max_pos_diff_mouse
total_dist = np.sum(np.sqrt(np.square(x_diff) + np.square(y_diff)))
