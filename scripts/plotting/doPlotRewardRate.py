import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go

from aeon.query import exp0_api
import aeon.signalProcessing.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/data")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=10.0, type=float)
    parser.add_argument("--time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Reward Rate")
    parser.add_argument("--title_pattern", help="title pattern", default="Session {:d} between {:.02f} and {:0.2f} sec (min_time={:.02f}, max_time={:.02f}, win_length={:.02f})")
    parser.add_argument("--data_filename", help="data filename", default="/ceph/aeon/aeon/preprocessing/experiment0/BAA-1099590/2021-03-25T15-16-18/FrameTop.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/reward_rate_session{:d}_start{:.02f}_duration{:.02f}.{:s}")

    args = parser.parse_args()

    session = args.session
    root = args.root
    t0_relative = args.start_time
    tf_relative = args.start_time+args.duration
    win_length_sec = args.win_length_sec
    time_resolution = args.time_resolution
    xlabel = args.xlabel
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    # <s Get good sessions
    # Get all session metadata from all `SessionData*` csv files (these are
    # 'start' and 'end files) within exp0 root.
    metadata = exp0_api.sessiondata(root)  # pandas df
    # Filter to only animal sessions (the others were test sessions).
    metadata = metadata[metadata.id.str.startswith('BAA')]
    # Drop bad sessions.
    metadata = metadata.drop([metadata.index[16], metadata.index[17], metadata.index[18]])
    # Match each 'start' with its 'end' to get more coherent sessions dataframe.
    metadata = exp0_api.sessionduration(metadata)
    # /s>
    session_start = metadata.index[session]
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)
    pellet_vals = exp0_api.pelletdata(root, start=t0_absolute, end=tf_absolute)
    pellets_times = pellet_vals.query("event == 'TriggerPellet'").index
    pellets_seconds = (pellets_times-session_start).total_seconds()
    time = np.arange(t0_relative, tf_relative, time_resolution)
    pellets_indices = (pellets_seconds/time_resolution).astype(int)
    pellets_samples = np.zeros(time.shape, dtype=np.double)
    pellets_samples[pellets_indices] = 1.0
    win_length_samples = int(win_length_sec/time_resolution)
    reward_rate = aeon.signalProcessing.utils.moving_average(values=pellets_samples, N=win_length_samples)

    session_duration_sec = metadata.iloc[session].duration.total_seconds()
    title = title_pattern.format(session, t0_relative, tf_relative, 0,
                                 session_duration_sec, win_length_sec)
    fig = go.Figure()
    trace = go.Scatter(x=time, y=reward_rate)
    fig.add_trace(trace)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
