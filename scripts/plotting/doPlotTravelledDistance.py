import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go

from aeon.query import exp0_api
import aeon.plotting.plot_functions as pf


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/data")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--pellet_event_name", help="Pallete event name to display", default="TriggerPellet")
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Travelled Distance (cm)")
    parser.add_argument("--pellet_line_color", help="pellet line color", default="red")
    parser.add_argument("--pellet_line_style", help="pellet line style", default="solid")
    parser.add_argument("--title_pattern", help="title pattern", default="Plotting session {:d} from {:.02f} to {:0.2f} sec\n (min_time={:.02f}, max_time={:.02f})")
    parser.add_argument("--data_filename", help="data filename", default="/ceph/aeon/aeon/preprocessing/experiment0/BAA-1099590/2021-03-25T15-16-18/FrameTop.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_session{:d}_start{:.02f}_duration{:.02f}.{:s}")

    args = parser.parse_args()

    session = args.session
    root = args.root
    t0_relative = args.start_time
    tf_relative = args.start_time+args.duration
    pellet_event_name = args.pellet_event_name
    xlabel = args.xlabel
    ylabel = args.ylabel
    pellet_line_color = args.pellet_line_color
    pellet_line_style = args.pellet_line_style
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
    wheel_encoder_vals = exp0_api.encoderdata(root, start=t0_absolute, end=tf_absolute)
    travelled_distance = exp0_api.distancetravelled(wheel_encoder_vals.angle)
    travelled_seconds = (travelled_distance.index-session_start).total_seconds()
    pellet_vals = exp0_api.pelletdata(root, start=t0_absolute, end=tf_absolute)
    triggered_pellets_times = pellet_vals.query("event == '{:s}'".format(pellet_event_name)).index
    triggered_pellets_seconds = (triggered_pellets_times-session_start).total_seconds()

    duration_sec = metadata.iloc[session].duration.total_seconds()
    title = title_pattern.format(session, t0_relative, tf_relative, 0, duration_sec)
    fig = go.Figure()
    trace = pf.get_travelling_distance_trace(travelled_seconds=travelled_seconds,
                                             travelled_distance=travelled_distance)
    fig.add_trace(trace)
    for triggered_pellet_second in triggered_pellets_seconds:
        fig.add_vline(x=triggered_pellet_second, line_color=pellet_line_color,
                     line_dash=pellet_line_style)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
