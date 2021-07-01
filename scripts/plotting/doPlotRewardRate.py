import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go
import plotly.subplots

import aeon.preprocess.api as api
import aeon.preprocess.utils
import aeon.signalProcessing.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=10.0, type=float)
    parser.add_argument("--time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Reward Rate")
    parser.add_argument("--pellet_line_color", help="pellet line color", default="red")
    parser.add_argument("--pellet_line_style", help="pellet line style", default="solid")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (min={:.02f}, max={:.02f})")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/reward_rate_session{:d}_start{:.02f}_duration{:.02f}.{:s}")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    tf_relative = args.start_time+args.duration
    patches_to_plot = args.patchesToPlot.split(",")
    pellet_event_name = args.pellet_event_name
    win_length_sec = args.win_length_sec
    time_resolution = args.time_resolution
    xlabel = args.xlabel
    ylabel = args.ylabel
    pellet_line_color = args.pellet_line_color
    pellet_line_style = args.pellet_line_style
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    session_start = metadata.index[session]
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)
    time = np.arange(t0_relative, tf_relative, time_resolution)

    pellets_seconds = {}
    reward_rate = {}
    max_reward_rate = -np.inf
    for patch_to_plot in patches_to_plot:
        wheel_encoder_vals = api.encoderdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
        pellet_vals = api.pelletdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
        pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
        pellets_seconds[patch_to_plot] = (pellets_times-session_start).total_seconds()
        pellets_indices = ((pellets_seconds[patch_to_plot]-t0_relative)/time_resolution).astype(int)
        pellets_samples = np.zeros(len(time), dtype=np.double)
        pellets_samples[pellets_indices] = 1.0
        win_length_samples = int(win_length_sec/time_resolution)
        reward_rate[patch_to_plot] = aeon.signalProcessing.utils.moving_average(values=pellets_samples, N=win_length_samples)
        patch_max_reward_rate = max(reward_rate[patch_to_plot])
        if patch_max_reward_rate>max_reward_rate:
            max_reward_rate = patch_max_reward_rate

    duration_sec = metadata.iloc[session].duration.total_seconds()
    title = title_pattern.format(str(metadata.index[session]), t0_relative, tf_relative, 0, duration_sec)
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot),
                                        subplot_titles=(patches_to_plot))
    for i, patch_to_plot in enumerate(patches_to_plot):
        trace = go.Scatter(x=time+win_length_sec/2.0,
                           y=reward_rate[patch_to_plot],
                           showlegend=False)
        fig.add_trace(trace, row=1, col=i+1)
        for pellet_second in pellets_seconds[patch_to_plot]:
            fig.add_vline(x=pellet_second, line_color=pellet_line_color,
                          line_dash=pellet_line_style, row=1, col=i+1)
        if i==0:
            fig.update_yaxes(title_text=ylabel, range=(0, max_reward_rate), row=1, col=i+1)
        else:
            fig.update_yaxes(range=(0, max_reward_rate), row=1, col=i+1)
        fig.update_xaxes(title_text=xlabel, row=1, col=i+1)
    fig.update_layout(title=title)
    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
