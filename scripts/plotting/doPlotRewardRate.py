import sys
import pdb
import numpy as np
import scipy.interpolate
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go
import plotly.subplots

sys.path.append("../..")
import aeon.preprocess.api as api
import aeon.preprocess.utils
import aeon.signalProcessing.utils
import aeon.plotting.plot_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=-1.0, type=float)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--ma_win_length", help="Reward rate moving average window length", default="60s")
    parser.add_argument("--time_resolution", help="Time resolution to compute the reward rate moving average (sec)", default=0.1, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Reward Rate")
    parser.add_argument("--trace_color", help="trace color", default="blue")
    parser.add_argument("--pellet_color", help="pellet color", default="red")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (min={:.02f}, max={:.02f})")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/reward_rate_session{:d}_start{:.02f}_duration{:.02f}.{:s}")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    if args.duration > 0:
        tf_relative = args.start_time+args.duration
    else:
        tf_relative = None
    patches_to_plot = args.patchesToPlot.split(",")
    pellet_event_name = args.pellet_event_name
    ma_win_length = args.ma_win_length
    time_resolution = args.time_resolution
    xlabel = args.xlabel
    ylabel = args.ylabel
    trace_color = args.trace_color
    pellet_color = args.pellet_color
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    session_start = metadata.iloc[session].start
    duration_sec = metadata.iloc[session].duration.total_seconds()
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    if tf_relative is None:
        tf_relative = args.start_time+duration_sec
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)

    pellets_seconds = {}
    reward_rate = {}
    max_reward_rate = -np.inf
    for patch_to_plot in patches_to_plot:
        pellet_vals = api.pelletdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
        pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
        pellets_seconds[patch_to_plot] = (pellets_times-session_start).total_seconds()
        pellets_events = pd.DataFrame(index=pd.TimedeltaIndex(data=pellets_seconds[patch_to_plot], unit="s"), data=np.ones(len(pellets_seconds[patch_to_plot])), columns=['count'])
        boundary_df = pd.DataFrame(index=pd.TimedeltaIndex(data=[t0_relative, tf_relative], unit="s"), data=dict(count=[0.0, 0.0]), columns=['count'])
        pellets_events = pellets_events.append(other=boundary_df)
        pellets_events.sort_index(inplace=True)
        pellets_events_binned = pellets_events.resample("{:f}S".format(time_resolution)).sum()
        pellets_events_ma = pellets_events_binned.rolling(window=ma_win_length, center=True).mean()
        reward_rate[patch_to_plot] = pellets_events_ma
        patch_max_reward_rate = reward_rate[patch_to_plot].max().item()
        if patch_max_reward_rate>max_reward_rate:
            max_reward_rate = patch_max_reward_rate

    title = title_pattern.format(str(metadata.index[session]), t0_relative, tf_relative, 0, duration_sec)
    fig = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot), subplot_titles=(patches_to_plot))
    for i, patch_to_plot in enumerate(patches_to_plot):
        trace = go.Scatter(x=reward_rate[patch_to_plot].index.total_seconds(), y=reward_rate[patch_to_plot].iloc[:,0], line=dict(color=trace_color), showlegend=False)
        fig.add_trace(trace, row=1, col=i+1)
        trace = aeon.plotting.plot_functions.get_pellets_trace(pellets_seconds=pellets_seconds[patch_to_plot], marker_color=pellet_color)
        fig.add_trace(trace, row=1, col=i+1)
        if i == 0:
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
