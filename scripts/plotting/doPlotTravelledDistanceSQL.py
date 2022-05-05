import sys
import pdb
import time
import numpy as np
import pandas as pd
import argparse
import MySQLdb
import plotly.graph_objects as go
import plotly.subplots
import datajoint as dj

sys.path.append("../..")
import aeon.storage.sqlStorageMgr as sm
import aeon.preprocess.api as api
import aeon.plotting.plot_functions


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_secs", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration_secs", help="Duration (sec)", default=-1, type=float)
    parser.add_argument("--plot_sample_rate", help="Plot sample rate", default=10.0, type=float)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="COM4,COM7")
    parser.add_argument("--pellet_event_label", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Travelled Distance (cm)")
    parser.add_argument("--pellet_color", help="pellet color", default="red")
    parser.add_argument("--title_pattern", help="title pattern", default="session_start_time={:s}, start_secs={:.02f}, duration_secs={:02f}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}_srate{:.02f}.{:s}")

    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_secs = args.start_secs
    duration_secs = args.duration_secs
    plot_sample_rate = args.plot_sample_rate
    patches_to_plot = args.patchesToPlot.split(",")
    pellet_event_label = args.pellet_event_label
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    xlabel = args.xlabel
    ylabel = args.ylabel
    pellet_color = args.pellet_color
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    delta = pd.Timedelta(1, "hour")
    session_start_time = pd.Timestamp(session_start_time_str)
    session_start_time_minusDelta = session_start_time - delta
    session_start_time_minusDelta_str = session_start_time_minusDelta.strftime("%Y-%m-%d %H:%M:%S")

    storageMgr = sm.SQLStorageMgr(host=tunneled_host, port=db_server_port,
                                  user=db_user, passwd=db_user_password)
    session_end_time = storageMgr.getSessionEndTime(session_start_time_str=
                                                    session_start_time_str)
    session_end_time_plusDelta = session_end_time + delta

    session_end_time_str = session_end_time.strftime("%Y-%m-%d %H:%M:%S")
    session_end_time_plusDelta_str = session_end_time_plusDelta.strftime("%Y-%m-%d %H:%M:%S")
    travelled_distance = {}
    travelled_seconds = {}
    pellets_seconds = {}
    max_travelled_distance = -np.inf
    for patch_to_plot in patches_to_plot:
        angles = storageMgr.getWheelAngles(
            start_time_str=session_start_time_minusDelta_str,
            end_time_str=session_end_time_plusDelta_str,
            patch_label=patch_to_plot)
        timestamps_secs = angles.index
        timestamps_secs_rel = (timestamps_secs - session_start_time).total_seconds()
        if duration_secs < 0:
            max_secs = timestamps_secs_rel.max()
        else:
            max_secs = start_secs + duration_secs
        indices_keep = np.where(np.logical_and(start_secs<=timestamps_secs_rel, timestamps_secs_rel<max_secs))[0]
        angles = angles[indices_keep]
        travelled_distance[patch_to_plot] = api.distancetravelled(angles)
        travelled_seconds[patch_to_plot] = (travelled_distance[patch_to_plot].index-session_start_time).total_seconds()
        max_travelled_distance = max([travelled_distance[key].max() for key in travelled_distance.keys()])
        pellets_times = storageMgr.getFoodPatchEventTimes(
            start_time_str=session_start_time_str,
            end_time_str=session_end_time_str,
            event_label=pellet_event_label,
            patch_label=patch_to_plot)
        pellets_secs_rel = (pellets_times-session_start_time).total_seconds()
        indices_keep = np.where(np.logical_and(start_secs<=pellets_secs_rel, pellets_secs_rel<max_secs))[0]
        pellets_seconds[patch_to_plot] = pellets_secs_rel[indices_keep]

    title = title_pattern.format(session_start_time_str, start_secs, duration_secs, plot_sample_rate)
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot),
                                        subplot_titles=(patches_to_plot))
    for i, patch_to_plot in enumerate(patches_to_plot):
        trace = aeon.plotting.plot_functions.get_travelled_distance_trace(
            travelled_seconds=travelled_seconds[patch_to_plot],
            travelled_distance=travelled_distance[patch_to_plot],
            sample_rate=plot_sample_rate,
            showlegend=False)
        fig.add_trace(trace, row=1, col=i+1)
        trace = aeon.plotting.plot_functions.get_pellets_trace(pellets_seconds=pellets_seconds[patch_to_plot], marker_color=pellet_color)
        fig.add_trace(trace, row=1, col=i+1)
        if i==0:
            fig.update_yaxes(title_text=ylabel, range=(0, max_travelled_distance), row=1, col=i+1)
        else:
            fig.update_yaxes(range=(0, max_travelled_distance), row=1, col=i+1)
        fig.update_xaxes(title_text=xlabel, row=1, col=i+1)
    fig.update_layout(title=title)
    fig.write_image(fig_filename_pattern.format(session_start_time_str, start_secs, duration_secs, plot_sample_rate, "png"))
    fig.write_html(fig_filename_pattern.format(session_start_time_str, start_secs, duration_secs, plot_sample_rate, "html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
