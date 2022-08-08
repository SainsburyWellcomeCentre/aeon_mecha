import sys
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go
import plotly.subplots

sys.path.append("../..")
import aeon.storage.sqlStorageMgr as sqlSM
import aeon.storage.filesStorageMgr as filesSM
import aeon.plotting.plot_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_secs", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration_secs", help="Duration (sec)", default=-1, type=float)
    parser.add_argument("--patchesToPlotSQL", help="Names of patches to plot", default="COM4,COM7")
    parser.add_argument("--patchesToPlotFiles", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--storageMgr_type",
                        help="Type of storage manager (SQL | Files)",
                        type=str, default="SQL")
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")
    parser.add_argument("--root", help="Root path for data access",
                        default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--pellet_event_label", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--ma_win_length", help="Reward rate moving average window length", default="60s")
    parser.add_argument("--time_resolution", help="Time resolution to compute the reward rate moving average (sec)", default=0.1, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Reward Rate")
    parser.add_argument("--trace_color", help="trace color", default="blue")
    parser.add_argument("--pellet_color", help="pellet color", default="red")
    parser.add_argument("--title_pattern", help="title pattern",
                        default="session_start_time={:s}, start_secs={:.02f}, duration_secs={:02f}, storage {:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern",
                        default="../../figures/reward_rate_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}_storage{:s}.{:s}")

    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_secs = args.start_secs
    duration_secs = args.duration_secs
    storageMgr_type = args.storageMgr_type
    tunneled_host=args.tunneled_host
    patches_to_plot_sql = args.patchesToPlotSQL.split(",")
    patches_to_plot_files = args.patchesToPlotFiles.split(",")
    pellet_event_label = args.pellet_event_label
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    root = args.root
    ma_win_length = args.ma_win_length
    time_resolution = args.time_resolution
    xlabel = args.xlabel
    ylabel = args.ylabel
    trace_color = args.trace_color
    pellet_color = args.pellet_color
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    t0_relative = start_secs
    session_start_time = pd.Timestamp(session_start_time_str)

    tf_relative = t0_relative + duration_secs

    if storageMgr_type == "SQL":
        storageMgr = sqlSM.SQLStorageMgr(host=tunneled_host,
                                         port=db_server_port,
                                         user=db_user,
                                         passwd=db_user_password)
        patches_to_plot = patches_to_plot_sql
    elif storageMgr_type == "Files":
        storageMgr = filesSM.FilesStorageMgr(root=root)
        patches_to_plot = patches_to_plot_files
    else:
        raise ValueError(f"Invalid storageMgr_type={storageMgr_type}")

    session_end_time = storageMgr.getSessionEndTime(session_start_time_str=
                                                    session_start_time_str)
    session_end_time_str = session_end_time.strftime("%Y-%m-%d %H:%M:%S")

    if duration_secs < 0:
        duration_secs = (session_end_time - session_start_time).total_seconds()

    pellets_seconds = {}
    reward_rate = {}
    max_reward_rate = -np.inf
    for patch_to_plot in patches_to_plot:
        pellets_times = storageMgr.getFoodPatchEventTimes(
            start_time_str=session_start_time_str,
            end_time_str=session_end_time_str,
            event_label=pellet_event_label,
            patch_label=patch_to_plot)
        pellets_seconds[patch_to_plot] = (pellets_times-session_start_time).total_seconds()
        import pdb; pdb.set_trace()
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

    title = title_pattern.format(session_start_time_str, t0_relative,
                                 tf_relative, storageMgr_type)
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
    fig.write_image(fig_filename_pattern.format(session_start_time_str,
                                                start_secs, duration_secs,
                                                storageMgr_type, "png"))
    fig.write_html(fig_filename_pattern.format(session_start_time_str,
                                               start_secs, duration_secs,
                                               storageMgr_type, "html"))

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
