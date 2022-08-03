import sys
import numpy as np
import pandas as pd
import argparse
import plotly.graph_objects as go

sys.path.append("../..")
import aeon.storage.sqlStorageMgr as sqlSM
import aeon.storage.filesStorageMgr as filesSM
import aeon.plotting.plot_functions


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_offset_secs",
                        help="Start plotting start_offset_secs after the start of the session",
                        type=float, default=0.0)
    parser.add_argument("--duration_secs",
                        help="Plot duration_sec seconds",
                        type=float, default=-1.0)
    parser.add_argument("--sample_rate", help="Plot sample rate", type=float,
                        default=20.0)
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
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")
    parser.add_argument("--title_pattern", help="title pattern", default="session_start_time={:s}, start_offset_secs={:.02f}, duration_secs={:02f}, sample_rate={:.02f} Hz, storage={:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectory_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}_srate{:.02f}_storage{:s}.{:s}")

    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_offset_secs = args.start_offset_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    storageMgr_type = args.storageMgr_type
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    root = args.root
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    xlabel = args.xlabel
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                       columns=["lower_x", "higher_x",
                                                "lower_y", "higher_y"])
    nest_coordinates = pd.DataFrame(data=nest_coordinates_matrix,
                                    columns=["lower_x", "higher_x",
                                             "lower_y", "higher_y"])

    if storageMgr_type == "SQL":
        storageMgr = sqlSM.SQLStorageMgr(host=tunneled_host,
                                         port=db_server_port,
                                         user=db_user,
                                         passwd=db_user_password)
    elif storageMgr_type == "Files":
        storageMgr = filesSM.FilesStorageMgr(root=root)
    else:
        raise ValueError(f"Invalid storageMgr_type={storageMgr_type}")

    positions = storageMgr.getSessionPositions(
        session_start_time_str=session_start_time_str,
        start_offset_secs=start_offset_secs,
        duration_secs=duration_secs)
    time_stamps = positions.index
    x = positions["x"].to_numpy()
    y = positions["y"].to_numpy()
    time_stamps0_sec = time_stamps[0]
    time_stamps_secs = np.array([ts-time_stamps0_sec
                                 for ts in time_stamps])
#     if duration_secs<0:
#         max_secs = time_stamps_secs.max()
#     else:
#         max_secs = start_offset_secs+duration_secs
#     indices_keep = np.where(
#         np.logical_and(start_offset_secs<=time_stamps_secs,
#                        time_stamps_secs<max_secs))[0]
#     time_stamps_secs = time_stamps_secs[indices_keep]
#     time_stamps = time_stamps[indices_keep]
#     x = x[indices_keep]
#     y = y[indices_keep]

    title = title_pattern.format(session_start_time_str, start_offset_secs, duration_secs, sample_rate, storageMgr_type)
    storageMgr_type = args.storageMgr_type
    fig = go.Figure()
    trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps_secs, sample_rate=sample_rate)
    fig.add_trace(trajectory_trace)
    patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
    for patch_trace in patches_traces:
        fig.add_trace(patch_trace)
    nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
    fig.add_trace(nest_trace)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image(fig_filename_pattern.format(session_start_time_str,
                                                start_offset_secs, duration_secs,
                                                sample_rate, storageMgr_type, "png"))
    fig.write_html(fig_filename_pattern.format(session_start_time_str,
                                               start_offset_secs, duration_secs,
                                               sample_rate, storageMgr_type, "html"))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
