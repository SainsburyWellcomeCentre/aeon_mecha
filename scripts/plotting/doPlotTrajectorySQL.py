import sys
import numpy as np
import pandas as pd
import argparse
import MySQLdb
import plotly.graph_objects as go
import datajoint as dj

sys.path.append("../..")
import aeon.plotting.plot_functions


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_secs",
                        help="Start plotting start_secs after the start of the session",
                        type=float, default=0.0)
    parser.add_argument("--duration_secs",
                        help="Plot duration_sec seconds",
                        type=float, default=-1.0)
    parser.add_argument("--sample_rate", help="Plot sample rate", type=float,
                        default=20.0)
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")
    parser.add_argument("--title_pattern", help="title pattern", default="session_start_time={:s}, start_secs={:.02f}, duration_secs={:02f}, sample_rate={:.02f} Hz")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectory_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}_srate{:.02f}.{:s}")

    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_secs = args.start_secs
    duration_secs = args.duration_secs
    sample_rate = args.sample_rate
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
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

    conn = MySQLdb.connect(host=tunneled_host,
                           port=db_server_port, user=db_user,
                           passwd=db_user_password)
    cur = conn.cursor()
    sql_stmt = "SELECT timestamps, position_x, position_y FROM aeon_tracking._subject_position WHERE session_start=\"{:s}\"".format(session_start_time_str)
    cur.execute(sql_stmt)
    records = cur.fetchall()
    time_stamps = x = y = None
    time_stamps = np.array([], dtype=object)
    x = np.array([], dtype=np.double)
    y = np.array([], dtype=np.double)
    for row in records:
        timestamps_blob = row[0]
        position_x_blob = row[1]
        position_y_blob = row[2]
        time_stamps = np.append(time_stamps,
                                dj.blob.Blob().unpack(blob=timestamps_blob))
        x = np.append(x, dj.blob.Blob().unpack(blob=position_x_blob))
        y = np.append(y, dj.blob.Blob().unpack(blob=position_y_blob))
    cur.close()
    conn.close()
    time_stamps0_sec = time_stamps[0].timestamp()
    time_stamps_secs = np.array([ts.timestamp()-time_stamps0_sec
                                 for ts in time_stamps])
    if duration_secs<0:
        max_secs = time_stamps_secs.max()
    else:
        max_secs = start_secs+duration_secs
    indices_keep = np.where(
        np.logical_and(start_secs<=time_stamps_secs,
                       time_stamps_secs<max_secs))[0]
    time_stamps_secs = time_stamps_secs[indices_keep]
    time_stamps = time_stamps[indices_keep]
    x = x[indices_keep]
    y = y[indices_keep]

    title = title_pattern.format(session_start_time_str, start_secs, duration_secs, sample_rate)
    fig = go.Figure()
    trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps_secs, sample_rate=sample_rate)
    fig.add_trace(trajectory_trace)
    patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
    for patch_trace in patches_traces:
        fig.add_trace(patch_trace)
    nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
    fig.add_trace(nest_trace)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image(fig_filename_pattern.format(session_start_time_str, start_secs, duration_secs, sample_rate, "png"))
    fig.write_html(fig_filename_pattern.format(session_start_time_str, start_secs, duration_secs, sample_rate, "html"))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
