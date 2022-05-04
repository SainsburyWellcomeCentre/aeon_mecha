
import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import pymysql
import plotly.graph_objects as go
import datajoint as dj

sys.path.append("../..")
import aeon.plotting.plot_functions
import aeon.preprocess.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse_label", help="Mouse label", type=str,
                        default="BAA-1099791")
    parser.add_argument("--session_start_time", help="Chunk start",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_secs",
                        help="Start plotting start_secs after the start of the session",
                        type=float, default=0.0)
    parser.add_argument("--duration_secs",
                        help="Plot duration_sec seconds",
                        type=float, default=-1.0)
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")
    parser.add_argument("--db_name", help="DB name", type=str,
                        default="aeon_tracking")
    parser.add_argument("--db_table", help="DB table", type=str,
                        default="_subject_position")
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--ylabel", help="ylabel", default="Proportion of Time")
    parser.add_argument("--ylim", help="ylim", default="[0,1]")
    parser.add_argument("--title_pattern", help="title pattern", default="Mouse {:s}, session_start_time {:s}, start_secs={:.02f}, duration_secs={:02f}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/activityTimes_mouse{:s}_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}.{:s}")

    args = parser.parse_args()

    mouse_label = args.mouse_label
    session_start_time = args.session_start_time
    start_secs = args.start_secs
    duration_secs = args.duration_secs
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    db_name = args.db_name
    db_table = args.db_table
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    ylabel = args.ylabel
    ylim = [float(str) for str in args.ylim[1:-1].split(",")]
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    conn = pymysql.connect(host=tunneled_host,
                           port=db_server_port, user=db_user,
                           passwd=db_user_password)
    cur = conn.cursor()
    sql_stmt = "SELECT timestamps, position_x, position_y FROM {:s}.{:s} WHERE subject=\"{:s}\" AND session_start=\"{:s}\"".format(db_name, db_table, mouse_label, session_start_time)

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
    con.close()
    time_stamps0_sec = time_stamps[0].timestamp()
    time_stamps_secs = np.array([ts.timestamp()-time_stamps0_sec
                                 for ts in time_stamps])
    if duration_secs<0:
        max_secs = time_stamps_secs.max()
    else:
        max_secs = start_secs + duration_secs
    indices_keep = np.where(
        np.logical_and(start_secs<=time_stamps_secs,
                       time_stamps_secs<max_secs))[0]
    time_stamps_secs = time_stamps_secs[indices_keep]
    time_stamps = time_stamps[indices_keep]
    x = x[indices_keep]
    y = y[indices_keep]

    patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                       columns=["lower_x", "higher_x",
                                                "lower_y", "higher_y"])
    nest_coordinates = pd.DataFrame(data=nest_coordinates_matrix,
                                    columns=["lower_x", "higher_x",
                                             "lower_y", "higher_y"])
    positions_labels = aeon.preprocess.utils.get_positions_labels(
        x=x, y=y,
        patches_coordinates=patches_coordinates,
        nest_coordinates=nest_coordinates)
    title = title_pattern.format(mouse_label, session_start_time, start_secs, duration_secs)
    fig = go.Figure()
    cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
        positions_labels=positions_labels,
    )
    fig.add_trace(cumTimePerActivity_trace)
    fig.update_layout(title=title, yaxis_title=ylabel)
    fig.update_yaxes(range=ylim)
    fig.write_image(fig_filename_pattern.format(mouse_label, session_start_time, start_secs, duration_secs, "png"))
    fig.write_html(fig_filename_pattern.format(mouse_label, session_start_time, start_secs, duration_secs, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
