
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
    parser.add_argument("--time_slice_start", help="Time slice start",
                        type=str, default="2021-10-01 13:03:45.835619")
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
    parser.add_argument("--title_pattern", help="title pattern", default="Mouse {:s}, time_slice_start {:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filenampattern", default="../../figures/activity_times_mouse{:s}_timeSliceStart{:s}.{:s}")

    args = parser.parse_args()

    mouse_label = args.mouse_label
    time_slice_start = args.time_slice_start
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
    sql_stmt = "SELECT position_x, position_y FROM {:s}.{:s} WHERE subject=\"{:s}\" AND time_slice_start>=\"{:s}\"".format(db_name, db_table, mouse_label, time_slice_start)

    cur.execute(sql_stmt)
    row = cur.fetchone()
    position_x_blob = row[0]
    position_y_blob = row[1]
    x = dj.blob.Blob().unpack(blob=position_x_blob)
    y = dj.blob.Blob().unpack(blob=position_y_blob)

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
    title = title_pattern.format(mouse_label, time_slice_start)
    fig = go.Figure()
    cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
        positions_labels=positions_labels,
    )
    fig.add_trace(cumTimePerActivity_trace)
    fig.update_layout(title=title, yaxis_title=ylabel)
    fig.update_yaxes(range=ylim)
    fig.write_image(fig_filename_pattern.format(mouse_label, time_slice_start, "png"))
    fig.write_html(fig_filename_pattern.format(mouse_label, time_slice_start, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
