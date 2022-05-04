import sys
import pdb
import time
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import datetime
import MySQLdb
import plotly.graph_objects as go
import plotly.subplots
import datajoint as dj

sys.path.append("../..")
import aeon.preprocess.api as api
import aeon.plotting.plot_functions


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_start_time", help="session start time",
                        type=str, default="2021-10-01 13:03:45.835619")
    parser.add_argument("--start_secs", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration_secs", help="Duration (sec)", default=-1, type=float)
    parser.add_argument("--plot_sample_rate", help="Plot sample rate", default=10.0, type=float)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot",
                        default="COM4,COM7")
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")
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
    parser.add_argument("--title_pattern", help="title pattern", default="session_start_time {:s}, start_secs={:.02f}, duration_secs={:02f}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_sessionStart{:s}_startSecs_{:02f}_durationSecs_{:02f}_srate{:.02f}.{:s}")

    args = parser.parse_args()

    session_start_time_str = args.session_start_time
    start_secs = args.start_secs
    duration_secs = args.duration_secs
    plot_sample_rate = args.plot_sample_rate
    patches_to_plot = args.patchesToPlot.split(",")
    pellet_event_name = args.pellet_event_name
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password
    xlabel = args.xlabel
    ylabel = args.ylabel
    pellet_color = args.pellet_color
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    delta = datetime.timedelta(hours=1)
    session_start_time = pd.Timestamp(session_start_time_str)
    session_start_time_minusDelta = session_start_time - delta
    session_start_time_minusDelta_str = session_start_time_minusDelta.strftime("%Y-%m-%d %H:%M:%S")
    # session_start_time = datetime.datetime.fromisoformat(session_start_time_str)

    conn = MySQLdb.connect(host=tunneled_host,
                           port=db_server_port, user=db_user,
                           passwd=db_user_password)
    # mouse_label = BAA-1099791
    # mysql> select in_arena_end from aeon_analysis.__in_arena_end where in_arena_start="2021-10-01 13:03:45.835619";
    sql_stmt = "SELECT in_arena_end FROM aeon_analysis.__in_arena_end WHERE in_arena_start=\"{:s}\"".format(session_start_time_str)
    cur = conn.cursor()
    cur.execute(sql_stmt)
    # session_end_time="2021-10-01 17:20:20.224289"
    session_end_time = cur.fetchone()[0]
    session_end_time_plusDelta = session_end_time + delta
    cur.close()

    session_end_time_str = session_end_time.strftime("%Y-%m-%d %H:%M:%S")
    session_end_time_plusDelta_str = session_end_time_plusDelta.strftime("%Y-%m-%d %H:%M:%S")
    travelled_distance = {}
    travelled_seconds = {}
    pellets_seconds = {}
    max_travelled_distance = -np.inf
    for patch_to_plot in patches_to_plot:
        # mysql> select timestamps, angle from aeon_acquisition._food_patch_wheel where chunk_start between session_start and session_end;
        # sql_stmt = "SELECT timestamps, angle FROM aeon_acquisition._food_patch_wheel WHERE chunk_start BETWEEN \"{:s}\" AND \"{:s}\" AND food_patch_serial_number=\"{:s}\"".format(session_start_time_str, session_end_time_str, patch_to_plot)
        sql_stmt = "SELECT timestamps, angle FROM aeon_acquisition._food_patch_wheel WHERE chunk_start BETWEEN \"{:s}\" AND \"{:s}\" AND food_patch_serial_number=\"{:s}\"".format(session_start_time_minusDelta_str, session_end_time_plusDelta_str, patch_to_plot)
        start = time.time()
        cur = conn.cursor()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"SQL query for timestaps and angle took {end - start} seconds")
        records = cur.fetchall()
        nChuncks = len(records)
        all_angles_series = None
        # begin debug code
        # for i in range(2):
        #     row = records[i]
        # end debug code
        for i, row in enumerate(records):
            timestamps_blob = row[0]
            angle_blob = row[1]
            # wheel_encoder_vals = api.encoderdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
            start = time.time()
            timestamps_secs = pd.DatetimeIndex(dj.blob.Blob().unpack(blob=timestamps_blob))
            end = time.time()
            print(f"Unpacking timestamps blog took {end - start} seconds ({i+1}/{nChuncks})")
            start = time.time()
            angle = dj.blob.Blob().unpack(blob=angle_blob)
            end = time.time()
            print(f"Unpacking angles blob took {end - start} seconds ({i+1}/{nChuncks})")
            angle_series = pd.Series(angle, index=timestamps_secs)
            if all_angles_series is None:
                all_angles_series = angle_series
            else:
                all_angles_series = pd.concat((all_angles_series, angle_series))
            timestamps_secs = all_angles_series.index
            timestamps_secs_rel = (timestamps_secs - session_start_time).total_seconds()
            if duration_secs < 0:
                max_secs = timestamps_secs_rel.max()
            else:
                max_secs = start_secs + duration_secs
            indices_keep = np.where(np.logical_and(start_secs<=timestamps_secs_rel, timestamps_secs_rel<max_secs))[0]
            all_angles_series = all_angles_series[indices_keep]
            travelled_distance[patch_to_plot] = api.distancetravelled(all_angles_series)
            travelled_seconds[patch_to_plot] = (travelled_distance[patch_to_plot].index-session_start_time).total_seconds()
        cur.close()
        max_travelled_distance = max([travelled_distance[key].max() for key in travelled_distance.keys()])
        # SELECT event_time FROM aeon_acquisition._food_patch_event 
        # INNER JOIN aeon_acquisition.`#event_type` ON 
        #  aeon_acquisition._food_patch_event.event_code=aeon_acquisition.`#event_type`.event_code 
        # WHERE aeon_acquisition.`#event_type`.event_type="TriggerPellet" AND 
        #       food_patch_serial_number="COM4" AND 
        #       event_time BETWEEN "2021-10-01 13:03:45.835619" AND "2021-10-01 17:20:20.224289";
        sql_stmt = "SELECT event_time FROM aeon_acquisition._food_patch_event " \
                   "INNER JOIN aeon_acquisition.`#event_type` ON " \
                     "aeon_acquisition._food_patch_event.event_code=aeon_acquisition.`#event_type`.event_code " \
                   "WHERE aeon_acquisition.`#event_type`.event_type=\"TriggerPellet\" AND "\
                         "food_patch_serial_number=\"{:s}\" AND " \
                         "event_time BETWEEN \"{:s}\" AND \"{:s}\"".format(
                             patch_to_plot, session_start_time_str,
                             session_end_time_str)
        cur = conn.cursor()
        cur.execute(sql_stmt)
        records = cur.fetchall()
        pellets_times = pd.DatetimeIndex([pd.Timestamp(row[0]) for row in records])
        cur.close()

        # pellet_vals = api.pelletdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
        # pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
        pellets_secs_rel = (pellets_times-session_start_time).total_seconds()
        indices_keep = np.where(np.logical_and(start_secs<=pellets_secs_rel, pellets_secs_rel<max_secs))[0]
        pellets_seconds[patch_to_plot] = pellets_secs_rel[indices_keep]

    title = title_pattern.format(session_start_time_str, start_secs, duration_secs, plot_sample_rate)
    fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot),
                                        subplot_titles=(patches_to_plot))
    for i, patch_to_plot in enumerate(patches_to_plot):
        trace = aeon.plotting.plot_functions.get_travelled_distance_trace(travelled_seconds=travelled_seconds[patch_to_plot],
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
