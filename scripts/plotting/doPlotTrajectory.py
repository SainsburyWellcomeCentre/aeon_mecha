import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go

import aeon.preprocess.api as api
import aeon.preprocess.utils
import aeon.plotting.plot_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (max={:.02f} sec)")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectory_session{:d}_start{:.02f}_end{:.02f}.{:s}")


    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    tf_relative = args.start_time+args.duration
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    xlabel = args.xlabel
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    session_start = metadata.index[session]
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)

    position = api.positiondata(root, start=t0_absolute, end=tf_absolute)          # get position data between start and end
    # position = position[position.area < 2000]                         # filter for objects of the correct size

    patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                       columns=["lower_x", "higher_x",
                                                "lower_y", "higher_y"])
    nest_coordinates = pd.DataFrame(data=nest_coordinates_matrix,
                                    columns=["lower_x", "higher_x",
                                             "lower_y", "higher_y"])
    x = position["x"].to_numpy()
    y = position["y"].to_numpy()
    duration_sec = metadata.iloc[session].duration.total_seconds()
    title = title_pattern.format(str(metadata.index[session]), t0_relative, tf_relative, duration_sec)
    time_stamps = (position.index-session_start).total_seconds().to_numpy()
    fig = go.Figure()
    trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps)
    fig.add_trace(trajectory_trace)
    patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
    for patch_trace in patches_traces:
        fig.add_trace(patch_trace)
    nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
    fig.add_trace(nest_trace)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
