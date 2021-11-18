
import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go

sys.path.append("../..")
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
    parser.add_argument("--ylabel", help="ylabel", default="Proportion of Time")
    parser.add_argument("--ylim", help="ylim", default="[0,1]")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (max={:.02f} sec)")
    parser.add_argument("--fig_filename_pattern", help="figure filenampattern", default="../../figures/activity_times_session{:d}_start{:.02f}_end{:.02f}.{:s}")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    tf_relative = args.start_time+args.duration
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    ylabel = args.ylabel
    ylim = [float(str) for str in args.ylim[1:-1].split(",")]
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    session_start = metadata.iloc[session].start
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
    x = position["x"]
    y = position["y"]

    duration_sec = metadata.iloc[session].duration.total_seconds()
    positions_labels = aeon.preprocess.utils.get_positions_labels(
        x=x, y=y,
        patches_coordinates=patches_coordinates,
        nest_coordinates=nest_coordinates)
    title = title_pattern.format(str(metadata.index[session]), t0_relative, tf_relative, duration_sec)
    fig = go.Figure()
    cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
        positions_labels=positions_labels,
    )
    fig.add_trace(cumTimePerActivity_trace)
    fig.update_layout(title=title, yaxis_title=ylabel)
    fig.update_yaxes(range=ylim)
    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
