import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import argparse
import scipy.interpolate
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
    parser.add_argument("--duration", help="Duration (sec)", default=-1.0, type=float)
    parser.add_argument("--max_points", help="Maximum number of points to plot", default=100, type=int)
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (max={:.02f} sec) max_points={:d}, sample_rate={:.02f} Hz")
    # parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectory_session{:d}_start{:.02f}_end{:.02f}_maxPoints{:d}.{:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectories_aroundNaNs_session{:d}_start{:.02f}_end{:.02f}.{:s}")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    if args.duration > 0:
        tf_relative = args.start_time+args.duration
    else:
        tf_relative = None
    max_points = args.max_points
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
    session_start = metadata.iloc[session].start
    duration_sec = metadata.iloc[session].duration.total_seconds()
    if tf_relative is None:
        tf_relative = args.start_time+duration_sec
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
    time_stamps = (position.index-session_start).total_seconds().to_numpy()
    nan_run_thr = 5
    not_nan_indices_x = set(np.where(np.logical_not(np.isnan(x)))[0])
    not_nan_indices_y = set(np.where(np.logical_not(np.isnan(y)))[0])
    not_nan_indices = np.array(sorted(not_nan_indices_x.union(not_nan_indices_y)))
    not_nan_indices_largeGap_indices = np.where(np.diff(not_nan_indices)>nan_run_thr)[0]+1
    nan_indices_bounds = np.empty((len(not_nan_indices_largeGap_indices), 2), dtype=np.int)
    for i in range(len(not_nan_indices_largeGap_indices)):
        nan_indices_bounds[i, 0] = not_nan_indices[not_nan_indices_largeGap_indices[i]-1]
        nan_indices_bounds[i, 1] = not_nan_indices[not_nan_indices_largeGap_indices[i]]
    fig = go.Figure()
    patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
    for patch_trace in patches_traces:
        fig.add_trace(patch_trace)
    nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
    fig.add_trace(nest_trace)
    margin = 100
    for i in range(nan_indices_bounds.shape[0]):
        x_around_nan = x[(nan_indices_bounds[i, 0]-margin):(nan_indices_bounds[i, 1]+margin)]
        y_around_nan = y[(nan_indices_bounds[i, 0]-margin):(nan_indices_bounds[i, 1]+margin)]
        trace_bounds_start = go.Scatter(x=[x[nan_indices_bounds[i, 0]]], y=[y[nan_indices_bounds[i, 0]]], mode="markers", marker_symbol="cross", marker=dict(size=10, color="gray"), showlegend=False)
        fig.add_trace(trace_bounds_start)
        trace_bounds_end = go.Scatter(x=[x[nan_indices_bounds[i, 1]]], y=[y[nan_indices_bounds[i, 1]]], mode="markers", marker_symbol="circle", marker=dict(size=10, color="gray"), showlegend=False)
        fig.add_trace(trace_bounds_end)
        trace_around_nan = go.Scatter(x=x_around_nan, y=y_around_nan, name="Trajectory {:d}".format(i))
        fig.add_trace(trace_around_nan)
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, max_points, "png"))
    # fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, max_points, "html"))
    fig.write_image(fig_filename_pattern.format(session, t0_relative, tf_relative, "png"))
    fig.write_html(fig_filename_pattern.format(session, t0_relative, tf_relative, "html"))
    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
