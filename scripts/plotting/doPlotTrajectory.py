import sys
import pdb
import numpy as np
import pandas as pd
import pathlib
import argparse
import plotly.graph_objects as go

import aeon.plotting.plot_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="950,970,450,530")
    parser.add_argument("--video_frame_rate", help="Top camera frame rate (Hz)", default=50.0, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="x (pixels)")
    parser.add_argument("--ylabel", help="ylabel", default="y (pixels)")
    parser.add_argument("--title_pattern", help="title pattern", default="Plotting {:.02f} to {:0.2f} sec\n (min_time={:.02f}, max_time={:.02f})")
    parser.add_argument("--data_filename", help="data filename", default="/ceph/aeon/aeon/preprocessing/experiment0/BAA-1099590/2021-03-25T15-16-18/FrameTop.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/trajectory_start{:.02f}_duration{:.02f}.{:s}")


    args = parser.parse_args()

    t0 = args.start_time
    tf = args.start_time+args.duration
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    frame_rate = args.video_frame_rate
    xlabel = args.xlabel
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                       columns=["lower_x", "higher_x",
                                                "lower_y", "higher_y"])
    pos_data = pd.read_csv(pathlib.Path(data_filename),
                           names=["X", "Y", "Orientation", "MajorAxisLength", "MinoxAxisLength", "Area"])
    time = np.arange(pos_data.shape[0])/frame_rate
    min_time = np.min(time)
    max_time = np.max(time)
    in_range_samples = np.where(np.logical_and(t0<=time, time<tf))[0]
    t0 = time[in_range_samples[0]]
    tf = time[in_range_samples[-1]]
    pos_data = pos_data.iloc[in_range_samples]
    x = pos_data["X"].to_numpy()
    y = pos_data["Y"].to_numpy()
    title = title_pattern.format(t0, tf, min_time, max_time)
    time_stamps = pos_data.index.to_numpy()/frame_rate
    fig = go.Figure()
    trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps)
    fig.add_trace(trajectory_trace)
    patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
    for patch_trace in patches_traces:
        fig.add_trace(patch_trace)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    # fig.update_coloraxes(showscale=True)

    fig.write_image(fig_filename_pattern.format(t0, tf, "png"))
    fig.write_html(fig_filename_pattern.format(t0, tf, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
