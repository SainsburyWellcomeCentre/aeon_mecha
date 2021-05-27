
import sys
import pdb
import numpy as np
import pandas as pd
import pathlib
import argparse
import plotly.graph_objects as go

import aeon.plotting.plot_functions
import aeon.preprocess.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="950,970,450,530")
    parser.add_argument("--video_frame_rate", help="Top camera frame rate (Hz)", default=50.0, type=float)
    parser.add_argument("--ylabel", help="ylabel", default="Proportion of Time")
    parser.add_argument("--title_pattern", help="title pattern",
                        default="Plotting {:.02f} to {:0.2f} sec\n (min_time={:.02f}, max_time={:.02f})")
    parser.add_argument("--data_filename", help="data filename", default="/ceph/aeon/aeon/preprocessing/experiment0/BAA-1099590/2021-03-25T15-16-18/FrameTop.csv")
    parser.add_argument("--fig_filename_pattern", help="figure filenampattern", default="../../figures/activity_times_start{:.02f}_duration{:.02f}.{:s}")

    args = parser.parse_args()

    t0 = args.start_time
    duration = args.duration
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    frame_rate = args.video_frame_rate
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    data_filename = args.data_filename
    fig_filename_pattern = args.fig_filename_pattern

    tf = args.start_time+args.duration
    patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                       columns=["lower_x", "higher_x",
                                                "lower_y", "higher_y"])
    pos_data = pd.read_csv(pathlib.Path(data_filename),
                           names=["X", "Y", "Orientation", "MajorAxisLength",
                                  "MinoxAxisLength", "Area"])
    time = np.arange(pos_data.shape[0])/frame_rate
    min_time = np.min(time)
    max_time = np.max(time)
    in_range_samples = np.where(np.logical_and(t0<=time, time<tf))[0]
    t0 = time[in_range_samples[0]]
    tf = time[in_range_samples[-1]]
    pos_data = pos_data.iloc[in_range_samples]
    positions = pos_data[["X","Y"]]
    positions.columns = ["x", "y"]
    positions_labels = aeon.preprocess.utils.get_positions_labels(
        positions=positions, patches_coordinates=patches_coordinates)
    title = title_pattern.format(t0, tf, min_time, max_time)
    fig = go.Figure()
    cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
        positions_labels=positions_labels,
    )
    fig.add_trace(cumTimePerActivity_trace)
    fig.update_layout(title=title, yaxis_title=ylabel)
    fig.write_image(fig_filename_pattern.format(t0, duration, "png"))
    fig.write_html(fig_filename_pattern.format(t0, duration, "html"))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
