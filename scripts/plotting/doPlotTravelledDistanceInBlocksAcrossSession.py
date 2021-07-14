import sys
import pdb
import time
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go
import plotly.subplots

sys.path.append("../../")

import aeon.preprocess.api as api                                                                                                     
import aeon.query.travelledDistance
import aeon.plotting.plot_functions as pf


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    # parser.add_argument("--session_start_time", help="Session start time", default="2021-06-23 08:05:46.224669933")
    parser.add_argument("--session_start_time", help="Session start time", default="")
    parser.add_argument("--block_duration", help="Block duration (sec)", type=float, default=40*60)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--ylabel", help="ylabel", default="Travelled Distance (cm)")
    parser.add_argument("--title_pattern", help="title pattern", default="{:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_in_blocks_across_session{:s}.{:s}")

    args = parser.parse_args()

    root = args.root
    if args.session_start_time=="":
        sessions_start_times = None
    else:
        sessions_start_times = [pd.Timestamp(args.session_start_time)]
    block_duration = args.block_duration
    patches_to_plot = args.patchesToPlot.split(",")
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    if sessions_start_times is None:
        sessions_start_times = aeon.query.utils.getAllSessionsStartTimes(root=root)

    for session_start_time in sessions_start_times:
        if session_start_time.strftime("%Y-%m-%d %H:%M:%S") in ["2021-06-22 14:49:30"]:
            continue

        print("Processing ", session_start_time)
        start = time.time()
        mouse_travelled_distances = aeon.query.travelledDistance.getTravelledDistanceInBlocksAcrossSession(session_start_time=session_start_time, block_duration=block_duration, root=root, patchesIDs=("Patch1", "Patch2"))
        elapsed = time.time()-start
        print("{:s} elapsed time: {:f}".format(session_start_time.strftime("%m_%d_%Y %H_%M_%S"), elapsed))

        datetimes_thr_changes = aeon.query.utils.getDatetimesOfThrChanges(session_start_time=session_start_time, session_end_time=session_end_time, patchesIDs=patchesIDs)

        start_times_str = [dt.strftime("%H:%M:%S") for dt in mouse_travelled_distances.index]
        traces = []
        for i in range(mouse_travelled_distances.shape[1]):
            traces.append(go.Bar(name=mouse_travelled_distances.columns[i],
                                 x=start_times_str,
                                 y=mouse_travelled_distances.iloc[:,i]))

        session_start_time_str = session_start_time.strftime("%m_%d_%Y-%H_%M_%S")
        title = title_pattern.format(session_start_time_str)
        fig = go.Figure(data=traces)
        fig.update_yaxes(title_text=ylabel)
        fig.update_layout(title=title, barmode='stack')

        fig.write_image(fig_filename_pattern.format(session_start_time_str, "png"))
        fig.write_html(fig_filename_pattern.format(session_start_time_str, "html"))

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
