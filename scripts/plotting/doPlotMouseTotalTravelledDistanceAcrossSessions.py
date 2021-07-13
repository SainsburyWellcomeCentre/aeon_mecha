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

import aeon.preprocess.api as api                                                                                                     
import aeon.query.travelledDistance
import aeon.plotting.plot_functions as pf


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--mouse_id", help="Mouse name to plot", default="BAA-1099791")
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--ylabel", help="ylabel", default="Travelled Distance (cm)")
    parser.add_argument("--title_pattern", help="title pattern", default="{:s}")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_across_sessions_mouse{:s}.{:s}")

    args = parser.parse_args()

    root = args.root
    mouse_id = args.mouse_id
    patches_to_plot = args.patchesToPlot.split(",")
    ylabel = args.ylabel
    title_pattern = args.title_pattern
    fig_filename_pattern = args.fig_filename_pattern

    start = time.time()
    mouse_travelled_distances = aeon.query.travelledDistance.getMouseTotalTravelledDistanceAcrossSessions(mouse_id=mouse_id, root=root, patchesIDs=("Patch1", "Patch2"))
    elapsed = time.time()-start
    print("Elapsed time:", elapsed)

    start_times_str = [dt.strftime("%m/%d/%Y, %H:%M:%S") for dt in mouse_travelled_distances.index]
    traces = []
    for i in range(mouse_travelled_distances.shape[1]):
        traces.append(go.Bar(name=mouse_travelled_distances.columns[i],
                             x=start_times_str,
                             y=mouse_travelled_distances.iloc[:,i]))

    title = title_pattern.format(mouse_id)
    fig = go.Figure(data=traces)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title, barmode='stack')

    fig.write_image(fig_filename_pattern.format(mouse_id, "png"))
    fig.write_html(fig_filename_pattern.format(mouse_id, "html"))

    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
