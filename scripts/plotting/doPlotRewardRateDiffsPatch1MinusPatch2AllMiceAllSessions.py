import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go
import plotly.subplots
import plotly.express as px

sys.path.append("../..")
import aeon.preprocess.api as api
import aeon.preprocess.utils
import aeon.signalProcessing.utils

def rate(events, window, frequency, weight=1, start=None, end=None, smooth=None, center=False):
    counts = pd.Series(weight, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts = counts.resample(frequency).sum()
    rate = counts.rolling(window,center=center).sum()
    return rate.rolling(window if smooth is None else smooth,center=center).mean()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=10.0, type=float)
    parser.add_argument("--time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--xlabel", help="xlabel", default="Time (sec)")
    parser.add_argument("--ylabel", help="ylabel", default="Reward Rate")
    parser.add_argument("--pellet_line_color", help="pellet line color", default="red")
    parser.add_argument("--pellet_line_style", help="pellet line style", default="solid")
    parser.add_argument("--title_pattern", help="title pattern", default="Start {:s}, from {:.02f} to {:0.2f} sec\n (min={:.02f}, max={:.02f})")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/reward_rate_diffsPatch1MinusPatch2_allSessions_allMice.{:s}")

    args = parser.parse_args()

    root = args.root
    session = args.session
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = api.sessionduration(metadata)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    # metadata = metadata[(metadata.duration > pd.Timedelta('1h')) | metadata.duration.isna()]
    metadata = metadata[(metadata.duration > pd.Timedelta('1h'))]
    metadata = metadata[metadata.start >= pd.Timestamp('20210622')]

    nSessions = metadata.shape[0]
    longest_rateDiff = None
    mice_rateDiff_info = []
    mouse_ids = metadata.id.unique()
    for mouse_id in mouse_ids:
        mouse_metadata = metadata[metadata.id == mouse_id]
        mouse_starts = []
        mouse_rateDiffs = []
        for i in range(mouse_metadata.shape[0]):
            start = mouse_metadata.iloc[i].start
            end = mouse_metadata.iloc[i].end
            print("Prcessing {:s}, start time {}".format(mouse_id, start))
            pellets1 = aeon.preprocess.api.pelletdata(root, 'Patch1', start=start, end=end)
            rate1 = rate(events=pellets1, window='600s', frequency='5s', weight=0.1, start=start, end=end, smooth='120s')
            pellets2 = aeon.preprocess.api.pelletdata(root, 'Patch2', start=start, end=end)
            rate2 = rate(events=pellets2, window='600s', frequency='5s', weight=0.1, start=start, end=end, smooth='120s')
            rateDiff = rate1-rate2
            # import matplotlib.pyplot as plt; plt.plot(rateDiff); plt.show()
            len_rateDiff = len(rateDiff)
            if longest_rateDiff is None or len_rateDiff > longest_rateDiff:
                longest_rateDiff = len_rateDiff
            mouse_starts.append(start)
            mouse_rateDiffs.append(rateDiff)
        mice_rateDiff_info.append(dict(mouse_id=mouse_id, starts=mouse_starts, rateDiffs=mouse_rateDiffs))
    rateDiffs_matrix = np.empty((nSessions, longest_rateDiff))
    rateDiffs_matrix[:] = np.NaN
    y_labels = []
    row_index = 0
    for mouse_rateDiff_info in mice_rateDiff_info:
        for rateDiff_info_index, rateDiff_item in enumerate(mouse_rateDiff_info["rateDiffs"]):
            rateDiffs_matrix[row_index, :len(rateDiff_item)] = rateDiff_item.to_numpy()
            y_labels.append("{:s}_{:s}".format(mouse_rateDiff_info["mouse_id"], mouse_rateDiff_info["starts"][rateDiff_info_index].strftime("%m/%d/%Y")))
            row_index += 1
    print("rateDiffs_matrix.shape=", rateDiffs_matrix.shape)
    print("rateDiffs_matrix.min()=", rateDiffs_matrix.min())
    print("rateDiffs_matrix.max()=", rateDiffs_matrix.max())
    fig = px.imshow(img=rateDiffs_matrix, y=y_labels, aspect="auto", labels=dict(color="Reward Rate<br>Patch1-Patch2"))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    fig.show()

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
