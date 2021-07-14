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
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/reward_rate_diffsPatch1MinusPatch2_allSessions_allMice.{:s}")

    args = parser.parse_args()

    root = args.root
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
