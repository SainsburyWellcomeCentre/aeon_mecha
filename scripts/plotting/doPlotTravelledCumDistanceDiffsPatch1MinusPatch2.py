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
import aeon.query.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/cum_travelled_distance_diffsPatch1MinusPatch2.{:s}")

    args = parser.parse_args()

    root = args.root
    fig_filename_pattern = args.fig_filename_pattern

    metadata = api.sessiondata(root)
    metadata = api.sessionduration(metadata)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    # metadata = metadata[(metadata.duration > pd.Timedelta('1h')) | metadata.duration.isna()]
    metadata = metadata[(metadata.duration > pd.Timedelta('1h'))]
    metadata = metadata[metadata.start >= pd.Timestamp('20210614')]

    nSessions = metadata.shape[0]
    longest_rateDiff = None
    mice_rateDiff_info = []
    mouse_ids = metadata.id.unique()
    for mouse_id in mouse_ids:
    # for mouse_id in mouse_ids[:2]:
        mouse_metadata = metadata[metadata.id == mouse_id]
        mouse_starts = []
        mouse_rateDiffs = []
        for i in range(mouse_metadata.shape[0]):
        # for i in range(2):
            start = mouse_metadata.iloc[i].start
            end = mouse_metadata.iloc[i].end
            print("Prcessing {:s}, start time {}".format(mouse_id, start))
            wheel_encoder_vals1 = api.encoderdata(root, 'Patch1', start=start, end=end)
            cum_travelled_distance1 = api.distancetravelled(wheel_encoder_vals1.angle)
#             travelled_distance1 = cum_travelled_distance1.diff()
            ma_cum_travelled_distance1 = aeon.query.utils.get_moving_average(x=cum_travelled_distance1, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s', center=True)
            wheel_encoder_vals2 = api.encoderdata(root, 'Patch2', start=start, end=end)
            cum_travelled_distance2 = api.distancetravelled(wheel_encoder_vals2.angle)
#             travelled_distance2 = cum_travelled_distance2.diff()
            ma_cum_travelled_distance2 = aeon.query.utils.get_moving_average(x=cum_travelled_distance2, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s', center=True)
            rateDiff = ma_cum_travelled_distance2-ma_cum_travelled_distance1
            len_rateDiff = len(rateDiff)
            if longest_rateDiff is None or len_rateDiff > longest_rateDiff:
                longest_rateDiff = len_rateDiff
            mouse_starts.append(start)
            mouse_rateDiffs.append(rateDiff)
        mice_rateDiff_info.append(dict(mouse_id=mouse_id, starts=mouse_starts, rateDiffs=mouse_rateDiffs))
    rateDiffs_matrix = np.empty((nSessions, longest_rateDiff))
    rateDiffs_matrix[:] = np.NaN
    max_session_elapsed_times = None
    y_labels = []
    row_index = 0
    for mouse_rateDiff_info in mice_rateDiff_info:
        for rateDiff_info_index, rateDiff_item in enumerate(mouse_rateDiff_info["rateDiffs"]):
            session_elapsed_times = rateDiff_item.index-rateDiff_item.index[0]
            if max_session_elapsed_times is None or session_elapsed_times[-1]>max_session_elapsed_times[-1]:
                max_session_elapsed_times = session_elapsed_times
            rateDiffs_matrix[row_index, :len(rateDiff_item)] = rateDiff_item.to_numpy()
            y_labels.append("{:s}_{:s}".format(mouse_rateDiff_info["mouse_id"], mouse_rateDiff_info["starts"][rateDiff_info_index].strftime("%m/%d/%Y")))
            row_index += 1
    print("rateDiffs_matrix.shape=", rateDiffs_matrix.shape)
    print("rateDiffs_matrix.min()=", np.nanmin(rateDiffs_matrix))
    print("rateDiffs_matrix.max()=", np.nanmax(rateDiffs_matrix))
    absZmax = np.nanmax(np.absolute(rateDiffs_matrix))
    fig = px.imshow(img=rateDiffs_matrix,
                    x=max_session_elapsed_times.total_seconds()/60, y=y_labels,
                    zmin=-absZmax, zmax=absZmax, aspect="auto", color_continuous_scale='RdBu_r',
                    labels=dict(color="Cum Travelled Dist (cm)<br>Patch2-Patch1"))
    fig.update_layout(
        xaxis_title="Time (min)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.write_image(fig_filename_pattern.format("png"))
    fig.write_html(fig_filename_pattern.format("html"))
    fig.show()

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
