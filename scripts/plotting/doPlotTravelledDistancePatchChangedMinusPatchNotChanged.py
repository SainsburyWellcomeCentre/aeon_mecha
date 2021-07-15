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
    parser.add_argument("--fig_filename_pattern", help="figure filename pattern", default="../../figures/travelled_distance_diffsChangedMinusNotChanged.{:s}")

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
    nSessions_with_thrChange = 0
    mice_rateDiff_info = []
    thr_change_elapsed_times_secs = []
    mouse_ids = metadata.id.unique()
    for mouse_id in mouse_ids:
        mouse_metadata = metadata[metadata.id == mouse_id]
        mouse_starts = []
        mouse_rateDiffs = []
        for i in range(mouse_metadata.shape[0]):
            start = mouse_metadata.iloc[i].start
            end = mouse_metadata.iloc[i].end
            print("Prcessing {:s}, start time {}".format(mouse_id, start))
            # begin check if the threshold changed for patch1 or patch2
            state1 = aeon.preprocess.api.patchdata(root, 'Patch1', start=start, end=end)
            thr_changes1 = state1["threshold"].diff()
            dates_changes1 = None
            dates_changes2 = None
            if not (thr_changes1[1:]==0).all():
                dates_changes1 = state1["threshold"].index[thr_changes1 > 0]
            else:
                state2 = aeon.preprocess.api.patchdata(root, 'Patch2', start=start, end=end)
                thr_changes2 = state2["threshold"].diff()
                if not (thr_changes2[1:]==0).all():
                    dates_changes2 = state2["threshold"].index[thr_changes2 > 0]
            changed_patch = None
            if dates_changes1 is not None:
                changed_patch = "Patch1"
                not_changed_patch = "Patch2"
                thr_change_elapsed_times_secs.append((dates_changes1-start).total_seconds().values[0])
            elif dates_changes2 is not None:
                changed_patch = "Patch2"
                not_changed_patch = "Patch1"
                thr_change_elapsed_times_secs.append((dates_changes2-start).total_seconds().values[0])
            # end  check if the threshold changed for patch1 or patch2
            if changed_patch is not None:
                nSessions_with_thrChange += 1

#                 changed_patch_pellets = aeon.preprocess.api.pelletdata(root, changed_patch, start=start, end=end)
#                 changed_patch_pellets = changed_patch_pellets[changed_patch_pellets.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch1

                changed_wheel_encoder_vals = api.encoderdata(root, changed_patch, start=start, end=end)
                changed_cum_travelled_distance = api.distancetravelled(changed_wheel_encoder_vals.angle)
                changed_travelled_distance = changed_cum_travelled_distance.diff()
                changed_ma_travelled_distance = aeon.query.utils.get_moving_average(x=changed_travelled_distance, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s', center=True)

                not_changed_wheel_encoder_vals = api.encoderdata(root, not_changed_patch, start=start, end=end)
                not_changed_cum_travelled_distance = api.distancetravelled(not_changed_wheel_encoder_vals.angle)
                not_changed_travelled_distance = not_changed_cum_travelled_distance.diff()
                not_changed_ma_travelled_distance = aeon.query.utils.get_moving_average(x=not_changed_travelled_distance, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s', center=True)

#                 not_changed_patch_pellets = aeon.preprocess.api.pelletdata(root, not_changed_patch, start=start, end=end)
#                 not_changed_patch_pellets = not_changed_patch_pellets[not_changed_patch_pellets.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch1

#                 changed_patch_rate = aeon.query.utils.get_events_rates(events=changed_patch_pellets, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s')
#                 not_changed_patch_rate = aeon.query.utils.get_events_rates(events=not_changed_patch_pellets, window_len_sec=600, frequency='5s', start=start, end=end, smooth='120s')
                rateDiff = changed_ma_travelled_distance-not_changed_ma_travelled_distance
                len_rateDiff = len(rateDiff)
                if longest_rateDiff is None or len_rateDiff > longest_rateDiff:
                    longest_rateDiff = len_rateDiff
                mouse_starts.append(start)
                mouse_rateDiffs.append(rateDiff)
        if len(mouse_rateDiffs)>0:
            mice_rateDiff_info.append(dict(mouse_id=mouse_id, starts=mouse_starts, rateDiffs=mouse_rateDiffs))
    rateDiffs_matrix = np.empty((nSessions_with_thrChange, longest_rateDiff))
    rateDiffs_matrix[:] = np.NaN
    y_labels = []
    max_session_elapsed_times = None
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
    thr_change_elapsed_times_secs = np.array(thr_change_elapsed_times_secs)
    fig = go.Figure()
    fig = px.imshow(img=rateDiffs_matrix,
                    x=max_session_elapsed_times.total_seconds()/60, y=y_labels,
                    zmin=-absZmax, zmax=absZmax, aspect="auto",
                    color_continuous_scale='PIYG', labels=dict(color="Travelled Distance (cm)<br>Changed-Not Changed<br>Patch"))
    change_times_trace = go.Scatter(x=thr_change_elapsed_times_secs/60, y=y_labels,  marker_symbol="triangle-right", marker_size=10, mode="markers")
    # change_times_trace = go.Scatter(x=thr_change_elapsed_times_secs/60, y=y_labels)
    fig.add_trace(change_times_trace)
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
