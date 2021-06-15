
import pdb
import sys

import argparse
import numpy as np
import pandas as pd
import datetime
import pathlib

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH

from aeon.query import exp0_api
import aeon.signalProcessing.utils
import aeon.preprocess.utils
import aeon.plotting.plot_functions

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="run the flask server only locally", action="store_true")
    parser.add_argument("--debug", help="start GUI with debug functionality", action="store_true")
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/data")
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="950,970,450,530")
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=10.0, type=float)
    parser.add_argument("--time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--video_frame_rate", help="Top camera frame rate (Hz)", default=50.0, type=float)
    parser.add_argument("--pellet_event_name", help="Pallete event name to display", default="TriggerPellet")
    parser.add_argument("--xlabel_trajectory", help="xlabel for trajectory plot", default="x (pixels)")
    parser.add_argument("--ylabel_trajectory", help="ylabel for trajectory plot", default="y (pixels)")
    parser.add_argument("--ylabel_cumTimePerActivity", help="ylabel", default="Proportion of Time")
    parser.add_argument("--xlabel_travelledTime", help="xlabel for travelledTime plot", default="Time (sec)")
    parser.add_argument("--ylabel_travelledTime", help="ylabel for travelledTime plot", default="Travelled Distance (cm)")
    parser.add_argument("--xlabel_rewardRate", help="xlabel for reward rate", default="Time (sec)")
    parser.add_argument("--ylabel_rewardRate", help="ylabe for reward ratel", default="Reward Rate")
    parser.add_argument("--pellet_line_color", help="pellet line color", default="red")
    parser.add_argument("--pellet_line_style", help="pellet line style", default="solid")
    parser.add_argument("--data_filename", help="data filename", default="/ceph/aeon/aeon/preprocessing/experiment0/BAA-1099590/2021-03-25T15-16-18/FrameTop.csv")

    args = parser.parse_args()

    local = args.local
    root = args.root
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    frame_rate = args.video_frame_rate
    win_length_sec = args.win_length_sec
    time_resolution = args.time_resolution
    pellet_event_name = args.pellet_event_name
    xlabel_trajectory = args.xlabel_trajectory
    ylabel_trajectory = args.ylabel_trajectory
    ylabel_cumTimePerActivity = args.ylabel_cumTimePerActivity
    xlabel_travelledTime = args.xlabel_travelledTime
    ylabel_travelledTime = args.ylabel_travelledTime
    xlabel_rewardRate = args.xlabel_rewardRate
    ylabel_rewardRate = args.ylabel_rewardRate
    pellet_line_color = args.pellet_line_color
    pellet_line_style = args.pellet_line_style
    data_filename = args.data_filename

    # <s Get good sessions
    # Get all session metadata from all `SessionData*` csv files (these are
    # 'start' and 'end files) within exp0 root.
    metadata = exp0_api.sessiondata(root)  # pandas df
    # Filter to only animal sessions (the others were test sessions).
    metadata = metadata[metadata.id.str.startswith('BAA')]
    # Drop bad sessions.
    metadata = metadata.drop([metadata.index[16], metadata.index[17], metadata.index[18]])
    # Match each 'start' with its 'end' to get more coherent sessions dataframe.
    metadata = exp0_api.sessionduration(metadata)
    # /s>
    mouse_names = metadata["id"].unique()
    options_mouse_names = [{"label": mouse_name, "value": mouse_name} for mouse_name in mouse_names]
    sessions_start_times = metadata.index.unique()
    options_sessions_start_times = [{"label": session_start_time, "value": session_start_time} for session_start_time in sessions_start_times]
    def serve_layout():
        aDiv = html.Div(children=[
            html.H1(children="Behavioral Analysis Dashboard"),
            html.Hr(),
            html.H4(children="Mouse Name"),
            dcc.Dropdown(
                id="mouseNameDropDown",
                options=options_mouse_names,
                value=mouse_names[0],
            ),
            html.H4(children="Patch ID"),
            dcc.Dropdown(
                id="patchIDDropDown",
                options=[
                    {'label': "0", "value": 0},
                ],
                value=0
            ),
            html.H4(children="Session Start Time"),
            dcc.Dropdown(
                id="sessionStartTimeDropdown",
                options=options_sessions_start_times,
                value=sessions_start_times[0],
            ),
            html.H4(children="Plotting Time (sec)"),
            dcc.RangeSlider(
                id="plotTimeRangeSlider",
                min=0,
                max=7845,
                step=60,
                marks=dict(zip(range(0, 7845, 600), [str(aNum) for aNum in range(0, 7845, 600)])),
                value=[600,1200]
            ),
            html.Button(children="Plot", id="plotButton", n_clicks=0),
            html.H4(children="Trajectory"),
            dcc.Graph(
                id="trajectoryGraph",
            ),
            html.H4(children="Activities"),
            dcc.Graph(
                id="activitiesGraph",
            ),
            html.H4(children="Distance Travelled"),
            dcc.Graph(
                id="travelledDistanceGraph",
            ),
            html.H4(children="Reward Rate"),
            dcc.Graph(
                id="rewardRateGraph",
            )
        ])
        return aDiv

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.layout = serve_layout()

    @app.callback([Output('trajectoryGraph', 'figure'),
                   Output('activitiesGraph', 'figure'),
                   Output('travelledDistanceGraph', 'figure'),
                   Output('rewardRateGraph', 'figure')],
                  [Input('plotButton', 'n_clicks')],
                  [State('mouseNameDropDown', 'value'),
                   State('patchIDDropDown', 'value'),
                   State('sessionStartTimeDropdown', 'value'),
                   State('plotTimeRangeSlider', 'value'),
                  ])
    def update_plots(plotButton_nClicks,
                     mouseNameDropDown_value,
                     patchIDDropDown_value,
                     sessionStartTimeDropdown_value,
                     plotTimeRangeSlider_value):
        print("update_plots called")
        patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                           columns=["lower_x", "higher_x",
                                                    "lower_y", "higher_y"])
        pos_data = pd.read_csv(pathlib.Path(data_filename),
                               names=["X", "Y", "Orientation", "MajorAxisLength", "MinoxAxisLength", "Area"])
        time = np.arange(pos_data.shape[0])/frame_rate

        t0 = plotTimeRangeSlider_value[0]
        tf = plotTimeRangeSlider_value[1]
        in_range_samples = np.where(np.logical_and(t0<=time, time<tf))[0]
        time = time[in_range_samples]
        t0 = time[0]
        tf = time[-1]
        pos_data = pos_data.iloc[in_range_samples]
        x = pos_data["X"].to_numpy()
        y = pos_data["Y"].to_numpy()
        time_stamps = pos_data.index.to_numpy()/frame_rate

        # trajectory figure
        fig_trajectory = go.Figure()
        trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps)
        fig_trajectory.add_trace(trajectory_trace)
        patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
        for patch_trace in patches_traces:
            fig_trajectory.add_trace(patch_trace)
        fig_trajectory.update_layout(xaxis_title=xlabel_trajectory,
                                     yaxis_title=ylabel_trajectory)

        # activity figure
        positions_labels = aeon.preprocess.utils.get_positions_labels(
            x=x, y=y, patches_coordinates=patches_coordinates)
        fig_cumTimePerActivity = go.Figure()
        cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
            positions_labels=positions_labels,
        )
        fig_cumTimePerActivity.add_trace(cumTimePerActivity_trace)
        fig_cumTimePerActivity.update_layout(yaxis_title=ylabel_cumTimePerActivity)

        # distance travelled figure
        session_start = pd.Timestamp(sessionStartTimeDropdown_value)
        t0_absolute = session_start + datetime.timedelta(seconds=t0)
        tf_absolute = session_start + datetime.timedelta(seconds=tf)
        wheel_encoder_vals = exp0_api.encoderdata(root, start=t0_absolute, end=tf_absolute)
        travelled_distance = exp0_api.distancetravelled(wheel_encoder_vals.angle)
        travelled_seconds = (travelled_distance.index-session_start).total_seconds()
        pellet_vals = exp0_api.pelletdata(root, start=t0_absolute, end=tf_absolute)
        pellets_times = pellet_vals.query("event == '{:s}'".format(pellet_event_name)).index
        pellets_absolute_seconds = (pellets_times-session_start).total_seconds()

        fig_travelledDistance = go.Figure()
        trace = aeon.plotting.plot_functions.get_travelling_distance_trace(travelled_seconds=travelled_seconds, travelled_distance=travelled_distance)
        fig_travelledDistance.add_trace(trace)
        for pellet_absolute_second in pellets_absolute_seconds:
            fig_travelledDistance.add_vline(x=pellet_absolute_second,
                                            line_color=pellet_line_color,
                                            line_dash=pellet_line_style)
        fig_travelledDistance.update_layout(xaxis_title=xlabel_travelledTime,
                                            yaxis_title=ylabel_travelledTime)

        # reward rate figure
        pellets_samples = np.zeros(len(time), dtype=np.double)
        pellets_seconds = (pellets_times-t0_absolute).total_seconds()
        pellets_indices = (pellets_seconds/time_resolution).astype(int)
        reward_rate_time = np.arange(t0, tf, time_resolution)
        pellets_samples = np.zeros(len(reward_rate_time), dtype=np.double)
        pellets_samples[pellets_indices] = 1.0
        win_length_samples = int(win_length_sec/time_resolution)
        reward_rate = aeon.signalProcessing.utils.moving_average(values=pellets_samples, N=win_length_samples)

        fig_rewardRate = go.Figure()
        trace = go.Scatter(x=reward_rate_time+win_length_sec/2.0, y=reward_rate)
        fig_rewardRate.add_trace(trace)
        for pellet_index in pellets_indices:
            fig_rewardRate.add_vline(x=reward_rate_time[pellet_index],
                                      line_color=pellet_line_color,
                                      line_dash=pellet_line_style)
        fig_rewardRate.update_layout(xaxis_title=xlabel_rewardRate,
                                     yaxis_title=ylabel_rewardRate)

        return fig_trajectory, fig_cumTimePerActivity, fig_travelledDistance, fig_rewardRate

    if(args.local):
        app.run_server(debug=args.debug)
    else:
        app.run_server(debug=args.debug, host="0.0.0.0")

if __name__=="__main__":
    main(sys.argv)
