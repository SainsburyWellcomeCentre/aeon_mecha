
import sys
import math
import argparse
import numpy as np
import pandas as pd
import datetime

import flask

import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash.exceptions

sys.path.append("../..")

import aeon.preprocess.api
import aeon.preprocess.utils
import aeon.signalProcessing.utils
import aeon.preprocess.utils
import aeon.plotting.plot_functions

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="run the flask server only locally", action="store_true")
    parser.add_argument("--port", help="port on which to run the falsh app", default=8050, type=int)
    parser.add_argument("--debug", help="start GUI with debug functionality",
                        action="store_true")
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--patchesToPlot", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--sample_rate_for_trajectory0", help="Initial value for the sample rate for the trajectory", default=0.5, type=float)
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=60.0, type=float)
    parser.add_argument("--reward_rate_time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")
    parser.add_argument("--xlabel_trajectory", help="xlabel for trajectory plot", default="x (pixels)")
    parser.add_argument("--ylabel_trajectory", help="ylabel for trajectory plot", default="y (pixels)")
    parser.add_argument("--ylabel_cumTimePerActivity", help="ylabel", default="Proportion of Time")
    parser.add_argument("--xlabel_travelledDistance", help="xlabel for travelledDistance plot", default="Time (sec)")
    parser.add_argument("--ylabel_travelledDistance", help="ylabel for travelledDistance plot", default="Travelled Distance (cm)")
    parser.add_argument("--xlabel_rewardRate", help="xlabel for reward rate", default="Time (sec)")
    parser.add_argument("--ylabel_rewardRate", help="ylabe for reward ratel", default="Reward Rate")
    parser.add_argument("--ylim_cumTimePerActivity", help="ylim cummulative time per activity plot", default="[0,1]")
    parser.add_argument("--travelled_distance_trace_color", help="travelled distance trace color", default="blue")
    parser.add_argument("--reward_rate_trace_color", help="reward rate trace color", default="blue")
    parser.add_argument("--pellet_line_color", help="pellet line color", default="red")
    parser.add_argument("--pellet_line_style", help="pellet line style", default="solid")
    parser.add_argument("--trajectories_width", help="width of the trajectories plot", type=int, default=1000)
    parser.add_argument("--trajectories_height", help="height of the trajectories plot", type=int, default=1000)
    parser.add_argument("--trajectories_colorscale", help="colorscale for trajectories", default="Rainbow")
    parser.add_argument("--trajectories_opacity", help="opacity for trajectories", default=0.3, type=float)
    parser.add_argument("--mouse_figure_width", help="width of the mouse_figure plot", type=int, default=1000)
    parser.add_argument("--mouse_figure_height", help="height of the mouse_figure plot", type=int, default=1000)
    parser.add_argument("--travelled_distance_sample_rate", help="sampling rate for travelled distance plot", default=10.0, type=float)

    args = parser.parse_args()

    local = args.local
    root = args.root
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    patches_to_plot = args.patchesToPlot.split(",")
    sample_rate_for_trajectory0 = args.sample_rate_for_trajectory0
    win_length_sec = args.win_length_sec
    reward_rate_time_resolution = args.reward_rate_time_resolution
    pellet_event_name = args.pellet_event_name
    xlabel_trajectory = args.xlabel_trajectory
    ylabel_trajectory = args.ylabel_trajectory
    ylabel_cumTimePerActivity = args.ylabel_cumTimePerActivity
    xlabel_travelledDistance = args.xlabel_travelledDistance
    ylabel_travelledDistance = args.ylabel_travelledDistance
    xlabel_rewardRate = args.xlabel_rewardRate
    ylabel_rewardRate = args.ylabel_rewardRate
    ylim_cumTimePerActivity = [float(str) for str in args.ylim_cumTimePerActivity[1:-1].split(",")]
    travelled_distance_trace_color = args.travelled_distance_trace_color
    reward_rate_trace_color = args.reward_rate_trace_color
    pellet_line_color = args.pellet_line_color
    pellet_line_style = args.pellet_line_style
    trajectories_height = args.trajectories_height
    trajectories_width = args.trajectories_width
    trajectories_colorscale = args.trajectories_colorscale
    trajectories_opacity = args.trajectories_opacity
    mouse_figure_height = args.mouse_figure_height
    mouse_figure_width = args.mouse_figure_width
    travelled_distance_sample_rate  = args.travelled_distance_sample_rate

    metadata = aeon.preprocess.api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = aeon.preprocess.api.sessionduration(metadata)

    mouse_names = metadata["id"].unique()
    options_mouse_names = [{"label": mouse_name, "value": mouse_name} for mouse_name in mouse_names]
    cameras = ["FrameTop", "FramePatch1", "FramePatch2", "FrameNorth", "FrameSouth", "FrameEast", "FrameWest", "FrameGate"]
    options_cameras = [{"label": camera, "value": camera} for camera in cameras]
    sessions_start_times = metadata[metadata["id"]==mouse_names[0]].index
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
                style={'width': '50%'},
            ),
            html.H4(children="Session Start Time"),
            dcc.Dropdown(
                id="sessionStartTimeDropdown",
                options=options_sessions_start_times,
                value=sessions_start_times[0],
                style={'width': '50%'},
            ),
            html.H4(children="Plotting Time (sec)"),
            dcc.RangeSlider(
                id="plotTimeRangeSlider",
            ),
            html.Label(children="Start Time (sec)"),
            dcc.Input(
                id="startTimeInput",
                type="number",
                value=float('nan'),
                style={'width': '10%'},
            ),
            html.Label(children="End Time (sec)"),
            dcc.Input(
                id="endTimeInput",
                type="number",
                value=float('nan'),
                style={'width': '10%'},
            ),
            html.H4(children="Sample Rate for Trajectory Plot"),
            dcc.Input(
                id="sRateInputForTrajectoryPlot",
                type="number",
                style={'width': '10%'},
            ),
            html.Label(id="nTrajectoryPointsToPlot"),
            html.Hr(),
            html.Button(children="Plot", id="plotButton", n_clicks=0),
            html.Div(
                id="plotsContainer",
                children=[
                    # html.H4(children="Trajectory"),
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    dcc.Graph(
                                        id="trajectoryGraph",
                                    ),
                                ],
                                style={'padding': 10, 'flex': 1}
                            ),
                            html.Div(
                                id="mouse_graph_container",
                                children=[
                                    html.H4(children="Camera"),
                                    dcc.Dropdown(
                                        id="cameraDropDown",
                                        options=options_cameras,
                                        value=cameras[0],
                                        style={'width': '40%'},
                                    ),
                                    dcc.Graph(
                                        id="mouse_graph",
                                    ),
                                ],
                                style={'padding': 10, 'flex': 1},
                                hidden=True,
                            )
                        ], 
                        style={'display': 'flex', 'flex-direction': 'row'}
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
                ],
                hidden=True)
        ])
        return aDiv

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.layout = serve_layout()

    @app.callback([Output('sessionStartTimeDropdown', 'options'),
                   Output('sessionStartTimeDropdown', 'value'),
                   Output('sRateInputForTrajectoryPlot', 'value'),
                  ],
                  Input('mouseNameDropDown', 'value'))
    def get_sessions_start_times(mouseNameDropDown_value):
        sessions_start_times = metadata[metadata["id"]==mouseNameDropDown_value]["start"]
        options_sessions_start_times = [{"label": session_start_time, "value": session_start_time} for session_start_time in sessions_start_times]
        return options_sessions_start_times, sessions_start_times.iloc[0], sample_rate_for_trajectory0

    @app.callback([Output('nTrajectoryPointsToPlot', 'children'),],
                  [Input('sRateInputForTrajectoryPlot', 'value'),
                   Input('plotTimeRangeSlider', 'value'),]
                 )
    def get_num_trajectory_points_to_plot_label(sRateForTrajectoryPlot_value,
                                                plotTimeRangeSlider_value):
        if sRateForTrajectoryPlot_value is None or plotTimeRangeSlider_value is None:
            raise dash.exceptions.PreventUpdate
        num_trajectory_points_to_plot = int((plotTimeRangeSlider_value[1]-plotTimeRangeSlider_value[0])*sRateForTrajectoryPlot_value)
        answer = ["Number of trajectory points to plot: {:d}".format(num_trajectory_points_to_plot)]
        return answer

    @app.callback([Output('plotTimeRangeSlider', 'min'),
                   Output('plotTimeRangeSlider', 'max'),
                   Output('plotTimeRangeSlider', 'marks'),
                   Output('plotTimeRangeSlider', 'value'),
                  ],
                  [Input('sessionStartTimeDropdown', 'value'),
                   Input('startTimeInput', 'value'), 
                   Input('endTimeInput', 'value')],
                  [State('plotTimeRangeSlider', 'min'),
                   State('plotTimeRangeSlider', 'max'),
                   State('plotTimeRangeSlider', 'marks'),
                   State('plotTimeRangeSlider', 'value')],
                  )
    def get_plotTimeRange_options(sessionStartTimeDropdown_value,
                                  startTimeInput_value,
                                  endTimeInput_value,
                                  plotTimeRangeSlider_min,
                                  plotTimeRangeSlider_max,
                                  plotTimeRangeSlider_marks,
                                  plotTimeRangeSlider_value):
        ctx = dash.callback_context
        if not ctx.triggered:
            print("not ctx.triggered")
            raise dash.exceptions.PreventUpdate
        print("ctx.triggered")
        print(ctx.triggered[0])
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if component_id == "sessionStartTimeDropdown":
            sessions_duration_sec = metadata[metadata.start == pd.to_datetime(sessionStartTimeDropdown_value)].duration.item().total_seconds()
            slider_min = 0
            slider_max = int(sessions_duration_sec)
            slider_value = [0, slider_max]
        elif component_id == "startTimeInput":
            print("marks=", plotTimeRangeSlider_marks)
            slider_min = plotTimeRangeSlider_min
            slider_max = plotTimeRangeSlider_max
            slider_value = [startTimeInput_value, plotTimeRangeSlider_value[1]]
        elif component_id == "endTimeInput":
            print("marks=", plotTimeRangeSlider_marks)
            slider_min = plotTimeRangeSlider_min
            slider_max = plotTimeRangeSlider_max
            slider_value = [plotTimeRangeSlider_value[0], endTimeInput_value]
        slider_marks = dict(zip(range(0, slider_max, 600), [str(aNum) for aNum in range(0, slider_max, 600)]))
        return slider_min, slider_max, slider_marks, slider_value

    @app.callback([Output('startTimeInput', 'value'),
                   Output('endTimeInput', 'value')],
                  [Input('plotTimeRangeSlider', 'value')])
    def set_start_end_inputs_from_slider_value(plotTimeRangeSlider_value):
        print("set_start_end_inputs_from_slider_value called")
        if plotTimeRangeSlider_value is None:
            raise dash.exceptions.PreventUpdate
        return plotTimeRangeSlider_value

#     @app.callback([Output('auxStartTime', 'children'),
#                    Output('auxEndTime', 'children'),
#                    Output('plotTimeRangeSlider', 'value'],
#                   [Input('startTimeInput', 'value'),
#                    Input('endTimeInput', 'value')],
#                   )
#     def set_slider_value_from_start_end_inputs(startTimeInput_children,
#                                                endTimeInput_children):
#         print("set_slider_value_from_start_end_inputs called")
#         if math.isnan(startTimeInput_children) or math.isnan(endTimeInput_children):
#             raise dash.exceptions.PreventUpdate
#         return "{:d}".format(startTimeInput_children), "{:d}".format(endTimeInput_children)

    @app.callback([Output('trajectoryGraph', 'figure'),
                   Output('activitiesGraph', 'figure'),
                   Output('travelledDistanceGraph', 'figure'),
                   Output('rewardRateGraph', 'figure'),
                   Output('plotsContainer', 'hidden'),
                   Output('plotButton', 'children'),
                  ],
                  [Input('plotButton', 'n_clicks')],
                  [State('mouseNameDropDown', 'value'),
                   State('sessionStartTimeDropdown', 'value'),
                   State('plotTimeRangeSlider', 'value'),
                   State('sRateInputForTrajectoryPlot', 'value')],
                  )
    def update_plots(plotButton_nClicks,
                     mouseNameDropDown_value,
                     sessionStartTimeDropdown_value,
                     plotTimeRangeSlider_value,
                     sRateForTrajectoryPlot_value):
        if plotButton_nClicks == 0:
            print("update prevented ({:s})".format(flask.request.remote_addr))
            raise dash.exceptions.PreventUpdate
        print("update_plots called ({:s})".format(flask.request.remote_addr))
        patches_coordinates = pd.DataFrame(data=patches_coordinates_matrix,
                                           columns=["lower_x", "higher_x",
                                                    "lower_y", "higher_y"])
        nest_coordinates = pd.DataFrame(data=nest_coordinates_matrix,
                                        columns=["lower_x", "higher_x",
                                                 "lower_y", "higher_y"])

        t0_relative = plotTimeRangeSlider_value[0]
        tf_relative = plotTimeRangeSlider_value[1]

        session_start = pd.Timestamp(sessionStartTimeDropdown_value)
        t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
        tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)
        position = aeon.preprocess.api.positiondata(root, start=t0_absolute, end=tf_absolute)

        x = position["x"].to_numpy()
        y = position["y"].to_numpy()
        time_stamps = (position.index-session_start).total_seconds().to_numpy()

        # trajectory figure
        fig_trajectory = go.Figure()
        patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
        for patch_trace in patches_traces:
            fig_trajectory.add_trace(patch_trace)
        nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
        fig_trajectory.add_trace(nest_trace)
        trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, time_stamps=time_stamps, sample_rate=sRateForTrajectoryPlot_value, colorscale=trajectories_colorscale, opacity=trajectories_opacity)
        fig_trajectory.add_trace(trajectory_trace)
        fig_trajectory.update_layout(xaxis_title=xlabel_trajectory,
                                     yaxis_title=ylabel_trajectory,
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     height=trajectories_height,
                                     width=trajectories_width)
        fig_trajectory.update_yaxes(autorange="reversed") 

        # activity figure
        positions_labels = aeon.preprocess.utils.get_positions_labels(
            x=x, y=y,
            patches_coordinates=patches_coordinates,
            nest_coordinates=nest_coordinates)

        fig_cumTimePerActivity = go.Figure()
        cumTimePerActivity_trace = aeon.plotting.plot_functions.get_cumTimePerActivity_barplot_trace(
            positions_labels=positions_labels,
        )
        fig_cumTimePerActivity.add_trace(cumTimePerActivity_trace)
        fig_cumTimePerActivity.update_layout(yaxis_title=ylabel_cumTimePerActivity)
        fig_cumTimePerActivity.update_yaxes(range=ylim_cumTimePerActivity)

        # travelled distance figure
        travelled_distance = {}
        travelled_seconds = {}
        pellets_seconds = {}
        max_travelled_distance = -np.inf
        for patch_to_plot in patches_to_plot:
            wheel_encoder_vals = aeon.preprocess.api.encoderdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
            travelled_distance[patch_to_plot] = aeon.preprocess.api.distancetravelled(wheel_encoder_vals.angle)
            if travelled_distance[patch_to_plot][-1]>max_travelled_distance:
                max_travelled_distance = travelled_distance[patch_to_plot][-1]
            travelled_seconds[patch_to_plot] = (travelled_distance[patch_to_plot].index-session_start).total_seconds()
            pellet_vals = aeon.preprocess.api.pelletdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
            pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
            pellets_seconds[patch_to_plot] = (pellets_times-session_start).total_seconds()

        fig_travelledDistance = go.Figure()
        fig_travelledDistance = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot),
                                            subplot_titles=(patches_to_plot))
        for i, patch_to_plot in enumerate(patches_to_plot):
            trace = aeon.plotting.plot_functions.get_travelled_distance_trace(travelled_seconds=travelled_seconds[patch_to_plot], travelled_distance=travelled_distance[patch_to_plot], sample_rate=travelled_distance_sample_rate, color=travelled_distance_trace_color, showlegend=False)
            fig_travelledDistance.add_trace(trace, row=1, col=i+1)
            trace = aeon.plotting.plot_functions.get_pellets_trace(pellets_seconds=pellets_seconds[patch_to_plot], marker_color=pellet_line_color)
            fig_travelledDistance.add_trace(trace, row=1, col=i+1)
            if i==0:
                fig_travelledDistance.update_yaxes(title_text=ylabel_travelledDistance, range=(0, max_travelled_distance), row=1, col=i+1)
            else:
                fig_travelledDistance.update_yaxes(range=(-20, max_travelled_distance), row=1, col=i+1)
            fig_travelledDistance.update_xaxes(title_text=xlabel_travelledDistance, row=1, col=i+1)

        # reward rate figure
        pellets_seconds = {}
        reward_rate = {}
        max_reward_rate = -np.inf
        time = np.arange(t0_relative, tf_relative, reward_rate_time_resolution)
        for patch_to_plot in patches_to_plot:
            wheel_encoder_vals = aeon.preprocess.api.encoderdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
            pellet_vals = aeon.preprocess.api.pelletdata(root, patch_to_plot, start=t0_absolute, end=tf_absolute)
            pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
            pellets_seconds[patch_to_plot] = (pellets_times-session_start).total_seconds()
            pellets_indices = ((pellets_seconds[patch_to_plot]-t0_relative)/reward_rate_time_resolution).astype(int)
            pellets_samples = np.zeros(len(time), dtype=np.double)
            pellets_samples[pellets_indices] = 1.0
            win_length_samples = int(win_length_sec/reward_rate_time_resolution)
            reward_rate[patch_to_plot] = aeon.signalProcessing.utils.moving_average(values=pellets_samples, N=win_length_samples)
            patch_max_reward_rate = max(reward_rate[patch_to_plot])
            if patch_max_reward_rate>max_reward_rate:
                max_reward_rate = patch_max_reward_rate

        fig_rewardRate = go.Figure()
        fig_rewardRate = plotly.subplots.make_subplots(rows=1, cols=len(patches_to_plot),
                                                       subplot_titles=(patches_to_plot))
        for i, patch_to_plot in enumerate(patches_to_plot):
            trace = go.Scatter(x=time+win_length_sec/2.0,
                                y=reward_rate[patch_to_plot],
                                line=dict(color=reward_rate_trace_color),
                                showlegend=False)
            fig_rewardRate.add_trace(trace, row=1, col=i+1)
            trace = aeon.plotting.plot_functions.get_pellets_trace(pellets_seconds=pellets_seconds[patch_to_plot], marker_color=pellet_line_color)
            fig_rewardRate.add_trace(trace, row=1, col=i+1)
            if i == 0:
                fig_rewardRate.update_yaxes(title_text=ylabel_rewardRate, range=(0, max_reward_rate), row=1, col=i+1)
            else:
                fig_rewardRate.update_yaxes(range=(0, max_reward_rate), row=1, col=i+1)
            fig_rewardRate.update_xaxes(title_text=xlabel_rewardRate, row=1, col=i+1)

        plotsContainer_hidden = False
        plotButton_children = "Update"

        return fig_trajectory, fig_cumTimePerActivity, fig_travelledDistance, fig_rewardRate, plotsContainer_hidden, plotButton_children

    @app.callback(
        [Output('mouse_graph', 'figure'),
         Output('mouse_graph_container', 'hidden')],
        [Input('trajectoryGraph', 'clickData'),
         Input('cameraDropDown', 'value')],
        [State('sessionStartTimeDropdown', 'value')]
    )
    def display_mouse_figure(trajectory_graph_clickData,
                             camera_dropdown_value,
                             session_start_time_dropdown_value):
        if trajectory_graph_clickData is None:
            raise dash.exceptions.PreventUpdate
        video_data_duration_sec = 0.1
        frame_delay = trajectory_graph_clickData["points"][0]["customdata"]
        session_start_time_dropdown_value = pd.to_datetime(session_start_time_dropdown_value)
        frame_start_time = session_start_time_dropdown_value + datetime.timedelta(seconds=frame_delay)
        video_data_end_time = session_start_time_dropdown_value + datetime.timedelta(seconds=frame_delay+video_data_duration_sec)
        video_data = aeon.preprocess.api.videodata(root, camera_dropdown_value, start=frame_start_time, end=video_data_end_time)
        first_two_video_data_rows = video_data.iloc[0:1]
        frame = next(aeon.preprocess.api.videoframes(first_two_video_data_rows))
        # fig = px.imshow(frame, color_continuous_scale="gray")
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig = go.Figure(data=go.Image(z=frame), layout=layout)
        fig.update_layout(height=mouse_figure_height, width=mouse_figure_width)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        mouse_graph_container_hidden = False
        return fig, mouse_graph_container_hidden

    if(args.local):
        app.run_server(debug=args.debug, port=args.port)
    else:
        app.run_server(debug=args.debug, port=args.port, host="0.0.0.0")

if __name__=="__main__":
    main(sys.argv)
