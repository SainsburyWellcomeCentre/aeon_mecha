
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
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash.exceptions

sys.path.append("../..")

import aeon.preprocess.api
import aeon.preprocess.utils
import aeon.signalProcessing.utils
import aeon.preprocess.utils
import aeon.storage.sqlStorageMgr as sqlSM
import aeon.storage.filesStorageMgr as filesSM
import aeon.plotting.plot_functions

def getExperimentsNames():
    experiments_names = ["exp0.1-r0", "exp0.2-r0"]
    return experiments_names

def get_root(experiment_name):
    if experiment_name == "exp0.1-r0":
        root = "/ceph/aeon/aeon/data/raw/AEON/experiment0.1"
    elif experiment_name == "exp0.2-r0":
        root = "/ceph/aeon/aeon/data/raw/AEON2/experiment0.2"
    else:
        raise ValueError(f"Invalid experiment_name={experiment_name} in"
                          "get_root")
    return root


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", help="run the flask server only locally", action="store_true")
    parser.add_argument("--flask_port", help="port on which to run the falsh app", default=8050, type=int)
    parser.add_argument("--patches_coordinates", help="coordinates of patches", default="584,597,815,834;614,634,252,271")
    parser.add_argument("--debug", help="start GUI with debug functionality",
                        action="store_true")
    parser.add_argument("--nest_coordinates", help="coordinates of nest", default="170,260,450,540")
    parser.add_argument("--patchesToPlotSQL", help="Names of patches to plot", default="COM4,COM7")
    parser.add_argument("--patchesToPlotFiles", help="Names of patches to plot", default="Patch1,Patch2")
    parser.add_argument("--sample_rate_for_trajectory0", help="Initial value for the sample rate for the trajectory", default=0.5, type=float)
    parser.add_argument("--win_length_sec", help="Moving average window length (sec)", default=60.0, type=float)
    parser.add_argument("--reward_rate_time_resolution", help="Time resolution to compute the moving average (sec)", default=0.01, type=float)
    parser.add_argument("--pellet_event_label", help="Pellet event name to display", default="TriggerPellet")
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
    parser.add_argument("--subject_figure_width", help="width of the subject_figure plot", type=int, default=1000)
    parser.add_argument("--subject_figure_height", help="height of the subject_figure plot", type=int, default=1000)
    parser.add_argument("--travelled_distance_sample_rate", help="sampling rate for travelled distance plot", default=10.0, type=float)
    parser.add_argument("--storageMgr_type",
                        help="Type of storage manager (SQL | Files)",
                        type=str, default="SQL")
    parser.add_argument("--tunneled_host", help="Tunneled host IP address",
                        type=str, default="127.0.0.1")
    parser.add_argument("--db_server_port", help="Database server port",
                        type=int, default=3306)
    parser.add_argument("--db_user", help="DB user name", type=str,
                        default="rapela")
    parser.add_argument("--db_user_password", help="DB user password",
                        type=str, default="rapela-aeon")

    args = parser.parse_args()

    local = args.local
    flask_port = args.flask_port
    patches_coordinates_matrix = np.matrix(args.patches_coordinates)
    nest_coordinates_matrix = np.matrix(args.nest_coordinates)
    patches_to_plot_sql = args.patchesToPlotSQL.split(",")
    patches_to_plot_files = args.patchesToPlotFiles.split(",")
    sample_rate_for_trajectory0 = args.sample_rate_for_trajectory0
    win_length_sec = args.win_length_sec
    reward_rate_time_resolution = args.reward_rate_time_resolution
    pellet_event_label = args.pellet_event_label
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
    subject_figure_height = args.subject_figure_height
    subject_figure_width = args.subject_figure_width
    travelled_distance_sample_rate  = args.travelled_distance_sample_rate
    storageMgr_type = args.storageMgr_type
    tunneled_host=args.tunneled_host
    db_server_port = args.db_server_port
    db_user = args.db_user
    db_user_password = args.db_user_password

    if storageMgr_type == "SQL":
        storageMgr = sqlSM.SQLStorageMgr(host=tunneled_host,
                                         port=db_server_port,
                                         user=db_user,
                                         passwd=db_user_password)
        patches_to_plot = patches_to_plot_sql
    elif storageMgr_type == "Files":
        experiments_names = getExperimentsNames()
        root = get_root(experiment_name=experiments_names[0])
        storageMgr = filesSM.FilesStorageMgr(root=root)
        patches_to_plot = patches_to_plot_files
    else:
        raise ValueError(f"Invalid storageMgr_type={storageMgr_type}")

    experiments_names = getExperimentsNames()
    options_experiments_names = [{"label": experiment_name, "value": experiment_name} for experiment_name in experiments_names]
    cameras = ["FrameTop", "FramePatch1", "FramePatch2", "FrameNorth", "FrameSouth", "FrameEast", "FrameWest", "FrameGate"]
    options_cameras = [{"label": camera, "value": camera} for camera in cameras]

    def serve_layout():
        aDiv = html.Div(children=[
            html.H1(children="Behavioral Analysis Dashboard"),
            html.Hr(),
            html.H4(children="Experiment Name"),
            dcc.Dropdown(
                id="experimentNameDropdown",
                options=options_experiments_names,
                value=experiments_names[0],
                style={'width': '50%'},
            ),
            html.H4(children="Subject Name"),
            dcc.Dropdown(
                id="subjectNameDropdown",
                style={'width': '50%'},
            ),
            html.H4(children="Session Start Time"),
            dcc.Dropdown(
                id="sessionStartTimeDropdown",
                style={'width': '50%'},
            ),
            html.H4(children="Plotting Time (sec)"),
            html.Div(
                children=[
                    dcc.RangeSlider(
                        min=0,
                        max=10,
                        step=1,
                        value=[3,7],
                        id="plotTimeRangeSlider",
                    )
                ],
                hidden=False,
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
                                id="subject_graph_container",
                                children=[
                                    html.H4(children="Camera"),
                                    dcc.Dropdown(
                                        id="cameraDropdown",
                                        options=options_cameras,
                                        value=cameras[0],
                                        style={'width': '40%'},
                                    ),
                                    dcc.Graph(
                                        id="subject_graph",
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
                hidden=True),
        ])
        return aDiv

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    app.layout = serve_layout()

    @app.callback([Output('subjectNameDropdown', 'options'),
                   Output('subjectNameDropdown', 'value')], 
                  Input('experimentNameDropdown', 'value'))
    def get_subjects_names(experimentNameDropdown_value):
        print("Called get_subjects_names")
        subjects_names = storageMgr.getSubjectsNames(
            experiment_name=experimentNameDropdown_value)
        options_subjects_names = [{"label": subject_name, "value": subject_name} 
                                  for subject_name in subjects_names]
        return options_subjects_names, options_subjects_names[0]["value"]

    @app.callback([Output('sessionStartTimeDropdown', 'options'),
                   Output('sessionStartTimeDropdown', 'value'),
                   Output('sRateInputForTrajectoryPlot', 'value'),
                  ],
                  Input('subjectNameDropdown', 'value'),
                  State('experimentNameDropdown', 'value'),
                 )
    def get_sessions_start_times(subjectNameDropdown_value,
                                 experimentNameDropdown_value):
        print("Called get_sessions_start_times")
        sessions_start_times = storageMgr.getSubjectSessionsStartTimes(
            experiment_name=experimentNameDropdown_value,
            subject_name=subjectNameDropdown_value)
        options_sessions_start_times = [{"label": session_start_time, "value": session_start_time} for session_start_time in sessions_start_times]
        return options_sessions_start_times, sessions_start_times[0], sample_rate_for_trajectory0

    @app.callback([Output('nTrajectoryPointsToPlot', 'children'),],
                  [Input('sRateInputForTrajectoryPlot', 'value'),
                   Input('plotTimeRangeSlider', 'value'),],
                  State('experimentNameDropdown', 'value'),
                 )
    def get_num_trajectory_points_to_plot_label(sRateForTrajectoryPlot_value,
                                                plotTimeRangeSlider_value,
                                                experimentNameDropdown_value):
        print("Called get_num_trajectory_points_to_plot_label")
        if sRateForTrajectoryPlot_value is None or plotTimeRangeSlider_value is None:
            print("Preventing get_num_trajectory_points_to_plot_label update")
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
                  [State("experimentNameDropdown", "value"),
                   State("subjectNameDropdown", "value"),
                   State('plotTimeRangeSlider', 'min'),
                   State('plotTimeRangeSlider', 'max'),
                   State('plotTimeRangeSlider', 'marks'),
                   State('plotTimeRangeSlider', 'value')],
                  )
    def get_plotTimeRange_options(sessionStartTimeDropdown_value,
                                  startTimeInput_value,
                                  endTimeInput_value,
                                  experimentNameDropdown_value,
                                  subjectNameDropdown_value,
                                  plotTimeRangeSlider_min,
                                  plotTimeRangeSlider_max,
                                  plotTimeRangeSlider_marks,
                                  plotTimeRangeSlider_value):
        print("Called get_plotTimeRange_options")
        ctx = dash.callback_context
        if not ctx.triggered:
            print("not ctx.triggered")
            raise dash.exceptions.PreventUpdate
        print("ctx.triggered")
        print(ctx.triggered[0])
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if component_id == "sessionStartTimeDropdown":
            print("Entered sessionStartTimeDropdown")
            sessions_duration_sec = storageMgr.getSessionDuration(
                experiment_name=experimentNameDropdown_value,
                subject_name=subjectNameDropdown_value,
                session_start=sessionStartTimeDropdown_value)
            slider_min = 0
            slider_max = int(sessions_duration_sec)
            slider_value = [0, slider_max]
        elif component_id == "startTimeInput":
            print("Entered startTimeInput")
            print("marks=", plotTimeRangeSlider_marks)
            slider_min = plotTimeRangeSlider_min
            slider_max = plotTimeRangeSlider_max
            slider_value = [startTimeInput_value, plotTimeRangeSlider_value[1]]
        elif component_id == "endTimeInput":
            print("Entered endTimeInput")
            print("marks=", plotTimeRangeSlider_marks)
            slider_min = plotTimeRangeSlider_min
            slider_max = plotTimeRangeSlider_max
            slider_value = [plotTimeRangeSlider_value[0], endTimeInput_value]
        else:
            print("Entered else branch")
        slider_marks = dict(zip(range(0, slider_max, 600), [str(aNum) for aNum in range(0, slider_max, 600)]))
        # import pdb; pdb.set_trace()
        return slider_min, slider_max, slider_marks, slider_value


    @app.callback([Output('startTimeInput', 'value'),
                   Output('endTimeInput', 'value')],
                  [Input('plotTimeRangeSlider', 'value')])
    def set_start_end_inputs_from_slider_value(plotTimeRangeSlider_value):
        print("Called set_start_end_inputs_from_slider_value")
        if plotTimeRangeSlider_value is None:
            print("update prevented in set_start_end_inputs_from_slider_value ({:s})".format(flask.request.remote_addr))
            raise dash.exceptions.PreventUpdate
        print("plotTimeRangeSlider_value: ")
        print(plotTimeRangeSlider_value)
        return plotTimeRangeSlider_value


#     @app.callback([Output('auxStartTime', 'children'),
#                    Output('auxEndTime', 'children'),
#                    Output('plotTimeRangeSlider', 'value')],
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
                  [State("experimentNameDropdown", "value"),
                   State('sessionStartTimeDropdown', 'value'),
                   State('plotTimeRangeSlider', 'value'),
                   State('sRateInputForTrajectoryPlot', 'value')],
                  )
    def update_plots(plotButton_nClicks,
                     experimentNameDropdown_value,
                     sessionStartTimeDropdown_value,
                     plotTimeRangeSlider_value,
                     sRateForTrajectoryPlot_value):
        if plotButton_nClicks == 0:
            print("update prevented in update_plots ({:s})".format(flask.request.remote_addr))
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

        session_start_time = pd.Timestamp(sessionStartTimeDropdown_value)
        t0_absolute = session_start_time + datetime.timedelta(seconds=t0_relative)
        tf_absolute = session_start_time + datetime.timedelta(seconds=tf_relative)

        session_start_time_str = session_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")

        session_end_time = storageMgr.getSessionEndTime(
            experiment_name=experimentNameDropdown_value,
            session_start_time_str=session_start_time_str)
        session_end_time_str = session_end_time.strftime("%Y-%m-%d %H:%M:%S.%f")

        positions = storageMgr.getSessionPositions(
            session_start_time=session_start_time,
            start_offset_secs=t0_relative,
            duration_secs=tf_relative-t0_relative)
        timestamps = positions.index
        x = positions["x"].to_numpy()
        y = positions["y"].to_numpy()
        # timestamps_secs = (timestamps - timestamps[0])/np.timedelta64(1, "s")
        timestamps_secs = (timestamps - session_start_time)/np.timedelta64(1, "s")

        # trajectory figure
        fig_trajectory = go.Figure()
        patches_traces = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=patches_coordinates)
        for patch_trace in patches_traces:
            fig_trajectory.add_trace(patch_trace)
        nest_trace = aeon.plotting.plot_functions.get_patches_traces(patches_coordinates=nest_coordinates)[0]
        fig_trajectory.add_trace(nest_trace)
        trajectory_trace = aeon.plotting.plot_functions.get_trayectory_trace(x=x, y=y, timestamps=timestamps_secs, sample_rate=sRateForTrajectoryPlot_value, colorscale=trajectories_colorscale, opacity=trajectories_opacity)
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
            angles = storageMgr.getWheelAngles(
                experiment_name=experimentNameDropdown_value,
                session_start_time=session_start_time,
                start_offset_secs=t0_relative,
                duration_secs=tf_relative-t0_relative,
                patch_label=patch_to_plot)
            travelled_distance[patch_to_plot] = aeon.preprocess.api.distancetravelled(angles)
            if travelled_distance[patch_to_plot][-1]>max_travelled_distance:
                max_travelled_distance = travelled_distance[patch_to_plot][-1]
            travelled_seconds[patch_to_plot] = (travelled_distance[patch_to_plot].index-session_start_time).total_seconds()
            pellets_times = storageMgr.getFoodPatchEventTimes(
                start_time_str=t0_absolute,
                end_time_str=tf_absolute,
                event_label=pellet_event_label,
                patch_label=patch_to_plot)
            pellets_seconds[patch_to_plot] = (pellets_times-session_start_time).total_seconds()

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
        # fig_travelledDistance = go.Figure()

        # reward rate figure
#         pellets_seconds = {}
        reward_rate = {}
        max_reward_rate = -np.inf
        time = np.arange(t0_relative, tf_relative, reward_rate_time_resolution)
        for patch_to_plot in patches_to_plot:
#             pellets_times = storageMgr.getFoodPatchEventTimes(
#                 start_time_str=t0_absolute,
#                 end_time_str=tf_absolute,
#                 event_label=pellet_event_label,
#                 patch_label=patch_to_plot)
#             pellets_seconds[patch_to_plot] = (pellets_times-session_start_time).total_seconds()
            pellets_indices = ((pellets_seconds[patch_to_plot]-t0_relative)/reward_rate_time_resolution).astype(int)
            pellets_samples = np.zeros(len(time), dtype=np.double)
            if len(pellets_indices)>0:
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
            # import pdb; pdb.set_trace()

        plotsContainer_hidden = False
        plotButton_children = "Update"

        return fig_trajectory, fig_cumTimePerActivity, fig_travelledDistance, fig_rewardRate, plotsContainer_hidden, plotButton_children

    @app.callback(
        [Output('subject_graph', 'figure'),
         Output('subject_graph_container', 'hidden')],
        [Input('trajectoryGraph', 'clickData'),
         Input('cameraDropdown', 'value')],
        [State('sessionStartTimeDropdown', 'value'),
         State('experimentNameDropdown', 'value')]
    )
    def display_subject_figure(trajectory_graph_clickData,
                               camera_dropdown_value,
                               session_start_time_dropdown_value,
                               experimentNameDropdown_value):
        if trajectory_graph_clickData is None:
            raise dash.exceptions.PreventUpdate
        video_data_duration_sec = 0.1
        frame_delay = trajectory_graph_clickData["points"][0]["customdata"]
        session_start_time_dropdown_value = pd.to_datetime(session_start_time_dropdown_value)
        frame_start_time = session_start_time_dropdown_value + datetime.timedelta(seconds=frame_delay)
        video_data_end_time = session_start_time_dropdown_value + datetime.timedelta(seconds=frame_delay+video_data_duration_sec)
        root = get_root(experiment_name=experimentNameDropdown_value)
        video_data = aeon.preprocess.api.videodata(root, camera_dropdown_value, start=frame_start_time, end=video_data_end_time)
        first_two_video_data_rows = video_data.iloc[0:1]
        frame = next(aeon.preprocess.api.videoframes(first_two_video_data_rows))
        # fig = px.imshow(frame, color_continuous_scale="gray")
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig = go.Figure(data=go.Image(z=frame), layout=layout)
        fig.update_layout(height=subject_figure_height, width=subject_figure_width)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        subject_graph_container_hidden = False
        return fig, subject_graph_container_hidden

    if(local):
        app.run_server(debug=args.debug, port=flask_port)
    else:
        app.run_server(debug=args.debug, port=args.flask_port, host="0.0.0.0")

if __name__=="__main__":
    main(sys.argv)
