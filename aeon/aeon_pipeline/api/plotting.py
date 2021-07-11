import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from aeon.preprocess import api as aeon_api
from aeon.aeon_pipeline import lab, experiment, tracking, analysis


def plot_sessions_statistics(subject_key):
    """
    Plot statistics of all sessions from the specified subject
        + Sessions' durations
        + Distance travelled per session
        + Time spent at each food patch per session
        + Distance travelled by the wheel at each food patch per session
    """
    subject_sessions = (experiment.Session * experiment.SessionEnd
                        & 'session_duration > 0.5' & subject_key)

    session_starts, durations, time_fraction_in_nest, distance_travelled = (
            subject_sessions * analysis.SessionStatistics).fetch(
        'session_start', 'session_duration', 'time_fraction_in_nest', 'distance_travelled',
        order_by='session_start')
    session_ind = np.arange(len(session_starts))

    food_patch_stats = {}
    for food_patch_key in (lab.FoodPatch & (analysis.SessionStatistics.FoodPatchStatistics
                                            & subject_key)).fetch('KEY'):
        time_spent, wheel_distance = ((analysis.SessionStatistics & subject_sessions).join(
            analysis.SessionStatistics.FoodPatchStatistics, left=True)
                                      & subject_key & food_patch_key).fetch(
            'time_fraction_in_patch', 'total_wheel_distance_travelled', order_by='session_start')

        food_patch_stats[food_patch_key['food_patch_serial_number']] = {
            'time_fraction_in_patch': time_spent,
            'total_wheel_distance_travelled': wheel_distance
        }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))
    ax1.bar(session_ind, durations, label='Total duration')
    ax1.set_ylabel("Session duration (hour)")
    ax1.set_title(subject_key['subject'])
    ax1.bar(session_ind, time_fraction_in_nest * durations, label='Time in nest', color='gray')
    ax1.legend()

    ax1b = ax1.twinx()
    ax1b.plot(session_ind, distance_travelled, 'k', label='distance travelled')
    ax1b.set_ylabel("Animal's distance travelled")

    ax2_bottom = np.zeros_like(time_fraction_in_nest)
    ax3_bottom = np.zeros_like(time_fraction_in_nest)
    for foodpatch, stats in food_patch_stats.items():
        ax2.bar(session_ind, stats['time_fraction_in_patch'],
                bottom=ax2_bottom, label=foodpatch)
        ax3.bar(session_ind, stats['total_wheel_distance_travelled'],
                bottom=ax3_bottom, label=foodpatch)
        ax2_bottom += stats['time_fraction_in_patch']
        ax3_bottom += stats['total_wheel_distance_travelled']

    ax2.legend()
    ax2.set_ylabel("Time fraction of the session's duration")
    ax3.legend()
    ax3.set_ylabel("Wheel's distance travelled")

    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks(session_ind)
    ax3.set_xticklabels([t.strftime('%Y-%m-%d %H-%M-%S') for t in session_starts], rotation=-30)

    return fig


def plot_session_trajectory(session_key):
    """
    Plot animal's trajectory in a session
    """
    if session_key in experiment.NeverExitedSession:
        raise ValueError('Bad session - subject never exited')

    session_start, session_end = (experiment.Session * experiment.SessionEnd & session_key).fetch1(
        'session_start', 'session_end')

    # subject's position data in the epochs
    timestamps, position_x, position_y, speed, area = (
            tracking.SubjectPosition & session_key).fetch(
        'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

    # stack and structure in pandas DataFrame
    position = pd.DataFrame(dict(x=np.hstack(position_x),
                                 y=np.hstack(position_y),
                                 speed=np.hstack(speed),
                                 area=np.hstack(area)),
                            index=np.hstack(timestamps))
    position = position[position.area < 2000]  # remove position data where area >= 2000

    time_stamps = (position.index - session_start).total_seconds().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, color_data, title in zip(axes, (time_stamps, position.speed),
                                     ('Time (s)', 'Speed (px/s)')):
        sc = ax.scatter(position.x, position.y, c=color_data, s=5,
                        alpha=0.7, cmap='rainbow')
        clb = fig.colorbar(sc, ax=ax)
        clb.ax.set_title(title)
        ax.set_aspect(1)
        ax.invert_yaxis()

    return fig


def plot_session_patch_interaction(session_key):
    if session_key in experiment.NeverExitedSession:
        raise ValueError('Bad session - subject never exited')

    raw_data_dir = experiment.Experiment.get_raw_data_directory(session_key)
    session_start, session_end = (experiment.Session & session_key).join(
        experiment.SessionEnd, left=True).proj(
        ..., session_end='IFNULL(session_end, NOW())').fetch1(
        'session_start', 'session_end')

    session_food_patches = (
            experiment.Session
            * experiment.ExperimentFoodPatch.join(experiment.ExperimentFoodPatch.RemovalTime, left=True)
            & session_key
            & 'session_start >= food_patch_install_time'
            & 'session_start < IFNULL(food_patch_remove_time, "2200-01-01")').proj(
        'food_patch_description')

    food_patches = {}
    for food_patch_key in session_food_patches.fetch(as_dict=True):
        # wheel data
        wheeldata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                         device=food_patch_key['food_patch_description'],
                                         start=pd.Timestamp(session_start),
                                         end=pd.Timestamp(session_end))
        wheeldata['wheel_distance_travelled'] = aeon_api.distancetravelled(wheeldata.angle).values

        # times in patch
        in_patch_timestamps = (analysis.SessionStatistics.FoodPatchStatistics
                               & session_key & food_patch_key).fetch1('in_patch_timestamps')

        # pellet events
        pellet_times = (experiment.FoodPatchEvent * experiment.EventType
                        & food_patch_key
                        & 'event_type = "PelletDetected"'
                        & f'event_time BETWEEN "{session_start}" AND "{session_end}"').fetch(
            'event_time')

        # distance from patch
        timestamps, distance = (tracking.SubjectPosition
                                * tracking.SubjectDistance.FoodPatch
                                & session_key & food_patch_key).fetch(
            'timestamps', 'distance')
        distance = pd.DataFrame(dict(distance=np.hstack(distance)),
                                index=np.hstack(timestamps))
        distance = distance[session_start:session_end]

        food_patches[food_patch_key['food_patch_description']] = {
            'wheeldata': wheeldata,
            'pellet_times': pellet_times,
            'in_patch_timestamps': in_patch_timestamps,
            'distance': distance
        }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    for i, (food_patch_name, food_patch_data) in enumerate(food_patches.items()):
        l, = ax2.plot(food_patch_data['wheeldata'].index,
                      food_patch_data['wheeldata'].wheel_distance_travelled,
                      label=food_patch_name)
        ax2.plot(food_patch_data['pellet_times'],
                 np.full_like(food_patch_data['pellet_times'],
                              100 + food_patch_data['wheeldata'].wheel_distance_travelled[-1]),
                 '.', color=l.get_color(), label=f'{food_patch_name}_pellets')
        ax2.plot(food_patch_data['in_patch_timestamps'],
                 np.full_like(food_patch_data['in_patch_timestamps'], -200 * (i + 2)),
                 '|', color=l.get_color(), label=f'times_in_{food_patch_name}')

        ax1.plot(food_patch_data['distance'].index,
                 food_patch_data['distance'].distance,
                 color=l.get_color(), alpha=0.7, label=food_patch_name)
        ax1.plot(food_patch_data['in_patch_timestamps'],
                 np.full_like(food_patch_data['in_patch_timestamps'], -20 * (i + 2)),
                 '|', color=l.get_color(), label=f'times_in_{food_patch_name}')

    ax1.set_title('Distance away from food patch')
    ax2.set_title('Cumulative wheel distance travelled')
    ax1.legend()
    ax2.legend()

    return fig
