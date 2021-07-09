import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from aeon.aeon_pipeline import lab, subject, experiment, tracking, session


def plot_sessions_statistics(subject_key):
    """
    Plot statistics of all sessions from the specified subject
        + Sessions' durations
        + Distance travelled per session
        + Time spent at each food patch per session
        + Distance travelled by the wheel at each food patch per session
    """
    durations, time_fraction_in_nest, distance_travelled = (
            session.Session * session.SessionStatistics & subject_key).fetch(
        'session_duration', 'time_fraction_in_nest', 'distance_travelled', order_by='session_start')
    session_ind = np.arange(len(durations))

    food_patch_stats = {}
    for food_patch in (lab.FoodPatch & (session.SessionStatistics.FoodPatchStatistics
                                        & subject_key)).fetch('KEY'):
        time_spent, wheel_distance = (session.SessionStatistics.join(
            session.SessionStatistics.FoodPatchStatistics, left=True)
                                      & subject_key & food_patch).fetch(
            'time_fraction_in_patch', 'wheel_distance_travelled', order_by='session_start')

        food_patch_stats[food_patch['food_patch_serial_number']] = {
            'time_fraction_in_patch': time_spent,
            'wheel_distance_travelled': wheel_distance
        }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))
    ax1.bar(session_ind, durations)
    ax1.set_ylabel("Session duration (hour)")
    ax1.set_title(subject_key['subject'])

    ax1b = ax1.twinx()
    ax1b.plot(session_ind, distance_travelled, 'k')
    ax1b.set_ylabel("Animal's distance travelled")

    ax2.bar(session_ind, time_fraction_in_nest, label='Nest', color='gray')

    for foodpatch, stats in food_patch_stats.items():
        ax2.bar(session_ind, stats['time_fraction_in_patch'], label=foodpatch)
        ax3.bar(session_ind, stats['wheel_distance_travelled'], label=foodpatch)

    ax2.legend()
    ax2.set_ylabel("Time fraction of the session's duration")
    ax3.legend()
    ax3.set_ylabel("Wheel's distance travelled")

    return fig


def plot_session_trajectory(session_key):
    """
    Plot animal's trajectory in a session
    """

    session_start, session_end = (session.Session & session_key).fetch1(
        'session_start', 'session_end')

    session_epochs = session.find_session_epochs(session_key)

    # subject's position data in the epochs
    timestamps, position_x, position_y, speed, area = (tracking.SubjectPosition & session_epochs).fetch(
        'timestamps', 'position_x', 'position_y', 'speed', 'area', order_by='epoch_start')

    # stack and structure in pandas DataFrame
    position = pd.DataFrame(dict(x=np.hstack(position_x),
                                 y=np.hstack(position_y),
                                 speed=np.hstack(speed),
                                 area=np.hstack(area)),
                            index=np.hstack(timestamps))
    position = position[session_start:session_end]
    position = position[position.area < 2000]  # remove position data where area >= 2000

    time_stamps = (position.index - session_start).total_seconds().to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, color_data, title in zip(axes, (time_stamps, position.speed),
                                     ('Time (s)', 'Speed (px/s)')):
        sc = ax.scatter(position.x, position.y, c=color_data,
                        alpha=0.7, cmap='rainbow')
        clb = fig.colorbar(sc, ax=ax)
        clb.ax.set_title(title)
        ax.set_aspect(1)
        ax.invert_yaxis()

    return fig
