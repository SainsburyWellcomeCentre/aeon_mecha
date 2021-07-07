import datajoint as dj
import numpy as np
import matplotlib.pyplot as plt

from aeon.preprocess import api as aeon_api
from aeon.aeon_pipeline import lab, subject, experiment, tracking, session, paths


def get_video_frames(experiment_name, device,
                     start=None, end=None, time=None, tolerance=None,
                     plot=True):
    """
    Retrieve the video frames at the specified time points or time range, from the specified camera
    """
    repo_name, path = (experiment.Experiment.Directory
                       & 'directory_type = "raw"'
                       & {'experiment_name': experiment_name}).fetch1(
        'repository_name', 'directory_path')
    root = paths.get_repository_path(repo_name)
    raw_data_dir = root / path

    vd = aeon_api.videodata(raw_data_dir.as_posix(), device,
                            start=start, end=end, time=time, tolerance=tolerance)
    video_frames = aeon_api.videoframes(vd)

    if plot:
        col_count = 4
        video_frames = list(video_frames)
        fig, axes = plt.subplots(int(np.ceil(len(video_frames) / col_count)), col_count, figsize=(16, 8))
        for ax in axes.flatten():
            ax.set_axis_off()
        for timestamp, video_frame, ax in zip(time, video_frames, axes.flatten()):
            ax.set_axis_on()
            ax.imshow(video_frame)
            ax.set_xlabel(str(timestamp))
        return fig
    else:
        return video_frames


def plot_sessions_statistics(subject_key):
    """
    Plot statistics of all sessions from the specified subject
        + Sessions' durations
        + Distance travelled per session
        + Time spent at each food patch per session
        + Distance travelled by the wheel at each food patch per session
    """
    durations, distance_travelled = (session.Session * session.SessionStatistics
                                     & subject_key).fetch(
        'session_duration', 'distance_travelled', order_by='session_start')
    session_ind = np.arange(len(durations))

    food_patch_stats = {}
    for food_patch in (lab.FoodPatch & (session.SessionStatistics.FoodPatchStatistics
                                        & subject_key)).fetch('KEY'):
        time_spent, wheel_distance = (session.SessionStatistics.join(
            session.SessionStatistics.FoodPatchStatistics, left=True)
                                      & subject_key & food_patch).fetch(
            'time_spent_in_patch', 'wheel_distance_travelled', order_by='session_start')

        food_patch_stats[food_patch['food_patch_serial_number']] = {
            'time_spent_in_patch': time_spent,
            'wheel_distance_travelled': wheel_distance
        }

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 16))
    ax1.bar(session_ind, durations)
    ax1.set_ylabel("Session duration (hour)")
    ax1.set_title(subject_key['subject'])

    ax1b = ax1.twinx()
    ax1b.plot(session_ind, distance_travelled, 'k')
    ax1b.set_ylabel("Animal's distance travelled")

    for foodpatch, stats in food_patch_stats.items():
        ax2.bar(session_ind, stats['time_spent_in_patch'], label=foodpatch)
        ax3.bar(session_ind, stats['wheel_distance_travelled'], label=foodpatch)

    ax2.legend()
    ax2.set_ylabel('Time spent in patch (hour)')
    ax3.legend()
    ax3.set_ylabel("Wheel's distance travelled")

    return fig
