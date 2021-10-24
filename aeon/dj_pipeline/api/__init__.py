import numpy as np
import matplotlib.pyplot as plt

from aeon.preprocess import api as aeon_api
from aeon.dj_pipeline import acquisition, paths

from aeon.dj_pipeline.analysis import is_position_in_nest


def get_video_frames(experiment_name, device,
                     start=None, end=None, time=None, tolerance=None,
                     plot=True):
    """
    Retrieve the video frames at the specified time points or time range, from the specified camera
    """
    repo_name, path = (acquisition.Experiment.Directory
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
