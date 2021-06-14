import datajoint as dj
import pathlib
import pandas as pd
import datetime
import numpy as np

from aeon.preprocess import api as aeon_api

from . import experiment
from . import get_schema_name, paths


schema = dj.schema(get_schema_name('tracking'))


@schema
class SubjectPosition(dj.Imported):
    definition = """
    -> experiment.Epoch.Subject
    ---
    timestamps:        longblob  # (datetime) timestamps of the position data
    position_x:        longblob  # (mm) animal's x-position, in the arena's coordinate frame
    position_y:        longblob  # (mm) animal's y-position, in the arena's coordinate frame
    position_z=null:   longblob  # (mm) animal's z-position, in the arena's coordinate frame
    area=null:         longblob  # (mm^2) animal's size detected in the camera
    speed=null:        longblob  # (mm/s) speed
    """

    def make(self, key):
        repo_name, preprocessing_path = (
                experiment.Experiment.Directory
                & key & 'directory_type = "preprocessing"').fetch1(
            'repository_name', 'directory_path')
        preprocess_dir = paths.get_repository_path(repo_name) / preprocessing_path

        time_bin_start, time_bin_end = (experiment.TimeBin * experiment.Epoch.Subject
                                        & key).fetch1('time_bin_start', 'time_bin_end')
        epoch_start, epoch_end = (experiment.Epoch.Subject & key).fetch1(
            'epoch_start', 'epoch_end')

        repo_name, file_path = (experiment.TimeBin.File * experiment.PipelineRepository
                                * experiment.Epoch.Subject
                                & 'data_source = "VideoCamera"'
                                & 'file_name LIKE "FrameTop%.avi"'
                                & key).fetch('repository_name', 'file_path', limit=1)
        file_path = paths.get_repository_path(repo_name[0]) / file_path[0]
        # Retrieve FrameTop video timestamps for this TimeBin
        video_timestamps = aeon_api.harpdata(file_path.parent.parent.as_posix(),
                                             device='VideoEvents',
                                             register=68,
                                             start=pd.Timestamp(time_bin_start),
                                             end=pd.Timestamp(time_bin_end))
        video_timestamps = video_timestamps[video_timestamps[0] == 4]  # frametop timestamps
        # Read preprocessed position data for this TimeBin and animal
        tracking_dfs = []
        for frametop_csv in (preprocess_dir / key['subject']).rglob('FrameTop.csv'):
            parent_timestamp = datetime.datetime.strptime(
                frametop_csv.parent.name, '%Y-%m-%dT%H-%M-%S')
            if parent_timestamp < time_bin_start or parent_timestamp >= time_bin_end:
                continue

            clips_csv = frametop_csv.parent / 'FrameTop-Clips.csv'
            clips = pd.read_csv(clips_csv)
            frametop_df = pd.read_csv(frametop_csv,
                                      names=['X', 'Y', 'Orientation', 'MajorAxisLength',
                                             'MinoxAxisLength', 'Area'])

            matched_clip_idx = clips[clips.path == file_path.as_posix()].index[0]
            matched_clip = clips.iloc[matched_clip_idx]

            harp_start_idx = matched_clip.start
            harp_end_idx = harp_start_idx + matched_clip.duration

            tracking_data_starting_idx = 0 if matched_clip_idx == 0 else clips.iloc[matched_clip_idx - 1].duration
            tracking_data = frametop_df[tracking_data_starting_idx:tracking_data_starting_idx + matched_clip.duration].copy()
            tracking_data['timestamps'] = video_timestamps[harp_start_idx:harp_end_idx].index
            tracking_data.set_index('timestamps', inplace=True)
            tracking_dfs.append(tracking_data)

        if not tracking_dfs:
            return

        timebin_tracking = pd.concat(tracking_dfs, axis=1).sort_index()
        epoch_tracking = timebin_tracking[np.logical_and(timebin_tracking.index >= epoch_start,
                                                         timebin_tracking.index < epoch_end)]

        timestamps = (epoch_tracking.index.values - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        timestamps = np.array([datetime.datetime.utcfromtimestamp(t) for t in timestamps])

        self.insert1({**key,
                      'timestamps': timestamps,
                      'position_x': epoch_tracking.X.values,
                      'position_y': epoch_tracking.Y.values,
                      'position_z': np.full_like(epoch_tracking.X.values, 0.0)})


@schema
class EpochPosition(dj.Computed):
    definition = """  # All unique positions (x,y,z) of an animal in a given epoch
    x: decimal(6, 2)
    y: decimal(6, 2)
    z: decimal(6, 2)
    -> SubjectPosition
    """

    def make(self, key):
        position_x, position_y, position_z = (SubjectPosition & key).fetch1(
            'position_x', 'position_y', 'position_z')
        unique_positions = set(list(zip(np.round(position_x, 2),
                                        np.round(position_y, 2),
                                        np.round(position_z, 2))))
        self.insert([{**key, 'x': x, 'y': y, 'z': z}
                     for x, y, z in unique_positions
                     if not np.any(np.where(np.isnan([x, y, z])))])
