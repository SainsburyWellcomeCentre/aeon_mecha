import numpy as np
import base64
import pandas as pd
import pathlib
import datetime
import cv2

from aeon.io import api as io_api
from aeon.io import video as io_video
import aeon.io.reader as io_reader


camera_name = "CameraTop"
start_time = datetime.datetime(2022, 4, 3, 13, 0, 0)
end_time = datetime.datetime(2022, 4, 3, 15, 0, 0)
raw_data_dir = pathlib.Path("/ceph/aeon/aeon/data/raw/AEON2/experiment0.2")


def retrieve_video_frames(
    camera_name,
    start_time,
    end_time,
    desired_fps=50,
    start_frame=0,
    chunk_size=50,
    **kwargs,
):
    # do some data loading
    videodata = io_api.load(
        root=raw_data_dir.as_posix(),
        reader=io_reader.Video(camera_name),
        start=pd.Timestamp(start_time),
        end=pd.Timestamp(end_time),
    )
    if not len(videodata):
        raise ValueError(
            f"No video data found for {camera_name} camera and time period: {start_time} - {end_time}"
        )

    framedata = videodata[start_frame : start_frame + chunk_size]

    # downsample
    # actual_fps = 1 / np.median(np.diff(videodata.index) / np.timedelta64(1, "s"))
    # final_fps = min(desired_fps, actual_fps)
    # ds_factor = int(np.around(actual_fps / final_fps))
    # framedata = videodata[::ds_factor]
    final_fps = desired_fps

    # read frames
    frames = io_video.frames(framedata)

    encoded_frames = []
    for f in frames:
        encoded_f = cv2.imencode(".jpeg", f)[1].tobytes()
        encoded_frames.append(base64.b64encode(encoded_f).decode())

    last_frame_time = framedata.index[len(encoded_frames) - 1]

    return {
        "frameMeta": {
            "fps": final_fps,
            "frameCount": len(encoded_frames),
            "endTime": str(last_frame_time),
            "finalChunk": bool(last_frame_time >= end_time),
        },
        "frames": encoded_frames,
    }
