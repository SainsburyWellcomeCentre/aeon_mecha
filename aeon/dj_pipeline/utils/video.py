import base64
import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import aeon.io.reader as io_reader
from aeon.io import api as io_api
from aeon.io import video as io_video


def retrieve_video_frames(
    experiment_name,
    camera_name,
    start_time,
    end_time,
    raw_data_dir,
    desired_fps=50,
    start_frame=0,
    chunk_size=50,
    **kwargs,
):
    raw_data_dir = Path(raw_data_dir)
    assert raw_data_dir.exists()

    # Load video data
    videodata = io_api.load(
        root=raw_data_dir.as_posix(),
        reader=io_reader.Video(f"{camera_name}_*"),
        start=pd.Timestamp(start_time),
        end=pd.Timestamp(end_time),
    )
    if not len(videodata):
        raise ValueError(
            f"No video data found for {camera_name} camera and time period: {start_time} - {end_time}"
        )

    framedata = videodata[start_frame : start_frame + chunk_size]

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
