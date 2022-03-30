import cv2
import pandas as pd
from aeon.io.api import load
from aeon.io.reader import VideoReader

def clip(path, device, start=None, end=None):
    '''
    Extracts information about a continuous segment of video, possibly stored across
    multiple video files. For each video file covering the segment, a row is returned
    containing the path, start frame, and duration of the segment stored in that file.

    :param str path: The root path where all the video data is stored.
    :param str, device: The device prefix used to search for video files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing video clip storage information.
    '''
    framedata = load(path, device, start=start, end=end)
    if len(framedata) == 0:
        return pd.DataFrame(columns=['start','duration'], index=pd.DatetimeIndex([]))
    videoclips = framedata.groupby('path')
    startframe = videoclips.frame.min().rename('start')
    duration = (videoclips.frame.max() - startframe).rename('duration')
    return pd.concat([startframe, duration], axis=1)

def frames(data):
    '''
    Extracts the raw frames corresponding to the provided video metadata.

    :param DataFrame data:
    A pandas DataFrame where each row specifies video acquisition path and frame number.
    :return:
    An object to iterate over numpy arrays for each row in the DataFrame,
    containing the raw video frame data.
    '''
    capture = None
    filename = None
    index = 0
    try:
        for frameidx, path in zip(data.frame, data.path):
            if filename != path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(path)
                filename = path
                index = 0

            if frameidx != index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frameidx)
                index = frameidx
            success, frame = capture.read()
            if not success:
                raise ValueError('Unable to read frame {0} from video path "{1}".'.format(frameidx, path))
            yield frame
            index = index + 1
    finally:
        if capture is not None:
            capture.release()

def export(frames, file, fps, fourcc=None):
    '''
    Exports the specified frame sequence to a new video file.

    :param iterable frames: An object to iterate over the raw video frame data.
    :param str file: The path to the exported video file.
    :param fps: The frame rate of the exported video.
    :param optional fourcc:
    Specifies the four character code of the codec used to compress the frames.
    '''
    writer = None
    try:
        for frame in frames:
            if writer is None:
                if fourcc is None:
                    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
                writer = cv2.VideoWriter(file, fourcc, fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
