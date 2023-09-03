import cv2
import math

import numpy as np
import pandas as pd


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
        for frameidx, path in zip(data._frame, data._path):
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


def gridframes(frames, width, height, shape=None):
    '''
    Arranges a set of frames into a grid layout with the specified
    pixel dimensions and shape.

    :param list frames: A list of frames to include in the grid layout.
    :param int width: The width of the output grid image, in pixels.
    :param int height: The height of the output grid image, in pixels.
    :param optional shape:
    Either the number of frames to include, or the number of rows and columns
    in the output grid image layout.
    :return: A new image containing the arrangement of the frames in a grid.
    '''
    if shape is None:
        shape = len(frames)
    if type(shape) not in [list,tuple]:
        shape = math.ceil(math.sqrt(shape))
        shape = (shape, shape)

    dsize = (height, width, 3)
    cellsize = (height // shape[0], width // shape[1],3)
    grid = np.zeros(dsize, dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            k = i * shape[1] + j
            if k >= len(frames):
                continue
            frame = frames[k]
            i0 = i * cellsize[0]
            j0 = j * cellsize[1]
            i1 = i0 + cellsize[0]
            j1 = j0 + cellsize[1]
            grid[i0:i1, j0:j1] = cv2.resize(frame, (cellsize[1], cellsize[0]))
    return grid

def averageframes(frames):
    """Returns the average of the specified collection of frames."""
    return cv2.convertScaleAbs(sum(np.multiply(1 / len(frames), frames)))

def groupframes(frames, n, fun):
    '''
    Applies the specified function to each group of n-frames.

    :param iterable frames: A sequence of frames to process.
    :param int n: The number of frames in each group.
    :param callable fun: The function used to process each group of frames.
    :return: An iterable returning the results of applying the function to each group.
    '''
    i = 0
    group = []
    for frame in frames:
        group.append(frame)
        if len(group) >= n:
            yield fun(group)
            group.clear()
            i = i + 1

def triggerclip(data, events, before=pd.Timedelta(0), after=pd.Timedelta(0)):
    '''
    Split video data around the specified sequence of event timestamps.

    :param DataFrame data:
    A pandas DataFrame where each row specifies video acquisition path and frame number.
    :param iterable events: A sequence of timestamps to extract.
    :param Timedelta before: The left offset from each timestamp used to clip the data.
    :param Timedelta after: The right offset from each timestamp used to clip the data.
    :return:
    A pandas DataFrame containing the frames, clip and sequence numbers for each event timestamp.
    '''
    if before is not pd.Timedelta:
        before = pd.Timedelta(before)
    if after is not pd.Timedelta:
        after = pd.Timedelta(after)
    if events is not pd.Index:
        events = events.index

    clips = []
    for i,index in enumerate(events):
        clip = data.loc[(index-before):(index+after)].copy()
        clip['frame_sequence'] = list(range(len(clip)))
        clip['clip_sequence'] = i
        clips.append(clip)
    return pd.concat(clips)

def collatemovie(clipdata, fun):
    '''
    Collates a set of video clips into a single movie using the specified aggregation function.

    :param DataFrame clipdata:
    A pandas DataFrame where each row specifies video path, frame number, clip and sequence number.
    This DataFrame can be obtained from the output of the triggerclip function.
    :param callable fun: The aggregation function used to process the frames in each clip.
    :return: The sequence of processed frames representing the collated movie.
    '''
    clipcount = len(clipdata.groupby('clip_sequence').frame_sequence.count())
    allframes = frames(clipdata.sort_values(by=['frame_sequence', 'clip_sequence']))
    return groupframes(allframes, clipcount, fun)

def gridmovie(clipdata, width, height, shape=None):
    '''
    Collates a set of video clips into a grid movie with the specified pixel dimensions
    and grid layout.

    :param DataFrame clipdata:
    A pandas DataFrame where each row specifies video path, frame number, clip and sequence number.
    This DataFrame can be obtained from the output of the triggerclip function.
    :param int width: The width of the output grid movie, in pixels.
    :param int height: The height of the output grid movie, in pixels.
    :param optional shape:
    Either the number of frames to include, or the number of rows and columns
    in the output grid movie layout.
    :return: The sequence of processed frames representing the collated grid movie.
    '''
    return collatemovie(clipdata, lambda g:gridframes(g, width, height, shape))