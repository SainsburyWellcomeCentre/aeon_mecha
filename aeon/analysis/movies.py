import math

import cv2
import numpy as np
import pandas as pd

from aeon.io import video


def gridframes(frames, width, height, shape: None | int | tuple[int, int] = None):
    """Arranges a set of frames into a grid layout with the specified pixel dimensions and shape.

    :param list frames: A list of frames to include in the grid layout.
    :param int width: The width of the output grid image, in pixels.
    :param int height: The height of the output grid image, in pixels.
    :param optional shape:
    Either the number of frames to include, or the number of rows and columns
    in the output grid image layout.
    :return: A new image containing the arrangement of the frames in a grid.
    """
    if shape is None:
        shape = len(frames)
    if isinstance(shape, int):
        shape = math.ceil(math.sqrt(shape))
        shape = (shape, shape)

    dsize = (height, width, 3)
    cellsize = (height // shape[0], width // shape[1], 3)
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
    return cv2.convertScaleAbs(np.sum(np.multiply(1 / len(frames), frames)))


def groupframes(frames, n, fun):
    """Applies the specified function to each group of n-frames.

    :param iterable frames: A sequence of frames to process.
    :param int n: The number of frames in each group.
    :param callable fun: The function used to process each group of frames.
    :return: An iterable returning the results of applying the function to each group.
    """
    i = 0
    group = []
    for frame in frames:
        group.append(frame)
        if len(group) >= n:
            yield fun(group)
            group.clear()
            i = i + 1


def triggerclip(data, events, before=None, after=None):
    """Split video data around the specified sequence of event timestamps.

    :param DataFrame data:
    A pandas DataFrame where each row specifies video acquisition path and frame number.
    :param iterable events: A sequence of timestamps to extract.
    :param Timedelta before: The left offset from each timestamp used to clip the data.
    :param Timedelta after: The right offset from each timestamp used to clip the data.
    :return:
    A pandas DataFrame containing the frames, clip and sequence numbers for each event timestamp.
    """
    if before is None:
        before = pd.Timedelta(0)
    elif before is not pd.Timedelta:
        before = pd.Timedelta(before)

    if after is None:
        after = pd.Timedelta(0)
    elif after is not pd.Timedelta:
        after = pd.Timedelta(after)

    if events is not pd.Index:
        events = events.index

    clips = []
    for i, index in enumerate(events):
        clip = data.loc[(index - before) : (index + after)].copy()
        clip["frame_sequence"] = list(range(len(clip)))
        clip["clip_sequence"] = i
        clips.append(clip)
    return pd.concat(clips)


def collatemovie(clipdata, fun):
    """Collates a set of video clips into a single movie using the specified aggregation function.

    :param DataFrame clipdata:
    A pandas DataFrame where each row specifies video path, frame number, clip and sequence number.
    This DataFrame can be obtained from the output of the triggerclip function.
    :param callable fun: The aggregation function used to process the frames in each clip.
    :return: The sequence of processed frames representing the collated movie.
    """
    clipcount = len(clipdata.groupby("clip_sequence").frame_sequence.count())
    allframes = video.frames(clipdata.sort_values(by=["frame_sequence", "clip_sequence"]))
    return groupframes(allframes, clipcount, fun)


def gridmovie(clipdata, width, height, shape=None):
    """Collates a set of video clips into a grid movie with the specified pixel dimensions and grid layout.

    :param DataFrame clipdata:
    A pandas DataFrame where each row specifies video path, frame number, clip and sequence number.
    This DataFrame can be obtained from the output of the triggerclip function.
    :param int width: The width of the output grid movie, in pixels.
    :param int height: The height of the output grid movie, in pixels.
    :param optional shape:
    Either the number of frames to include, or the number of rows and columns
    in the output grid movie layout.
    :return: The sequence of processed frames representing the collated grid movie.
    """
    return collatemovie(clipdata, lambda g: gridframes(g, width, height, shape))
