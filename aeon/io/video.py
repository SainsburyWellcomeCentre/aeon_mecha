import cv2


def frames(data):
    """Extracts the raw frames corresponding to the provided video metadata.

    :param DataFrame data:
    A pandas DataFrame where each row specifies video acquisition path and frame number.
    :return:
    An object to iterate over numpy arrays for each row in the DataFrame,
    containing the raw video frame data.
    """
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
                raise ValueError(f'Unable to read frame {frameidx} from video path "{path}".')
            yield frame
            index = index + 1
    finally:
        if capture is not None:
            capture.release()


def export(frames, file, fps, fourcc=None):
    """Exports the specified frame sequence to a new video file.

    :param iterable frames: An object to iterate over the raw video frame data.
    :param str file: The path to the exported video file.
    :param fps: The frame rate of the exported video.
    :param optional fourcc:
    Specifies the four character code of the codec used to compress the frames.
    """
    writer = None
    try:
        for frame in frames:
            if writer is None:
                if fourcc is None:
                    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
                writer = cv2.VideoWriter(file, fourcc, fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
