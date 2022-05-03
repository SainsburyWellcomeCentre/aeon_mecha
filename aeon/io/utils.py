import cv2


def write_frames(frames, file, fps, fourcc=None):
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
