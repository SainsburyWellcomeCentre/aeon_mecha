import os
import glob
import bisect
import datetime
import pandas as pd
import numpy as np

"""The size of each time bin, in whole hours."""
BIN_SIZE = 1
_SECONDS_PER_TICK = 32e-6

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + pd.to_timedelta(seconds, 's')

def timebin(time):
    '''
    Returns the whole hour time bin for a measurement timestamp.
    
    :param datetime or Series time: An object or series specifying the measurement timestamps.
    :return: A datetime object or series specifying the time bin for the measurement timestamp.
    '''
    if isinstance(time, pd.Series):
        hour = BIN_SIZE * (time.dt.hour // BIN_SIZE)
        return pd.to_datetime(time.dt.date) + pd.to_timedelta(hour, 'h')
    else:
        hour = BIN_SIZE * (time.hour // BIN_SIZE)
        return pd.to_datetime(datetime.datetime.combine(time.date(), datetime.time(hour=hour)))

def timebin_range(start, end):
    '''
    Returns a range of whole hour time bins.

    :param datetime start: The left bound of the time range.
    :param datetime end: The right bound of the time range.
    :return: A DatetimeIndex representing the range of time bins.
    '''
    return pd.date_range(timebin(start), timebin(end), freq=pd.DateOffset(hours=BIN_SIZE))

def timebin_key(file):
    """Returns the time bin key for the specified file name."""
    filename = os.path.split(file)[-1]
    filename = os.path.splitext(filename)[0]
    timebin_str = filename.split("_")[-1]
    date_str, time_str = timebin_str.split("T")
    return datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))

def timebin_filter(files, timefilter):
    '''
    Filters a list of paths using the specified time filter. To use the time filter, files
    must conform to a naming convention where the timestamp of each timebin is appended to
    the end of each file path.

    :param str files: The list of timebin files to filter.
    :param iterable or callable timefilter:
    A list of time bins or a predicate used to test each file time.
    :return: A list of all matching filenames.
    '''
    try:
        timebins = [timebin for timebin in iter(timefilter)]
        timefilter = lambda x:x in timebins
    except TypeError:
        if not callable(timefilter):
            raise TypeError("timefilter must be iterable or callable")

    matches = []
    for file in files:
        timebin = timebin_key(file)
        if not timefilter(timebin):
            continue
        matches.append(file)
    return matches

def load(path, reader, device, prefix=None, extension="*.csv",
         start=None, end=None, time=None, tolerance=None):
    '''
    Extracts data from matching files in the specified root path, sorted chronologically,
    containing device and/or session metadata for the Experiment 0 arena. If no prefix is
    specified, metadata for all sessions is extracted.

    :param str path: The root path where all the session data is stored.
    :param callable reader: A callable object used to load session metadata from a file.
    :param str, device: The device name prefix used to search for data files.
    :param str, optional prefix: The pathname prefix used to search for data files.
    :param str, optional extension: The optional extension pattern used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    if prefix is None:
        prefix = ""

    pathname = path + "/**/" + device + "/" + prefix + extension
    files = glob.glob(pathname)
    files.sort()

    if time is not None:
        # ensure input is converted to timestamp series
        if isinstance(time, pd.DataFrame):
            time = time.index
        if not isinstance(time, pd.Series):
            time = pd.Series(time)
            time.index = time

        dataframes = []
        filetimes = [timebin_key(file) for file in files]
        for key,values in time.groupby(by=timebin):
            i = bisect.bisect_left(filetimes, key)
            if i < len(filetimes):
                frame = reader(files[i])
            else:
                frame = reader(None)
            data = frame.reset_index()
            data.set_index('time', drop=False, inplace=True)
            data = data.reindex(values, method='pad', tolerance=tolerance)
            missing = len(data.time) - data.time.count()
            if missing > 0 and i > 0:
                # expand reindex to allow adjacent timebins
                # to fill missing values
                previous = reader(files[i-1])
                data = pd.concat([previous, frame])
                data = data.reindex(values, method='pad', tolerance=tolerance)
            else:
                data.drop(columns='time', inplace=True)
            dataframes.append(data)
        return pd.concat(dataframes)

    if start is not None or end is not None:
        timefilter = timebin_range(start, end)
        files = timebin_filter(files, timefilter)
    else:
        timefilter = None

    if len(files) == 0:
        return reader(None)

    data = pd.concat([reader(file) for file in files])
    if timefilter is not None:
        try:
            return data.loc[start:end]
        except KeyError:
            import warnings
            if not data.index.has_duplicates:
                warnings.warn('data index for {0} contains out-of-order timestamps!'.format(device))
                data = data.sort_index()
            else:
                warnings.warn('data index for {0} contains duplicate keys!'.format(device))
                data = data[~data.index.duplicated(keep='first')]
            return data.loc[start:end]
    return data

def timebinreader(file):
    """Reads timebin information for the specified file."""
    names = ['time', 'path']
    if file is None:
        return pd.DataFrame(columns=names[1:], index=pd.DatetimeIndex([]))
    timebin = timebin_key(file)
    data = pd.DataFrame({ names[0]: [timebin], names[1]: [file] })
    data.set_index('time', inplace=True)
    return data

def timebindata(path, device, extension='*.csv', start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all timebin information from the specified root path, sorted chronologically,
    indicating recorded timebins for the specified device in the arena.

    :param str path: The root path where all the data is stored.
    :param str, device: The device name prefix used to search for timebin files.
    :param str, optional extension: The optional extension pattern used to search for timebin files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing timebin information, sorted by time.
    '''
    return load(
        path,
        timebinreader,
        device,
        extension=extension,
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def sessionreader(file):
    """Reads session metadata from the specified file."""
    names = ['time','id','weight','event']
    if file is None:
        return pd.DataFrame(columns=names[1:], index=pd.DatetimeIndex([]))
    data = pd.read_csv(file, header=0, names=names)
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def sessiondata(path, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all session metadata from the specified root path, sorted chronologically,
    indicating start and end times of manual sessions in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    return load(
        path,
        sessionreader,
        device='SessionData',
        prefix='SessionData_2',
        extension="*.csv",
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def annotationreader(file):
    """Reads session annotations from the specified file."""
    names = ['time','id','annotation']
    if file is None:
        return pd.DataFrame(columns=names[1:], index=pd.DatetimeIndex([]))
    data = pd.read_csv(
        file,
        header=0,
        usecols=range(3),
        names=names)
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def annotations(path, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts session metadata from the specified root path, sorted chronologically,
    indicating event times of manual annotations in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing annotation metadata, sorted by time.
    '''
    return load(
        path,
        annotationreader,
        device='SessionData',
        prefix='SessionData_Annotations',
        extension="*.csv",
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def videoreader(file):
    """Reads video metadata from the specified file."""
    names = ['time','hw_counter','hw_timestamp']
    if file is None:
        return pd.DataFrame(columns=['frame']+names[1:], index=pd.DatetimeIndex([]))
    data = pd.read_csv(file, header=0, names=names)
    data.insert(loc=1, column='frame', value=data.index)
    data['time'] = aeon(data['time'])
    data['path'] = os.path.splitext(file)[0] + '.avi'
    data.set_index('time', inplace=True)
    return data

def videodata(path, device, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all video metadata from the specified root path, sorted chronologically,
    indicating synchronized trigger frame times for cameras in the Experiment 0 arena.

    :param str path: The root path where all the video data is stored.
    :param str, device: The device prefix used to search for video files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing frame event metadata, sorted by time.
    '''
    return load(
        path,
        videoreader,
        device=device,
        extension="*.csv",
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def videoclip(path, device, start=None, end=None):
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
    framedata = videodata(path, device, start=start, end=end)
    if len(framedata) == 0:
        return pd.DataFrame(columns=['start','duration'], index=pd.DatetimeIndex([]))
    videoclips = framedata.groupby('path')
    startframe = videoclips.frame.min().rename('start')
    duration = (videoclips.frame.max() - startframe).rename('duration')
    return pd.concat([startframe, duration], axis=1)

"""Maps Harp payload types to numpy data type objects."""
payloadtypes = {
    1 : np.dtype(np.uint8),
    2 : np.dtype(np.uint16),
    4 : np.dtype(np.uint32),
    8 : np.dtype(np.uint64),
    129 : np.dtype(np.int8),
    130 : np.dtype(np.int16),
    132 : np.dtype(np.int32),
    136 : np.dtype(np.int64),
    68 : np.dtype(np.float32)
}

"""Map of Harp device registers used in the Experiment 1 arena."""
harpregisters = {
    ('Patch', 90) : ['angle, intensity'],   # wheel encoder
    ('Patch', 35) : ['bitmask'],            # trigger pellet delivery
    ('Patch', 32) : ['bitmask'],            # pellet detected by beam break
    ('VideoController', 68) : ['pwm_mask'], # camera trigger times (top and side)
    ('FrameTop', 200) : ['x', 'y', 'angle', 'major', 'minor', 'area'],
}

def harpreader(file, names=None):
    '''
    Reads Harp data from the specified file.
    
    :param str file: The path to a Harp binary file.
    :param str or array-like names: The optional column labels to use for the data.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    if file is None:
        return pd.DataFrame(columns=names, index=pd.DatetimeIndex([]))

    data = np.fromfile(file, dtype=np.uint8)
    if len(data) == 0:
        return pd.DataFrame(columns=names, index=pd.DatetimeIndex([]))

    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = ticks * _SECONDS_PER_TICK + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))
    time = aeon(seconds)
    time.name = 'time'
    return pd.DataFrame(payload, index=time, columns=names)

def harpdata(path, device, register, names=None, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all harp data from the specified root path, sorted chronologically,
    for an individual register acquired from a device in the Experiment 0 arena.

    :param str path: The root path where all the data is stored.
    :param str device: The device name used to search for data files.
    :param int register: The register number to extract data for.
    :param str or array-like names: The optional column labels to use for the data.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    return load(
        path,
        lambda file: harpreader(file, names),
        device=device,
        prefix="{0}_{1}".format(device, register),
        extension="*.bin",
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def encoderdata(path, device, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all encoder data from the specified root path, sorted chronologically,
    from the specified patch controller in the Experiment 0 arena.

    :param str path: The root path where all the data is stored.
    :param str device: The device name used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing encoder data, sorted by time.
    '''
    return harpdata(
        path,
        device, register=90,
        names=['angle', 'intensity'],
        start=start, end=end,
        time=time, tolerance=tolerance)

def pelletdata(path, device, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all pellet event data from the specified root path, sorted chronologically,
    indicating when delivery of a pellet was triggered and when pellets were detected
    by the feeder beam break.

    :param str path: The root path where all the session data is stored.
    :param str device: The device name used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing pellet event data, sorted by time.
    '''
    command = harpdata(path, device, register=35, names=['bitmask'],
                       start=start, end=end, time=time, tolerance=tolerance)
    beambreak = harpdata(path, device, register=32, names=['bitmask'],
                         start=start, end=end, time=time, tolerance=tolerance)
    command = command[command.bitmask == 0x80]
    beambreak = beambreak[beambreak.bitmask == 0x20]
    command['event'] = 'TriggerPellet'
    beambreak['event'] = 'PelletDetected'
    events = pd.concat([command.event, beambreak.event]).sort_index()
    return pd.DataFrame(events)

def patchreader(file):
    """Reads patch state metadata from the specified file."""
    names = ['time','threshold','d1','delta']
    if file is None:
        return pd.DataFrame(columns=names[1:], index=pd.DatetimeIndex([]))
    data = pd.read_csv(file, header=0, names=names)
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def patchdata(path, patch, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts patch metadata from the specified root path, sorted chronologically,
    indicating wheel threshold, d1 and delta state changes in the arena patch.

    :param str path: The root path where all the session data is stored.
    :param str patch: The patch name used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing patch state metadata, sorted by time.
    '''
    return load(
        path,
        patchreader,
        device=patch,
        prefix='{0}_State'.format(patch),
        extension="*.csv",
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def positiondata(path, device='FrameTop', start=None, end=None, time=None, tolerance=None):
    '''
    Extracts all position data from the specified root path, sorted chronologically,
    for the specified camera in the Experiment 0 arena.

    :param str path: The root path where all the data is stored.
    :param str device: The device name used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing position data, sorted by time.
    '''
    return harpdata(
        path,
        device, register=200,
        names=['x', 'y', 'angle', 'major', 'minor', 'area'],
        start=start,
        end=end,
        time=time,
        tolerance=tolerance)

def videoframes(data):
    '''
    Extracts the raw frames corresponding to the provided video metadata.

    :param DataFrame data:
    A pandas DataFrame where each row specifies video acquisition path and frame number.
    :return:
    An object to iterate over numpy arrays for each row in the DataFrame,
    containing the raw video frame data.
    '''
    import cv2
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

def distancetravelled(angle, radius=4.0):
    '''
    Calculates the total distance travelled on the wheel, by taking into account
    its radius and the total number of turns in both directions across time.

    :param Series angle: A series of magnetic encoder measurements.
    :param float radius: The radius of the wheel, in metric units.
    :return: The total distance travelled on the wheel, in metric units.
    '''
    maxvalue = int(np.iinfo(np.uint16).max >> 2)
    jumpthreshold = maxvalue // 2
    turns = angle.astype(int).diff()
    clickup = (turns < -jumpthreshold).astype(int)
    clickdown = (turns > jumpthreshold).astype(int) * -1
    turns = (clickup + clickdown).cumsum()
    distance = 2 * np.pi * radius * (turns + angle / maxvalue)
    distance = distance - distance[0]
    return distance

def sessionduration(data):
    '''
    Computes duration and summary metadata for each session, by subtracting the
    start and end times. Allows for missing data by trying to match session start times
    with subsequent end times. If the match fails, end session metadata is filled with NaN.

    :param DataFrame data: A pandas data frame containing session event metadata.
    :return: A pandas data frame containing duration and metadata for each session.
    '''
    start = data.event == 'Start'
    end = data[start.shift(1, fill_value=False)].reset_index()
    start = data[start].reset_index()
    data = start.join(end, lsuffix='_start', rsuffix='_end')
    valid_sessions = (data.id_start == data.id_end) & (data.event_end == 'End')
    data.loc[~valid_sessions, 'time_end'] = pd.NaT
    data.loc[~valid_sessions, 'weight_end'] = float('nan')
    data['duration'] = data.time_end - data.time_start
    data.rename({ 'time_start':'start', 'id_start':'id', 'time_end':'end'}, axis=1, inplace=True)
    return data[['id', 'weight_start', 'weight_end', 'start', 'end', 'duration']]

def exportvideo(frames, file, fps, fourcc=None):
    '''
    Exports the specified frame sequence to a new video file.

    :param iterable frames: An object to iterate over the raw video frame data.
    :param str file: The path to the exported video file.
    :param fps: The frame rate of the exported video.
    :param optional fourcc:
    Specifies the four character code of the codec used to compress the frames.
    '''
    import cv2
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
