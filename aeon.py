import os
import glob
import datetime
import pandas as pd
import numpy as np

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + pd.to_timedelta(seconds, 's')

def timebin(time, binsize=3):
    '''
    Returns the whole hour time bin for a measurement timestamp.
    
    :param datetime time: A datetime object specifying a measurement timestamp.
    :param int, optional binsize: The size of each time bin, in whole hours.
    :return: A datetime object specifying the time bin for the measurement timestamp.
    '''
    hour = binsize * (time.hour // binsize)
    return datetime.datetime.combine(time.date(), datetime.time(hour=hour))

def timebin_range(start, end, binsize=3):
    '''
    Returns a range of whole hour time bins.

    :param datetime start: The left bound of the time range.
    :param datetime end: The right bound of the time range.
    :param int, optional binsize: The size of each time bin, in whole hours.
    :return: A DatetimeIndex representing the range of time bins.
    '''
    startbin = timebin(start, binsize)
    endbin = timebin(end, binsize)
    return pd.date_range(startbin, endbin, freq=pd.DateOffset(hours=binsize))

def timebin_glob(pathname, timefilter=None):
    '''
    Returns a list of paths matching a filename pattern, with an optional time filter.
    To use the time filter, files must conform to a naming convention where the timestamp
    of each timebin is appended to the end of each file name.

    :param str pathname: The pathname pattern used to search for matching filenames.
    :param iterable or callable, optional timefilter:
    A list of time bins or a predicate used to test each file time.
    :return: A list of all matching filenames.
    '''
    files = glob.glob(pathname)
    files.sort()
    if timefilter is None:
        return files

    try:
        timebins = [timebin for timebin in iter(timefilter)]
        timefilter = lambda x:x in timebins
    except TypeError:
        if not callable(timefilter):
            raise TypeError("timefilter must be iterable or callable")

    matches = []
    for file in files:
        filename = os.path.split(file)[-1]
        filename = os.path.splitext(filename)[0]
        timebin_str = filename.split("_")[-1]
        date_str, time_str = timebin_str.split("T")
        timebin = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        if not timefilter(timebin):
            continue
        matches.append(file)
    return matches

def load(path, reader, prefix=None, extension="*.csv", start=None, end=None):
    '''
    Extracts data from matching files in the specified root path, sorted chronologically,
    containing device and/or session metadata for the Experiment 0 arena. If no prefix is
    specified, metadata for all sessions is extracted.

    :param str path: The root path where all the session data is stored.
    :param callable reader: A callable object used to load session metadata from a file.
    :param str, optional prefix: The optional pathname pattern used to search for data files.
    :param str, optional extension: The optional extension pattern used to search for data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    if start is not None or end is not None:
        timefilter = timebin_range(start, end)
    else:
        timefilter = None

    files = timebin_glob(path + "/**/" + prefix + extension, timefilter)
    data = pd.concat([reader(file) for file in files])
    if timefilter is not None:
        return data.loc[start:end]
    return data

def sessionreader(file):
    """Reads session metadata from the specified file."""
    data = pd.read_csv(file, header=None, names=['time','id','weight','event'])
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def sessiondata(path, start=None, end=None):
    '''
    Extracts all session metadata from the specified root path, sorted chronologically,
    indicating start and end times of manual sessions in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    return load(
        path,
        sessionreader,
        prefix='SessionData',
        extension="*.csv",
        start=start,
        end=end)

def videoreader(file):
    """Reads video metadata from the specified file."""
    data = pd.read_csv(file, header=None, names=['time','hw_counter','hw_timestamp'])
    data.insert(loc=1, column='frame', value=data.index)
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def videodata(path, prefix=None, start=None, end=None):
    '''
    Extracts all video metadata from the specified root path, sorted chronologically,
    indicating synchronized trigger frame times for cameras in the Experiment 0 arena.

    :param str path: The root path where all the video data is stored.
    :param str, optional prefix: The optional prefix used to search for video files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing frame event metadata, sorted by time.
    '''
    return load(
        path,
        videoreader,
        prefix=prefix,
        extension="*.csv",
        start=start,
        end=end)

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

"""Map of Harp device registers used in the Experiment 0 arena."""
harpregisters = {
    { 'PatchEvents', 90 } : ['angle, intensity'], # wheel encoder
    { 'PatchEvents', 35 } : ['bitmask'],          # trigger pellet delivery
    { 'PatchEvents', 32 } : ['bitmask'],          # pellet detected by beam break
    { 'VideoEvents', 68 } : ['pwm_mask'],         # camera trigger times (top and side)
}

def harpreader(file, names=None):
    '''
    Reads Harp data from the specified file.
    
    :param str file: The path to a Harp binary file.
    :param str or array-like names: The optional column labels to use for the data.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    data = np.fromfile(file, dtype=np.uint8)
    stride = data[1] + 2
    length = len(data) // stride
    payloadsize = stride - 12
    payloadtype = payloadtypes[data[4] & ~0x10]
    elementsize = payloadtype.itemsize
    payloadshape = (length, payloadsize // elementsize)
    seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
    micros = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
    seconds = micros * 32e-6 + seconds
    payload = np.ndarray(
        payloadshape,
        dtype=payloadtype,
        buffer=data, offset=11,
        strides=(stride, elementsize))
    time = aeon(seconds)
    time.name = 'time'
    return pd.DataFrame(payload, index=time, columns=names)

def harpdata(path, device, register, names=None, start=None, end=None):
    '''
    Extracts all harp data from the specified root path, sorted chronologically,
    for an individual register acquired from a device in the Experiment 0 arena.

    :param str path: The root path where all the data is stored.
    :param str device: The device name used to search for data files.
    :param int register: The register number to extract data for.
    :param str or array-like names: The optional column labels to use for the data.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing harp event data, sorted by time.
    '''
    return load(
        path,
        lambda file: harpreader(file, names),
        prefix="{0}_{1}*".format(device, register),
        extension="*.bin",
        start=start,
        end=end)

def encoderdata(path, device='PatchEvents', start=None, end=None):
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
        start=start, end=end)

def patchreader(file):
    """Reads patch state metadata from the specified file."""
    data = pd.read_csv(file, header=None, names=['time','threshold'])
    data['time'] = aeon(data['time'])
    data.set_index('time', inplace=True)
    return data

def patchdata(path, start=None, end=None):
    '''
    Extracts all patch metadata from the specified root path, sorted chronologically,
    indicating wheel threshold state changes in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing patch state metadata, sorted by time.
    '''
    return load(
        path,
        patchreader,
        prefix='WheelThreshold',
        extension="*.csv",
        start=start,
        end=end)

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
    start and end times. Assumes no missing data, i.e. the same number of start
    and end times.

    :param DataFrame data: A pandas data frame containing session event metadata.
    :return: A pandas data frame containing duration and metadata for each session.
    '''
    start = data[data.event == 'Start']
    end = data[data.event == 'End']
    duration = end.index - start.index
    data = start.drop(['weight','event'], axis=1)
    data['duration'] = duration
    data['start_weight'] = start.weight
    data['end_weight'] = end.weight.values
    return data
