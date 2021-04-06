import os
import glob
import datetime
import pandas as pd

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=seconds)

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

def load(path, reader, prefix=None, start=None, end=None):
    '''
    Extracts data from matching files in the specified root path, sorted chronologically,
    containing device and/or session metadata for the Experiment 0 arena. If no prefix is
    specified, metadata for all sessions is extracted.

    :param str path: The root path where all the session data is stored.
    :param callable reader: A callable object used to load session metadata from a file.
    :param str, optional prefix: The optional prefix used to search for session data files.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    if start is not None or end is not None:
        timefilter = timebin_range(start, end)
    else:
        timefilter = None

    files = timebin_glob(path + "/**/" + prefix + "*.csv", timefilter)
    data = pd.concat([reader(file) for file in files])
    data['time'] = data['time'].apply(aeon)
    data.set_index('time', inplace=True)

    if timefilter is not None:
        return data.loc[start:end]
    return data

def sessionreader(file):
    """Reads session metadata from the specified file."""
    return pd.read_csv(file, header=None, names=['time','id','weight','event'])

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
        start=start,
        end=end)

def videoreader(file):
    """Reads video metadata from the specified file."""
    data = pd.read_csv(file, header=None, names=['time','hw_counter','hw_timestamp'])
    data.insert(loc=1, column='frame', value=data.index)
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
        start=start,
        end=end)

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
