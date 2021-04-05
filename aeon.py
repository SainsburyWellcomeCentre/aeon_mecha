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
    :return: An iterable object specifying the matching filenames.
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

    for file in files:
        filename = os.path.split(file)[-1]
        filename = os.path.splitext(filename)[0]
        timebin_str = filename.split("_")[-1]
        date_str, time_str = timebin_str.split("T")
        timebin = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        if not timefilter(timebin):
            continue
        yield file

def csvdata(path, prefix=None, names=None, timefilter=None):
    '''
    Extracts data from matching text files in the specified root path, sorted
    chronologically, containing device and/or session metadata for the Experiment 0
    arena. If no prefix is specified, the metadata for all manual sessions is extracted.

    :param str path: The root path where all the session data is stored.
    :param str prefix: The optional prefix used to search for session data files.
    :param array-like names: The list of column names to use for loading session data.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    files = timebin_glob(path + "/**/" + prefix + "*.csv", timefilter)
    data = pd.concat(
        [pd.read_csv(file, header=None, names=names)
         for file in files])
    data['time'] = data['time'].apply(aeon)
    data.set_index('time', inplace=True)
    return data

def sessiondata(path, timefilter=None):
    '''
    Extracts all session metadata from the specified root path, sorted chronologically,
    indicating start and end times of manual sessions in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    return csvdata(
        path,
        prefix='SessionData',
        names=['time','id','weight','event'],
        timefilter=timefilter)

def videodata(path, prefix=None, timefilter=None):
    '''
    Extracts all video metadata from the specified root path, sorted chronologically,
    indicating synchronized trigger frame times for cameras in the Experiment 0 arena.

    :param str path: The root path where all the video data is stored.
    :return: A pandas data frame containing frame event metadata, sorted by time.
    '''
    return csvdata(
        path,
        prefix=prefix,
        names=['time','frame','timestamp'],
        timefilter=timefilter)

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
