import glob
import datetime
import pandas as pd

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + datetime.timedelta(seconds=seconds)

def sessiondata(path):
    '''
    Extracts all session metadata from the specified root path, sorted chronologically,
    indicating start and end times of manual sessions in the Experiment 0 arena.

    :param str path: The root path where all the session data is stored.
    :return: A pandas data frame containing session event metadata, sorted by time.
    '''
    files = glob.glob(path + "/**/SessionData*.csv")
    files.sort()
    data = pd.concat(
        [pd.read_csv(file, header=None, names=['time','id','weight','event'])
         for file in files])
    data['time'] = data['time'].apply(aeon)
    data.set_index('time', inplace=True)
    return data

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
