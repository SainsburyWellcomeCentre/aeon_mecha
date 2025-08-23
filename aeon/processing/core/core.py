import numpy as np
import pandas as pd

def visits(data, onset='Enter', offset='Exit'):
    '''
    Computes duration, onset and offset times from paired events. Allows for missing data
    by trying to match event onset times with subsequent offset times. If the match fails,
    event offset metadata is filled with NaN. Any additional metadata columns in the data
    frame will be paired and included in the output.

    :param DataFrame data: A pandas data frame containing visit onset and offset events.
    :param str, optional onset: The label used to identify event onsets.
    :param str, optional offset: The label used to identify event offsets.
    :return: A pandas data frame containing duration and metadata for each visit.
    '''
    lonset = onset.lower()
    loffset = offset.lower()
    lsuffix = '_{0}'.format(lonset)
    rsuffix = '_{0}'.format(loffset)
    id_onset = 'id' + lsuffix
    event_onset = 'event' + lsuffix
    event_offset = 'event' + rsuffix
    time_onset = 'time' + lsuffix
    time_offset = 'time' + rsuffix

    # find all possible onset / offset pairs
    data = data.reset_index()
    data_onset = data[data.event == onset]
    data_offset = data[data.event == offset]
    data = pd.merge(data_onset, data_offset, on='id', how='left', suffixes=[lsuffix, rsuffix])

    # valid pairings have the smallest positive duration
    data['duration'] = data[time_offset] - data[time_onset]
    valid_visits = data[data.duration >= pd.Timedelta(0)]
    data = data.iloc[valid_visits.groupby([time_onset, 'id']).duration.idxmin()]
    data = data[data.duration > pd.Timedelta(0)]

    # duplicate offsets indicate missing data from previous pairing
    missing_data = data.duplicated(subset=time_offset, keep='last')
    if missing_data.any():
        data.loc[missing_data, ['duration'] + [name for name in data.columns if rsuffix in name]] = pd.NA

    # rename columns and sort data
    data.rename({ time_onset:lonset, id_onset:'id', time_offset:loffset}, axis=1, inplace=True)
    data = data[['id'] + [name for name in data.columns if '_' in name] + [lonset, loffset, 'duration']]
    data.drop([event_onset, event_offset], axis=1, inplace=True)
    data.sort_index(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def rate(events, window, frequency, weight=1, start=None, end=None, smooth=None, center=False):
    '''
    Computes the continuous event rate from a discrete event sequence, given the specified
    window size and sampling frequency.

    :param Series events: The discrete sequence of events.
    :param offset window: The time period of each window used to compute the rate.
    :param DateOffset, Timedelta or str frequency: The sampling frequency for the continuous rate.
    :param number, optional weight: A weight used to scale the continuous rate of each window.
    :param datetime, optional start: The left bound of the time range for the continuous rate.
    :param datetime, optional end: The right bound of the time range for the continuous rate.
    :param datetime, optional smooth: The size of the smoothing kernel applied to the continuous rate output.
    :param DateOffset, Timedelta or str, optional smooth:
    The size of the smoothing kernel applied to the continuous rate output.
    :param bool, optional center: Specifies whether to center the convolution kernels.
    :return: A Series containing the continuous event rate over time.
    '''
    counts = pd.Series(weight, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts = counts.resample(pd.Timedelta(1 / frequency, 's')).sum()
    rate = counts.rolling(window,center=center).sum()
    return rate.rolling(window if smooth is None else smooth,center=center).mean()

def get_events_rates(events, window_len_sec, frequency, unit_len_sec=60, start=None, end=None, smooth=None, center=False):
    # events is an array with the time (in seconds) of event occurence
    # window_len_sec is the size of the window over which the event rate is estimated
    # unit_len_sec is the length of one sample point
    window_len_sec_str = "{:d}S".format(window_len_sec)
    counts = pd.Series(1.0, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts_resampled = counts.resample(frequency).sum()
    counts_rolled = counts_resampled.rolling(window_len_sec_str,center=center).sum()*unit_len_sec/window_len_sec
    counts_rolled_smoothed = counts_rolled.rolling(window_len_sec_str if smooth is None else smooth, center=center).mean()
    return counts_rolled_smoothed

def sessiontime(index, start=None):
    """Converts absolute to relative time, with optional reference starting time."""
    if (start is None):
        start = index[0]
    return (index-start).total_seconds() / 60

def distance(position, target):
    """Computes the euclidean distance to a specified target."""
    return np.sqrt(np.square(position[['x','y']] - target).sum(axis=1))
