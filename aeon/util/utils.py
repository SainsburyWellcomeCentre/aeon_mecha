import pandas as pd

import aeon.io.api as api

def get_moving_average(x, window_len_sec, frequency, unit_len_sec=60, start=None, end=None, smooth=None, center=False):
    if start is not None and start < x.index[0]:
        x[start] = 0
    if end is not None and end > x.index[-1]:
        x[end] = 0
    x.sort_index(inplace=True)
    x_resampled = x.resample(frequency).sum()
    return x_resampled

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

def getPairedEvents(metadata):
    paired_events = None
    i = 0
    while i < (len(metadata)-1):
        if metadata.iloc[i]["event"]=="Start" and metadata.iloc[i+1]["event"]=="End":
            if paired_events is None:
                paired_events = metadata.iloc[i:(i+2)]
            else:
                paired_events = paired_events.append(metadata.iloc[i:(i+2)])
            i += 2
        else:
            i += 1
    return paired_events

def getMouseSessionsStartTimesAndDurations(mouse_id, root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    durations = metadata.loc[metadata.id == mouse_id, "duration"]
    return durations


def getAllSessionsStartTimes(root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    answer = metadata.index
    return answer


def getSessionsDuration(session_start_time, root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    duration = metadata.loc[session_start_time, "duration"].total_seconds()
    return duration