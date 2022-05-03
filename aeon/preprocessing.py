"""Preprocesses aeon data. Typically called on data loaded from `aeon.io.api.load()`"""
import numpy as np
import pandas as pd
from dotmap import DotMap


def apply_bitmask(df, bitmask):
    """Filters a dataframe (`df`) by a Harp bitmask (`bitmask`)"""
    return df.iloc[np.where(df == bitmask)[0]]


def calc_wheel_cum_dist(angle, enc_res=2**14, radius=0.04):
    """Calculates cumulative wheel move distance from a Series containing encoder angle
    data (`angle`), the encoder bit res (`enc_res`), and wheel radius in m (`radius`)"""
    # Algo: Compute number of wheel turns (overflows - underflows) cumulatively at
    # each datapoint, then use this to compute cum dist at each datapoint.
    jump_thresh = enc_res // 2  # if diff in data > jump_thresh, assume over/underflow
    angle_diff = angle.diff()
    overflow = (angle_diff < -jump_thresh).astype(int)
    underflow = (angle_diff > jump_thresh).astype(int)
    turns = (overflow - underflow).cumsum()
    # cum_dist = circumference of wheel * fractional number of turns
    cum_dist = 2 * np.pi * radius * (turns + (angle / (enc_res - 1)))
    return cum_dist - cum_dist[0]


def visits(data, onset='Enter', offset='Exit'):
    """
    Computes duration, onset and offset times from paired events. Allows for missing data
    by trying to match event onset times with subsequent offset times. If the match fails,
    event offset metadata is filled with NaN. Any additional metadata columns in the data
    frame will be paired and included in the output.

    :param DataFrame data: A pandas data frame containing visit onset and offset events.
    :param str, optional onset: The label used to identify event onsets.
    :param str, optional offset: The label used to identify event offsets.
    :return: A pandas data frame containing duration and metadata for each visit.
    """
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
