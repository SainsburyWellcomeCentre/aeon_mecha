import numpy as np

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
    id_offset = 'id' + rsuffix
    event_onset = 'event' + lsuffix
    event_offset = 'event' + rsuffix
    time_onset = 'time' + lsuffix
    time_offset = 'time' + rsuffix

    data_onset = data.event == onset
    data_offset = data[data_onset.shift(1, fill_value=False)].reset_index()
    data_onset = data[data_onset].reset_index()
    data = data_onset.join(data_offset, lsuffix=lsuffix, rsuffix=rsuffix)
    valid_subjects = (data[id_onset] == data[id_offset]) & (data[event_offset] == offset)
    if ~valid_subjects.any():
        data_types = data.dtypes
        data.loc[~valid_subjects, [name for name in data.columns if rsuffix in name]] = None
        data = data.astype(data_types)
    data['duration'] = data[time_offset] - data[time_onset]
    data.rename({ time_onset:lonset, id_onset:'id', time_offset:loffset}, axis=1, inplace=True)
    data = data[['id'] + [name for name in data.columns if '_' in name] + [lonset, loffset, 'duration']]
    data.drop([id_offset, event_onset, event_offset], axis=1, inplace=True)
    return data