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

def visits(data):
    '''
    Computes duration, enter and exit times for a subject visit. Allows for missing data by
    trying to match subject enter times with subsequent exit times. If the match fails,
    subject exit metadata is filled with NaN. Any additional metadata columns in the data
    frame will be paired and included in the output.

    :param DataFrame data: A pandas data frame containing subject enter and exit events.
    :return: A pandas data frame containing duration and metadata for each visit.
    '''
    enter = data.event == 'Enter'
    exit = data[enter.shift(1, fill_value=False)].reset_index()
    enter = data[enter].reset_index()
    data = enter.join(exit, lsuffix='_enter', rsuffix='_exit')
    valid_subjects = (data.id_enter == data.id_exit) & (data.event_exit == 'Exit')
    if ~valid_subjects.any():
        data_types = data.dtypes
        data.loc[~valid_subjects, [name for name in data.columns if '_exit' in name]] = None
        data = data.astype(data_types)
    data['duration'] = data.time_exit - data.time_enter
    data.rename({ 'time_enter':'enter', 'id_enter':'id', 'time_exit':'exit'}, axis=1, inplace=True)
    data = data[['id'] + [name for name in data.columns if '_' in name] + ['enter', 'exit', 'duration']]
    data.drop(['id_exit', 'event_enter', 'event_exit'], axis=1, inplace=True)
    return data