import bisect
import datetime
import pandas as pd
from pathlib import Path

"""The duration of each acquisition chunk, in whole hours."""
CHUNK_DURATION = 1

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + pd.to_timedelta(seconds, 's')

def chunk(time):
    '''
    Returns the whole hour acquisition chunk for a measurement timestamp.
    
    :param datetime or Series time: An object or series specifying the measurement timestamps.
    :return: A datetime object or series specifying the acquisition chunk for the measurement timestamp.
    '''
    if isinstance(time, pd.Series):
        hour = CHUNK_DURATION * (time.dt.hour // CHUNK_DURATION)
        return pd.to_datetime(time.dt.date) + pd.to_timedelta(hour, 'h')
    else:
        hour = CHUNK_DURATION * (time.hour // CHUNK_DURATION)
        return pd.to_datetime(datetime.datetime.combine(time.date(), datetime.time(hour=hour)))

def chunk_range(start, end):
    '''
    Returns a range of whole hour acquisition chunks.

    :param datetime start: The left bound of the time range.
    :param datetime end: The right bound of the time range.
    :return: A DatetimeIndex representing the acquisition chunk range.
    '''
    return pd.date_range(chunk(start), chunk(end), freq=pd.DateOffset(hours=CHUNK_DURATION))

def chunk_key(file):
    """Returns the acquisition chunk key for the specified file name."""
    epoch = file.parts[-3]
    chunk_str = file.stem.split("_")[-1]
    try:
        date_str, time_str = chunk_str.split("T")
    except ValueError:
        epoch = file.parts[-2]
        date_str, time_str = epoch.split("T")
    return epoch, datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))

def _set_index(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = aeon(data.index)
    data.index.name = 'time'

def _empty(columns):
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='time'))

def load(root, reader, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts chunk data from the root path of an Aeon dataset using the specified data stream
    reader. A subset of the data can be loaded by specifying an optional time range, or a list
    of timestamps used to index the data on file. Returned data will be sorted chronologically.

    :param str root: The root path, or prioritised sequence of paths, where epoch data is stored.
    :param Reader reader: A data stream reader object used to read chunk data from the dataset.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing epoch event metadata, sorted by time.
    '''
    if isinstance(root, str):
        root = [root]

    fileset = {
        chunk_key(fname):fname
        for path in root
        for fname in Path(path).glob(f"**/**/{reader.pattern}*.{reader.extension}")}
    files = sorted(fileset.items())

    if time is not None:
        # ensure input is converted to timestamp series
        if isinstance(time, pd.DataFrame):
            time = time.index
        if not isinstance(time, pd.Series):
            time = pd.Series(time)
            time.index = time

        dataframes = []
        filetimes = [chunk for (_, chunk), _ in files]
        files = [file for _, file in files]
        for key,values in time.groupby(by=chunk):
            i = bisect.bisect_left(filetimes, key)
            if i < len(filetimes):
                frame = reader.read(files[i])
                _set_index(frame)
            else:
                frame = _empty(reader.columns)
            data = frame.reset_index()
            data.set_index('time', drop=False, inplace=True)
            data = data.reindex(values, method='pad', tolerance=tolerance)
            missing = len(data.time) - data.time.count()
            if missing > 0 and i > 0:
                # expand reindex to allow adjacent chunks
                # to fill missing values
                previous = reader.read(files[i-1])
                data = pd.concat([previous, frame])
                data = data.reindex(values, method='pad', tolerance=tolerance)
            else:
                data.drop(columns='time', inplace=True)
            dataframes.append(data)

        if len(dataframes) == 0:
            return _empty(reader.columns)
            
        return pd.concat(dataframes)

    if start is not None or end is not None:
        chunkfilter = chunk_range(start, end)
        files = list(filter(lambda item: item[0][1] in chunkfilter, files))
    else:
        chunkfilter = None

    if len(files) == 0:
        return _empty(reader.columns)

    data = pd.concat([reader.read(file) for _, file in files])
    _set_index(data)
    if chunkfilter is not None:
        try:
            return data.loc[start:end]
        except KeyError:
            import warnings
            filtered_data = data
            if not data.index.is_monotonic_increasing:
                warnings.warn('data index for {0} contains out-of-order timestamps!'.format(reader.pattern))
                filtered_data = filtered_data.sort_index()
            if data.index.has_duplicates:
                warnings.warn('data index for {0} contains duplicate keys!'.format(reader.pattern))
                filtered_data = filtered_data[~filtered_data.index.duplicated(keep='first')]
            # get exact timestamp after start
            start_timestamp = filtered_data.index[filtered_data.index.get_indexer([start], method='bfill')[0]]
            # get exact timestamp after end
            end_timestamp = filtered_data.index[filtered_data.index.get_indexer([end], method='ffill')[0]]
            if isinstance(data.loc[start_timestamp], pd.DataFrame): # non-unique start index
                start_timestamp = data.index.get_loc(start_timestamp).nonzero()[0][0]
            else:
                start_timestamp = data.index.get_loc(start_timestamp)
            if isinstance(data.loc[end_timestamp], pd.DataFrame): # non-unique end index
                end_timestamp = data.index.get_loc(end_timestamp).nonzero()[0][-1]
            else:
                end_timestamp = data.index.get_loc(end_timestamp)
            end_timestamp = end_timestamp if end_timestamp + 1 > len(data) else end_timestamp + 1
            return data.iloc[start_timestamp:end_timestamp]
    return data
