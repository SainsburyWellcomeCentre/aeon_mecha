'''
Low-level api that interfaces with aeon data files.
'''

from pathlib import Path
import glob
import bisect
from datetime import datetime, time
import pandas as pd
from dotmap import DotMap
import pdb

# This could also be read from the ceph experiment folder
# Create a function that creates this DotMap obj
# DATA = DotMap({
#     'position': 'CameraTop_200',
#     'pellet_all': ('Patch1_35', 'Patch1_32', 'Patch2_35, Patch2_32'),
#     'wheel_all': ('Patch1_90', 'Patch2_90'),
#     'weight': 'Nest_200'})
# The duration of each acquisition chunk, in whole hours.
CHUNK_DURATION = 1
# The default sampling rate of the Harp behavior board, in seconds.
HARP_FS = 32e-6


def gen_data_dict(path):
    """
    Generates a data dictionary, linking the names of types of data to their respective
    raw data files in a given path.

    :param str OR Path; path: The path to the experiment data root directory
    :return DotMap; data_dict: The data dictionary
    """
    data_dict = DotMap()
    # Get all device names
    return data_dict



def gen_data_dict2(path):
    """
    Generates a data dictionary, linking the names of types of data to their respective
    raw data files in a given path.

    Inputs
    ------
    path : str OR Path
        The path to the experiment data root directory

    Outputs
    -------
    data_dict : DotMap
        The data dictionary
    """
    data_dict = DotMap()
    # Get all device names


def load(path, dataname, start_ts, end_ts, spec_ts, tol=None):
    """
    Loads one type of data, which can be across multiple files of the same type
    (as determined by a start and end timestamp), into a pandas DataFrame.

    Inputs
    ------
    path : str OR Path
        The path to the experiment data root directory
    dataname : str
        The type of data to return (e.g. video, audio, wheel encoder, etc.)
    start_ts : Timestamp
        The left bound of the time range of the data to load
    end_ts : Timestamp
        The right bound of the time range of the data to load
    spec_ts : Timestamp OR array-like of Timestamps
        A single or set of specific timestamps of the data to load
    tol : Timedelta
        The temporal tolerance for specified timestamps for inexact matches

    Outputs
    -------
    df : DataFrame
        The loaded data
    """
    pass


def get_chunk(ts=None, start_ts=None, end_ts=None, dur=CHUNK_DURATION):
    """
    Returns whole-hour acquisition chunk timestamps for given measurement timestamps.

    Inputs
    ------
    ts : datetime OR Timestamp OR array-like of datetimes or Timestamps (optional:
        required if `start_ts` and `end_ts` not provided)
        Specific measurement timestamp(s) for which to return chunk timestamp(s)
    start_ts : datetime OR Timestamp (optional: required if `ts` not provided)
        The left bound of the time range of chunk timestamp(s) to return
    end_ts : datetime OR Timestamp (optional: required if `ts` not provided)
        The right bound of the time range of chunk timestamp(s) to return
    dur : int
        The chunk duration, in whole hours.

    Outputs
    -------
    Series of Timestamps
        The whole-hour acquisition chunk timestamps.
    """
    # If `start_ts` and `end_ts` given, then call recursively and return date range
    # between start and end chunk ts.
    if (start_ts is not None) and (end_ts is not None):
        return pd.date_range(get_chunk(start_ts), get_chunk(end_ts),
                             freq=pd.DateOffset(hours=dur))
    # Convert timestamps to Series and return chunk timestamps.
    ts = pd.Series(ts) if not isinstance(ts, pd.Series) else ts
    chunk_hour = dur * (ts.dt.hour // dur)
    return pd.to_datetime(ts.dt.date) + pd.to_timedelta(chunk_hour, 'h')


def chunk_key():
    pass


def chunk_filter():
    pass


def chunk_glob():
    pass






