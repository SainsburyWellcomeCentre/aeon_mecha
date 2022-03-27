"""
Low-level api that interfaces with aeon data files.

Much of aeon data comes from Harp devices: https://www.cf-hw.org/harp) Info on
protocols used to read from these devices: https://github.com/harp-tech/protocol
"""
from datetime import datetime, time
from warnings import warn
from pathlib import Path

import pandas as pd
import numpy as np
from dotmap import DotMap
import ipdb


# @todo: we could take all of these constants and put them in a different module,
# e.g. `harp_config.py`?
# The duration of each acquisition chunk, in whole hours.
_CHUNK_DURATION = 1
# The default temporal resolution of the Harp behavior board, in seconds.
_HARP_RES = 32e-6
# Map of Harp payload types to datatypes, including endianness (to read by `ndarray()`)
_HARP_PTYPES = {
    1: '<u1',     # little-endian, unsigned int, 1-byte representation
    2: '<u2',
    4: '<u4',
    8: '<u8',
    129: '<i1',
    130: '<i2',
    132: '<i4',
    136: '<i8',   # little-endian, signed int, 8-byte representation
    68: '<f4'     # little-endian, float, 4-byte representation
}
_HARP_T_PTYPES = {
    17: '<u1',
    18: '<u2',
    20: '<u4',
    24: '<u8',
    145: '<i1',
    146: '<i2',
    148: '<i4',
    152: '<i8',
    84: '<f4'
}
# Bitmasks that link Harp payload datatypes:
# non-timestamped -> timestamped:
# e.g. _HARP_T_PTYPES[17] == (_HARP_PTYPES[1] | _HARP_N2T_BITMASK)
_HARP_N2T_MASK = 0x10
# unsigned -> signed: e.g. _HARP_PTYPES[129] == (_HARP_PTYPES[1] | _HARP_U2I_BITMASK)
_HARP_U2I_MASK = 0x80
# unsigned -> float: e.g. _HARP_PTYPES[68] == (_HARP_PTYPES[1] | _HARP_U2F_BITMASK)
_HARP_U2F_MASK = 0x40
# Map of Harp device registers used in the Experiment 1 arena.
# @todo this isn't used so should be moved out of this module, but kept as doc
HARP_REGISTERS = {
    ('Patch', 90): ['angle, intensity'],   # wheel encoder
    ('Patch', 35): ['bitmask'],            # trigger pellet delivery
    ('Patch', 32): ['bitmask'],            # pellet detected by beam break
    ('VideoController', 68): ['pwm_mask'], # camera trigger times (top and side)
    ('PositionTracking', 200): ['x', 'y', 'angle', 'major', 'minor', 'area'],
    ('WeightScale', 200): ['value', 'stable']
}


# @todo this could read a config file in the exp root that specifies "read data"
#  function (as a lambda) for each datastream.
def gen_data_dict(exp_root):
    """
    Generates a data dictionary, linking the names of types of data to their respective
    raw data files in a given path.

    Inputs
    ------
    exp_root : str OR Path
        The path to the experiment data root directory

    Outputs
    -------
    data_dict : DotMap
        The data dictionary
    """
    data_dict = DotMap()
    # Get all device names
    return data_dict


def load(exp_root, datastream, start_ts=None, end_ts=None, spec_ts=None, ts_tol=None):
    """
    Loads one type of data, which can be across multiple files of the same type
    (as determined by a time filter), into a pandas DataFrame.

    Inputs
    ------
    exp_root : str OR Path
        The path to the experiment data root directory
    datastream : three-element array-like
        1) the name of the type of data to return [str] (e.g. video, audio,
        wheel encoder, etc.); 2) the file extension [str] (e.g. bin, csv, avi,
        etc.); 3) the data reader function [function]
    start_ts : Timestamp (optional)
        The left bound of the time range of the data to load
    end_ts : Timestamp (optional)
        The right bound of the time range of the data to load
    spec_ts : Timestamp OR array-like of Timestamps (optional)
        A single or set of specific timestamps of the data to load
    ts_tol : Timedelta (optional)
        The +/- timestamp tolerance allowed for `spec_ts`

    Outputs
    -------
    data : DataFrame
        The loaded data
    """
    # Procedure:
    # 1) Using glob, get _all_ files for the specified datastream within `exp_root`.
    # 2) Filter files according to timestamps.
    # 3) Using the associated datastream read fn, load all files together into a DF.

    # <s Ensure list of paths, and for each path get files for the specified datastream.
    ipdb.set_trace()
    exp_roots = [exp_root] if not isinstance(exp_root, list) else exp_root
    files = set()
    ds_name, ext, read = datastream[0], datastream[1], datastream[2]
    for p in exp_roots:  # for each path, search all epochs + devices
        files.update(Path(p).glob(f"**/**/{ds_name}*.{ext}"))
    files = sorted(files)  # _all_ files that correspond to `ds_name` in `exp_roots`
    if len(files) == 0:  # warning and return if no files found
        warn(f"No files found matching: {exp_roots}/**/**/{ds_name}*.{ext}")
        return
    # /s>
    # <s Filter files by time, read all into a single df, and rm irrelevant timestamps.
    filetimes = get_chunktime(file=files)
    # If given `spec_ts`, find nearest files.
    if spec_ts is not None:
        spec_ts = pd.Series(spec_ts)  # ensure Series.
        # Find intersection b/w chunk times and all filetimes.
        spec_ts_chunks = get_chunktime(ts=spec_ts).unique()
        file_mask = filetimes.isin(spec_ts_chunks).values
        files = np.asarray(files)[file_mask]
        # Read files into a single df and rm irrelevant timestamps.
        data = pd.concat([read(f) for f in files])
        data = data.reindex(spec_ts, method='ffill', tolerance=ts_tol)
    # If given `start_ts` and/or `end_ts`, use these as cutoffs.
    elif (start_ts is not None) or (end_ts is not None):
        # Find files within `start_ts` and `end_ts`.
        file_mask = np.logical_and(filetimes > start_ts, filetimes < end_ts)
        files = np.asarray(files)[file_mask]
        # Read files into a single df and rm irrelevant timestamps.
        data = pd.concat([read(f) for f in files])
        data_mask = np.logical_and(data.index > start_ts, data.index < end_ts)
        data = data.loc[data_mask]
    # /s>
    return data


def get_chunktime(file=None, ts=None, start_ts=None, end_ts=None,
                  chunk_dur=1):
    """
    Returns acquisition chunk timestamps for given measurement timestamps.

    Inputs
    ------
    file : str OR Path or array-like of str or path (optional: required if `ts`,
        `start_ts` & `end_ts` not provided)
        Datafile(s) from which to extract acquisition chunk timestamp(s)
    ts : datetime OR Timestamp OR array-like of datetimes or Timestamps (optional:
        required if `file`, `start_ts` & `end_ts` not provided)
        Specific measurement timestamp(s) for which to return chunk timestamp(s)
    start_ts : datetime OR Timestamp (optional: required if `file` & `ts` not provided)
        The left bound of the time range of chunk timestamp(s) to return
    end_ts : datetime OR Timestamp (optional: required if `file` & `ts` not provided)
        The right bound of the time range of chunk timestamp(s) to return
    chunk_dur : int
        The chunk duration, in whole hours.

    Outputs
    -------
    Series of Timestamps
        The whole-hour acquisition chunk timestamps.
    """
    # If `file` given, extract chunk timestamps.
    if file is not None:
        files = np.asarray(file)  # ensure array
        chunk_ts = len(files) * [None]
        for i, f in enumerate(files):
            f = Path(f)  # ensure Path
            date_str, time_str = f.stem.split("_")[-1].split("T")
            time_str = time_str.replace("-", ":")
            chunk_ts[i] = pd.Timestamp(f"{date_str} {time_str}")
        return pd.Series(chunk_ts)
    # If `start_ts` and `end_ts` given, then call recursively and return date range
    # between start and end chunk ts.
    if (start_ts is not None) and (end_ts is not None):
        return pd.date_range(get_chunktime(ts=start_ts)[0], get_chunktime(ts=end_ts)[0],
                             freq=pd.DateOffset(hours=chunk_dur))
    # Convert timestamps to Series and return chunk timestamps.
    ts = pd.Series(ts)
    chunk_hour = chunk_dur * (ts.dt.hour // chunk_dur)
    return pd.to_datetime(ts.dt.date) + pd.to_timedelta(chunk_hour, 'h')


def read_harp(file, cols=None, _harp_ptypes=_HARP_T_PTYPES, _harp_res=_HARP_RES):
    """
    Reads timestamped data from a Harp file into a dataframe according to the Harp
    binary protocol.

    Inputs
    ------
    file : str OR Path
        The Harp binary file.
    cols : array-like of str (optional)
        The column names to use in the returned dataframe.

    Outputs
    -------
    df : DataFrame
        The loaded Harp data.

    Notes
    -----
    See https://github.com/harp-tech/protocol for info on the Harp binary protocol.

    Each Harp message consists of 7 entities, listed as [<entity> (<n_bytes>)]:
    [MessageType (1)] [MessageLength (1)] [Address (1)] [Port (1)] [PayloadType (1)]
    [Payload (PayloadType)] [Checksum (1)]
    """
    data = np.fromfile(file, dtype=np.uint8)  # read raw uint8-bit data
    if len(data) == 0:  # if empty file, return empty dataframe
        return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([]))
    # The number of bytes of each message (+2 for the byte corresponding to
    # MessageType and the byte corresponding to MessageLength itself).
    msg_length = data[1] + 2
    # The number of messages in this file.
    n_msgs = len(data) // msg_length
    # The payload length (-12 to discount non-payload bytes: 6 for the other 6
    # entities, and 6 b/c the first 6 bytes of the payload contain the timestamp)
    p_len = msg_length - 12
    # If `p_t` = PayloadType code of Timestamped Data; `p_n` = PayloadType code of
    # non-Timestamped data, and `h_b` = harp bitmask, then: `p_n = p_t & ~h_b`
    p_type = _harp_ptypes[data[4]]
    # The number of bytes that make up one value in the payload.
    p_elem_size = int(p_type[-1])
    # The number of items per payload (i.e. cols in the returned dataframe)
    n_p_items = p_len // p_elem_size
    # Load the payload into arrays (the relevant device data, the timestamp
    # seconds, and the timestamp microseconds/32). Offsets are for the number of
    # bytes of the preceding entities in each Harp message.
    p_data = np.ndarray(shape=(n_msgs, n_p_items), dtype=p_type, buffer=data,
                         offset=11, strides=(msg_length, p_elem_size))
    p_s = np.ndarray(shape=n_msgs, dtype='<u4', buffer=data, offset=5,
                     strides=msg_length)
    p_us_div_32 = np.ndarray(shape=n_msgs, dtype='<u2', buffer=data, offset=9,
                             strides=msg_length)
    ts = get_aeontime(p_s + (p_us_div_32 * _harp_res))  # timestamps
    ts.name = 'time'
    return pd.DataFrame(p_data, index=ts, columns=cols)


def get_aeontime(seconds):
    """Converts a Harp timestamp, in seconds, to a pandas Series of Timestamps."""
    return pd.Series(datetime(1904, 1, 1) + pd.to_timedelta(seconds, 's'))


def read_csv():
    pass
