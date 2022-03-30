import os
import math
import numpy as np
import pandas as pd
from aeon.io.api import aeon, chunk_key, load

_SECONDS_PER_TICK = 32e-6

"""Maps Harp payload types to numpy data type objects."""
payloadtypes = {
    1 : np.dtype(np.uint8),
    2 : np.dtype(np.uint16),
    4 : np.dtype(np.uint32),
    8 : np.dtype(np.uint64),
    129 : np.dtype(np.int8),
    130 : np.dtype(np.int16),
    132 : np.dtype(np.int32),
    136 : np.dtype(np.int64),
    68 : np.dtype(np.float32)
}

class Reader:
    def __init__(self, name, extension, read):
        self.name = name
        self.extension = extension
        self.read = read

class HarpReader:
    """
    Reads data from raw binary files encoded using the Harp protocol.

    Parameters
    ----------
    name : str
        Name of the Harp data stream.
    columns : str or array-like, optional
        The column labels to use for the data.
    extension : str, optional
        Extension for data files.
    """
    def __init__(self, name, columns=None, extension="bin"):
        self.name = name
        self.extension = extension
        self.columns = columns

    def read(self, file):
        """
        Reads data from the specified Harp binary file.
        
        Parameters
        ----------
        file : str
            Path to a Harp binary file.
        
        Returns
        -------
        DataFrame
            Pandas data frame containing Harp event data, sorted by time.
        """
        if file is None:
            return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

        data = np.fromfile(file, dtype=np.uint8)
        if len(data) == 0:
            return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

        stride = data[1] + 2
        length = len(data) // stride
        payloadsize = stride - 12
        payloadtype = payloadtypes[data[4] & ~0x10]
        elementsize = payloadtype.itemsize
        payloadshape = (length, payloadsize // elementsize)
        seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
        ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
        seconds = ticks * _SECONDS_PER_TICK + seconds
        payload = np.ndarray(
            payloadshape,
            dtype=payloadtype,
            buffer=data, offset=11,
            strides=(stride, elementsize))
        time = aeon(seconds)
        time.name = 'time'

        if payloadshape[1] < len(self.columns):
            data = pd.DataFrame(payload, index=time, columns=self.columns[:payloadshape[1]])
            data[self.columns[payloadshape[1]:]] = math.nan
            return data
        else:
            return pd.DataFrame(payload, index=time, columns=self.columns)

class ChunkReader:
    def __init__(self, name, extension):
        self.name = name
        self.extension = extension
        self.columns = ['time', 'path']
    
    def read(self, file):
        if file is None:
            return pd.DataFrame(columns=self.columns[1:], index=pd.DatetimeIndex([]))
        chunk = chunk_key(file)
        data = pd.DataFrame({ self.columns[0]: [chunk], self.columns[1]: [file] })
        data.set_index('time', inplace=True)
        return data

class CsvReader:
    def __init__(self, name, columns, extension="csv"):
        self.name = name
        self.extension = extension
        self.columns = ['time', *columns]

    def read(self, file):
        if file is None:
            return pd.DataFrame(columns=self.columns[1:], index=pd.DatetimeIndex([]))
        data = pd.read_csv(file, header=0, names=self.columns)
        data['time'] = aeon(data['time'])
        data.set_index('time', inplace=True)
        return data

class SubjectReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['id', 'weight', 'event'])

class LogReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['priority', 'type', 'message'])

class VideoReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['hw_counter', 'hw_timestamp'])

class PatchStateReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['threshold', 'd1', 'delta'])

class EncoderReader(HarpReader):
    def __init__(self, name):
        super().__init__(name, columns=['angle', 'intensity'])

class WeightReader(HarpReader):
    def __init__(self, name):
        super().__init__(name, columns=['value', 'stable'])

class PositionReader(HarpReader):
    def __init__(self, name):
        super().__init__(name, columns=['x', 'y', 'angle', 'major', 'minor', 'area', 'id'])

class EventReader(HarpReader):
    def __init__(self, name, value, tag):
        super().__init__(name, columns=['event'])
        self.value = value
        self.tag = tag

    def read(self, file):
        data = super().read(self, file)
        data = data[data.event == self.value]
        data['event'] = self.tag
        return data

class VideoReader:
    def __init__(self, name):
        self.name = name
        self.extension = "csv"
        self.columns = ['time', 'hw_counter', 'hw_timestamp']

    def read(self, file):
        """Reads video metadata from the specified file."""
        if file is None:
            return pd.DataFrame(columns=['frame', *self.columns[1:]], index=pd.DatetimeIndex([]))
        data = pd.read_csv(file, header=0, names=self.columns)
        data.insert(loc=1, column='frame', value=data.index)
        data['time'] = aeon(data['time'])
        data['path'] = os.path.splitext(file)[0] + '.avi'
        data['epoch'] = file.rsplit(os.sep, maxsplit=3)[1]
        data.set_index('time', inplace=True)
        return data
