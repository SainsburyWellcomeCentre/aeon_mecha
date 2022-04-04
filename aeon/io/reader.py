import os
import math
import numpy as np
import pandas as pd
from aeon.io.api import chunk_key

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
    def __init__(self, name, columns, extension):
        self.name = name
        self.device = name.split('_')[0]
        self.columns = columns
        self.extension = extension

    def read(self, _):
        return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

class HarpReader(Reader):
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
    def __init__(self, name, columns, extension="bin"):
        super().__init__(name, columns, extension)

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

        if payloadshape[1] < len(self.columns):
            data = pd.DataFrame(payload, index=seconds, columns=self.columns[:payloadshape[1]])
            data[self.columns[payloadshape[1]:]] = math.nan
            return data
        else:
            return pd.DataFrame(payload, index=seconds, columns=self.columns)

class ChunkReader(Reader):
    def __init__(self, name, extension):
        super().__init__(name, columns=['path'], extension=extension)
    
    def read(self, file):
        chunk = chunk_key(file)
        return pd.DataFrame(file, index=[chunk], columns=self.columns)

class CsvReader(Reader):
    def __init__(self, name, columns, extension="csv"):
        super().__init__(name, columns, extension)

    def read(self, file):
        return pd.read_csv(file, header=0, names=self.columns, index_col=0)

class SubjectReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['id', 'weight', 'event'])

class LogReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['priority', 'type', 'message'])

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

class VideoReader(CsvReader):
    def __init__(self, name):
        super().__init__(name, columns=['frame', 'hw_counter', 'hw_timestamp', 'path', 'epoch'])
        self._rawcolumns = ['time'] + self.columns[1:3]

    def read(self, file):
        """Reads video metadata from the specified file."""
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data.insert(loc=1, column=self.columns[0], value=data.index)
        data['path'] = os.path.splitext(file)[0] + '.avi'
        data['epoch'] = file.parts[-3]
        data.set_index('time', inplace=True)
        return data
