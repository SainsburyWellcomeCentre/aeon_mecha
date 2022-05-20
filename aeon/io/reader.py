import os
import math
import json
import datetime
import numpy as np
import pandas as pd
from aeon.io.api import chunk_key
from dotmap import DotMap

_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
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
    """Extracts data from raw chunk files in an Aeon dataset.
    
    Attributes:
        pattern (str): Pattern used to find raw chunk files,
            usually in the format `<Device>_<DataStream>`.
        columns (str or array-like): Column labels to use for the data.
        extension (str): Extension of data file pathnames.
    """
    def __init__(self, pattern, columns, extension):
        self.pattern = pattern
        self.columns = columns
        self.extension = extension

    def read(self, _):
        """Reads data from the specified chunk file."""
        return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

class Harp(Reader):
    """Extracts data from raw binary files encoded using the Harp protocol."""
    def __init__(self, pattern, columns, extension="bin"):
        super().__init__(pattern, columns, extension)

    def read(self, file):
        """Reads data from the specified Harp binary file."""
        data = np.fromfile(file, dtype=np.uint8)
        if len(data) == 0:
            return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

        stride = data[1] + 2
        length = len(data) // stride
        payloadsize = stride - 12
        payloadtype = _payloadtypes[data[4] & ~0x10]
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

class Chunk(Reader):
    """Extracts path and epoch information from chunk files in the dataset."""
    def __init__(self, reader=None, pattern=None, extension=None):
        if isinstance(reader, Reader):
            pattern = reader.pattern
            extension = reader.extension
        super().__init__(pattern, columns=['path', 'epoch'], extension=extension)
    
    def read(self, file):
        """Returns path and epoch information for the specified chunk."""
        epoch, chunk = chunk_key(file)
        data = { 'path': file, 'epoch': epoch }
        return pd.DataFrame(data, index=[chunk], columns=self.columns)

class Metadata(Reader):
    """Extracts metadata information from all epochs in the dataset."""
    def __init__(self, pattern="Metadata"):
        super().__init__(pattern, columns=['workflow', 'commit', 'metadata'], extension="yml")

    def read(self, file):
        """Returns metadata for the specified epoch."""
        epoch_str = file.parts[-2]
        date_str, time_str = epoch_str.split("T")
        time = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        with open(file) as fp:
            metadata = json.load(fp)
        workflow = metadata.pop('Workflow')
        commit = metadata.pop('Commit', pd.NA)
        data = { 'workflow': workflow, 'commit': commit, 'metadata': [DotMap(metadata)] }
        return pd.DataFrame(data, index=[time], columns=self.columns)

class Csv(Reader):
    """
    Extracts data from comma-separated (csv) text files, where the first column
    stores the Aeon timestamp, in seconds.
    """
    def __init__(self, pattern, columns, dtype=None, extension="csv"):
        super().__init__(pattern, columns, extension)
        self.dtype = dtype

    def read(self, file):
        """Reads data from the specified CSV text file."""
        return pd.read_csv(
            file,
            header=0,
            names=self.columns,
            dtype=self.dtype,
            index_col=0)

class Subject(Csv):
    """
    Extracts metadata for subjects entering and exiting the environment.

    Columns:
        id (str): Unique identifier of a subject in the environment.
        weight (float): Weight measurement of the subject on entering
            or exiting the environment.
        event (str): Event type. Can be one of `Enter`, `Exit` or `Remain`.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['id', 'weight', 'event'])

class Log(Csv):
    """
    Extracts message log data.
    
    Columns:
        priority (str): Priority level of the message.
        type (str): Type of the log message.
        message (str): Log message data. Can be structured using tab
            separated values.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['priority', 'type', 'message'])

class PatchState(Csv):
    """
    Extracts patch state data for linear depletion foraging patches.
    
    Columns:
        threshold (float): Distance to travel before the next pellet is delivered.
        d1 (float): y-intercept of the line specifying the depletion function.
        delta (float): Slope of the linear depletion function.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['threshold', 'd1', 'delta'])

class Encoder(Harp):
    """
    Extract magnetic encoder data.
    
    Columns:
        angle (float): Absolute angular position, in radians, of the magnetic encoder.
        intensity (float): Intensity of the magnetic field.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['angle', 'intensity'])

class Weight(Harp):
    """
    Extract weight measurements from an electronic weighing device.
    
    Columns:
        value (float): Absolute weight reading, in grams.
        stable (float): Normalized value in the range [0, 1]
            indicating how much the reading is stable.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['value', 'stable'])

class Position(Harp):
    """
    Extract 2D position tracking data for a specific camera.

    Columns:
        x (float): x-coordinate of the object center of mass.
        y (float): y-coordinate of the object center of mass.
        angle (float): angle, in radians, of the ellipse fit to the object.
        major (float): length, in pixels, of the major axis of the ellipse
            fit to the object.
        minor (float): length, in pixels, of the minor axis of the ellipse
            fit to the object.
        area (float): number of pixels in the object mass.
        id (float): unique tracking ID of the object in a frame.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['x', 'y', 'angle', 'major', 'minor', 'area', 'id'])

class BitmaskEvent(Harp):
    """
    Extracts event data matching a specific digital I/O bitmask.

    Columns:
        event (str): Unique identifier for the event code.
    """
    def __init__(self, pattern, value, tag):
        super().__init__(pattern, columns=['event'])
        self.value = value
        self.tag = tag

    def read(self, file):
        """
        Reads a specific event code from digital data and matches it to the
        specified unique identifier.
        """
        data = super().read(file)
        data = data[data.event == self.value]
        data['event'] = self.tag
        return data

class Video(Csv):
    """
    Extracts video frame metadata.
    
    Columns:
        hw_counter (int): Hardware frame counter value for the current frame.
        hw_timestamp (int): Internal camera timestamp for the current frame.
    """
    def __init__(self, pattern):
        super().__init__(pattern, columns=['hw_counter', 'hw_timestamp', '_frame', '_path', '_epoch'])
        self._rawcolumns = ['time'] + self.columns[0:2]

    def read(self, file):
        """Reads video metadata from the specified file."""
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data['_frame'] = data.index
        data['_path'] = os.path.splitext(file)[0] + '.avi'
        data['_epoch'] = file.parts[-3]
        data.set_index('time', inplace=True)
        return data

def from_dict(data, pattern=None):
    reader_type = data.get('type', None)
    if reader_type is not None:
        kwargs = {k:v for k,v in data.items() if k != 'type'}
        return globals()[reader_type](pattern=pattern, **kwargs)

    return DotMap({
        k:from_dict(v, f"{pattern}_{k}" if pattern is not None else k)
        for k,v in data.items()
    })

def to_dict(dotmap):
    if isinstance(dotmap, Reader):
        kwargs = { k:v for k,v in vars(dotmap).items()
                       if k not in ['pattern'] and not k.startswith('_') }
        kwargs['type'] = type(dotmap).__name__
        return kwargs
    return {
        k:to_dict(v) for k,v in dotmap.items()
    }