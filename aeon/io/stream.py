import pandas as _pd
import aeon.io.reader as _reader
from enum import Enum as _Enum


class Area(_Enum):
    Null = 0
    Nest = 1
    Corridor = 2
    Arena = 3
    Patch1 = 4
    Patch2 = 5

class _RegionReader(_reader.Harp):
    def __init__(self, name):
        super().__init__(name, columns=['region'])

    def read(self, file):
        data = super().read(file)
        categorical = _pd.Categorical(data.region, categories=range(len(Area._member_names_)))
        data['region'] = categorical.rename_categories(Area._member_names_)
        return data

def video(name):
    """Video frame metadata."""
    return { "Video": _reader.Video(name) }

def position(name):
    """Position tracking data for the specified camera."""
    return { "Position": _reader.Position(f"{name}_200") }

def region(name):
    """Region tracking data for the specified camera."""
    return { "Region": _RegionReader(f"{name}_201") }

def depletionFunction(name):
    """State of the linear depletion function for foraging patches."""
    return { "DepletionState": _reader.PatchState(f"{name}_State") }

def encoder(name):
    """Wheel magnetic encoder data."""
    return { "Encoder": _reader.Encoder(f"{name}_90") }

def feeder(name):
    """Feeder commands and events."""
    return {
        "BeamBreak": _reader.Event(f"{name}_32", 0x22, 'PelletDetected'),
        "DeliverPellet": _reader.Event(f"{name}_35", 0x80, 'TriggerPellet')
    }

def patch(name):
    """Data streams for a patch."""
    return compositeStream(name, depletionFunction, encoder, feeder)

def weight(name):
    """Weight measurement data streams for a specific nest."""
    return {
        "WeightRaw": _reader.Weight(f"{name}_200"),
        "WeightFiltered": _reader.Weight(f"{name}_202"),
        "WeightSubject": _reader.Weight(f"{name}_204")
    }

def environment(name):
    """Metadata for environment mode and subjects."""
    return {
        "EnvironmentState": _reader.Csv(f"{name}_EnvironmentState", ['state']),
        "SubjectState": _reader.Subject(f"{name}_SubjectState")
    }

def messageLog(name):
    """Message log data."""
    return { "MessageLog": _reader.Log(f"{name}_MessageLog") }

def metadata(name):
    """Metadata for acquisition epochs."""
    return { name: _reader.Metadata(name) }

def session(name):
    """Session metadata for Experiment 0.1."""
    return { name: _reader.Csv(f"{name}_2", columns=['id','weight','event']) }


class Device:
    """
    Groups multiple data streams into a logical device.

    If a device contains a single stream with the same name as the device
    `name`, it will be considered a singleton, and the stream reader will be
    paired directly with the device without nesting.

    Attributes
    ----------
    name : str
        Name of the device.
    args : Any
        Data streams collected from the device.
    """
    def __init__(self, name, *args):
        self.name = name
        self.schema = compositeStream(name, *args)

    def __iter__(self):
        if len(self.schema) == 1:
            singleton = self.schema.get(self.name, None)
            if singleton:
                return iter((self.name, singleton))
        return iter((self.name, self.schema))


def compositeStream(name, *args):
    """Merges multiple data streams into one stream."""
    schema = {}
    if args:
        for stream in args:
            schema.update(stream(name))
    return schema