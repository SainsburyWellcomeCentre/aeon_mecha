import pandas as _pd
import aeon.io.reader as _reader
import aeon.io.device as _device
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
    return name, { "Video": _reader.Video(name) }

def position(name):
    """Position tracking data for the specified camera."""
    return name, { "Position": _reader.Position(f"{name}_200") }

def region(name):
    """Region tracking data for the specified camera."""
    return name, { "Region": _RegionReader(f"{name}_201") }

def depletionFunction(name):
    """State of the linear depletion function for foraging patches."""
    return name, { "DepletionState": _reader.PatchState(f"{name}_State") }

def encoder(name):
    """Wheel magnetic encoder data."""
    return name, { "Encoder": _reader.Encoder(f"{name}_90") }

def feeder(name):
    """Feeder commands and events."""
    return name, {
        "BeamBreak": _reader.Event(f"{name}_32", 0x20, 'PelletDetected'),
        "DeliverPellet": _reader.Event(f"{name}_35", 0x80, 'TriggerPellet')
    }

def patch(name):
    """Data streams for a patch."""
    return _device.compositeStream(name, depletionFunction, encoder, feeder)

def weight(name):
    """Weight measurement data streams for a specific nest."""
    return name, {
        "WeightRaw": _reader.Weight(f"{name}_200"),
        "WeightFiltered": _reader.Weight(f"{name}_202"),
        "WeightSubject": _reader.Weight(f"{name}_204")
    }

def metadata(name):
    """Metadata for environment mode and subjects."""
    return name, {
        "EnvironmentState": _reader.Csv(f"{name}_EnvironmentState", ['state']),
        "SubjectState": _reader.Subject(f"{name}_SubjectState")
    }

def messageLog(name):
    """Message log data."""
    return name, { "MessageLog": _reader.Log(f"{name}_MessageLog") }
