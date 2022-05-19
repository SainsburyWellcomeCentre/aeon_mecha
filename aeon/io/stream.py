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
    def __init__(self, pattern):
        super().__init__(pattern, columns=['region'])

    def read(self, file):
        data = super().read(file)
        categorical = _pd.Categorical(data.region, categories=range(len(Area._member_names_)))
        data['region'] = categorical.rename_categories(Area._member_names_)
        return data

def video(pattern):
    """Video frame metadata."""
    return { "Video": _reader.Video(pattern) }

def position(pattern):
    """Position tracking data for the specified camera."""
    return { "Position": _reader.Position(f"{pattern}_200") }

def region(pattern):
    """Region tracking data for the specified camera."""
    return { "Region": _RegionReader(f"{pattern}_201") }

def depletionFunction(pattern):
    """State of the linear depletion function for foraging patches."""
    return { "DepletionState": _reader.PatchState(f"{pattern}_State") }

def encoder(pattern):
    """Wheel magnetic encoder data."""
    return { "Encoder": _reader.Encoder(f"{pattern}_90") }

def feeder(pattern):
    """Feeder commands and events."""
    return _device.compositeStream(pattern, beam_break, deliver_pellet)

def beam_break(pattern):
    """Beam break events for pellet detection."""
    return { "BeamBreak": _reader.BitmaskEvent(f"{pattern}_32", 0x22, 'PelletDetected') }

def deliver_pellet(pattern):
    """Pellet delivery commands."""
    return { "DeliverPellet": _reader.BitmaskEvent(f"{pattern}_35", 0x80, 'TriggerPellet') }

def patch(pattern):
    """Data streams for a patch."""
    return _device.compositeStream(pattern, depletionFunction, encoder, feeder)

def weight(pattern):
    """Weight measurement data streams for a specific nest."""
    return _device.compositeStream(pattern, weight_raw, weight_filtered, weight_subject)

def weight_raw(pattern):
    """Raw weight measurement for a specific nest."""
    return { "WeightRaw": _reader.Weight(f"{pattern}_200") }

def weight_filtered(pattern):
    """Filtered weight measurement for a specific nest."""
    return { "WeightFiltered": _reader.Weight(f"{pattern}_202") }

def weight_subject(pattern):
    """Subject weight measurement for a specific nest."""
    return { "WeightSubject": _reader.Weight(f"{pattern}_204") }

def environment(pattern):
    """Metadata for environment mode and subjects."""
    return _device.compositeStream(pattern, environment_state, subject_state)

def environment_state(pattern):
    """Environment state log."""
    return { "EnvironmentState": _reader.Csv(f"{pattern}_EnvironmentState", ['state']) }

def subject_state(pattern):
    """Subject state log."""
    return { "SubjectState": _reader.Subject(f"{pattern}_SubjectState") }

def messageLog(pattern):
    """Message log data."""
    return { "MessageLog": _reader.Log(f"{pattern}_MessageLog") }

def metadata(pattern):
    """Metadata for acquisition epochs."""
    return { pattern: _reader.Metadata(pattern) }

def session(pattern):
    """Session metadata for Experiment 0.1."""
    return { pattern: _reader.Csv(f"{pattern}_2", columns=['id','weight','event']) }
