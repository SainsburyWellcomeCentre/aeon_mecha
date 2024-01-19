from enum import Enum as _Enum

import pandas as pd

import aeon.io.device as _device
import aeon.io.reader as _reader
import aeon.schema.core as _stream


class Area(_Enum):
    Null = 0
    Nest = 1
    Corridor = 2
    Arena = 3
    Patch1 = 4
    Patch2 = 5


class _RegionReader(_reader.Harp):
    def __init__(self, pattern):
        super().__init__(pattern, columns=["region"])

    def read(self, file):
        data = super().read(file)
        categorical = pd.Categorical(data.region, categories=range(len(Area._member_names_)))
        data["region"] = categorical.rename_categories(Area._member_names_)
        return data


class _PatchState(_reader.Csv):
    """Extracts patch state data for linear depletion foraging patches.

    Columns:
        threshold (float): Distance to travel before the next pellet is delivered.
        d1 (float): y-intercept of the line specifying the depletion function.
        delta (float): Slope of the linear depletion function.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["threshold", "d1", "delta"])


class _Weight(_reader.Harp):
    """Extract weight measurements from an electronic weighing device.

    Columns:
        value (float): Absolute weight reading, in grams.
        stable (float): Normalized value in the range [0, 1]
            indicating how much the reading is stable.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["value", "stable"])


def region(pattern):
    """Region tracking data for the specified camera."""
    return {"Region": _RegionReader(f"{pattern}_201_*")}


def depletion_function(pattern):
    """State of the linear depletion function for foraging patches."""
    return {"DepletionState": _PatchState(f"{pattern}_State_*")}


def feeder(pattern):
    """Feeder commands and events."""
    return _device.register(pattern, beam_break, deliver_pellet)


def beam_break(pattern):
    """Beam break events for pellet detection."""
    return {"BeamBreak": _reader.BitmaskEvent(f"{pattern}_32_*", 0x22, "PelletDetected")}


def deliver_pellet(pattern):
    """Pellet delivery commands."""
    return {"DeliverPellet": _reader.BitmaskEvent(f"{pattern}_35_*", 0x80, "TriggerPellet")}


def pellet_manual_delivery(pattern):
    """Manual pellet delivery."""
    return {"ManualDelivery": _reader.Harp(f"{pattern}_*", ["manual_delivery"])}


def missed_pellet(pattern):
    """Missed pellet delivery."""
    return {"MissedPellet": _reader.Harp(f"{pattern}_*", ["missed_pellet"])}


def pellet_retried_delivery(pattern):
    """Retry pellet delivery."""
    return {"RetriedDelivery": _reader.Harp(f"{pattern}_*", ["retried_delivery"])}


def pellet_depletion_state(pattern):
    """Pellet delivery state."""
    return {"DepletionState": _reader.Csv(f"{pattern}_*", ["threshold", "offset", "rate"])}


def patch(pattern):
    """Data streams for a patch."""
    return _device.register(pattern, depletion_function, _stream.encoder, feeder)


def weight(pattern):
    """Weight measurement data streams for a specific nest."""
    return _device.register(pattern, weight_raw, weight_filtered, weight_subject)


def weight_raw(pattern):
    """Raw weight measurement for a specific nest."""
    return {"WeightRaw": _Weight(f"{pattern}_200_*")}


def weight_filtered(pattern):
    """Filtered weight measurement for a specific nest."""
    return {"WeightFiltered": _Weight(f"{pattern}_202_*")}


def weight_subject(pattern):
    """Subject weight measurement for a specific nest."""
    return {"WeightSubject": _Weight(f"{pattern}_204_*")}


def session(pattern):
    """Session metadata for Experiment 0.1."""
    return {pattern: _reader.Csv(f"{pattern}_2*", columns=["id", "weight", "event"])}
