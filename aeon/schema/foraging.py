"""Schema definition for foraging experiments."""

from enum import Enum

import pandas as pd

import aeon.io.reader as _reader
import aeon.schema.core as _stream
from aeon.schema.streams import Stream, StreamGroup


class Area(Enum):
    Null = 0
    Nest = 1
    Corridor = 2
    Arena = 3
    Patch1 = 4
    Patch2 = 5


class _RegionReader(_reader.Harp):
    def __init__(self, pattern):
        """Initializes the RegionReader class."""
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
        """Initializes the PatchState class."""
        super().__init__(pattern, columns=["threshold", "d1", "delta"])


class _Weight(_reader.Harp):
    """Extract weight measurements from an electronic weighing device.

    Columns:
        value (float): Absolute weight reading, in grams.
        stable (float): Normalized value in the range [0, 1]
            indicating how much the reading is stable.
    """

    def __init__(self, pattern):
        """Initializes  the Weight class."""
        super().__init__(pattern, columns=["value", "stable"])


class Region(Stream):
    """Region tracking data for the specified camera."""

    def __init__(self, pattern):
        """Initializes the Region stream."""
        super().__init__(_RegionReader(f"{pattern}_201_*"))


class DepletionFunction(Stream):
    """State of the linear depletion function for foraging patches."""

    def __init__(self, pattern):
        """Initializes the DepletionFunction stream."""
        super().__init__(_PatchState(f"{pattern}_State_*"))


class Feeder(StreamGroup):
    """Feeder commands and events."""

    def __init__(self, pattern):
        """Initializes the Feeder stream group."""
        super().__init__(pattern, BeamBreak, DeliverPellet)


class BeamBreak(Stream):
    """Beam break events for pellet detection."""

    def __init__(self, pattern):
        """Initializes the BeamBreak stream."""
        super().__init__(_reader.BitmaskEvent(f"{pattern}_32_*", 0x22, "PelletDetected"))


class DeliverPellet(Stream):
    """Pellet delivery commands."""

    def __init__(self, pattern):
        """Initializes the DeliverPellet stream."""
        super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x01, "TriggerPellet"))


class Patch(StreamGroup):
    """Data streams for a patch."""

    def __init__(self, pattern):
        """Initializes the Patch stream group."""
        super().__init__(pattern, DepletionFunction, _stream.Encoder, Feeder)


class Weight(StreamGroup):
    """Weight measurement data streams for a specific nest."""

    def __init__(self, pattern):
        """Initializes the Weight stream group."""
        super().__init__(pattern, WeightRaw, WeightFiltered, WeightSubject)


class WeightRaw(Stream):
    """Raw weight measurement for a specific nest."""

    def __init__(self, pattern):
        """Initializes the WeightRaw stream."""
        super().__init__(_Weight(f"{pattern}_200_*"))


class WeightFiltered(Stream):
    """Filtered weight measurement for a specific nest."""

    def __init__(self, pattern):
        """Initializes the WeightFiltered stream."""
        super().__init__(_Weight(f"{pattern}_202_*"))


class WeightSubject(Stream):
    """Subject weight measurement for a specific nest."""

    def __init__(self, pattern):
        """Initializes the WeightSubject stream."""
        super().__init__(_Weight(f"{pattern}_204_*"))


class SessionData(Stream):
    """Session metadata for Experiment 0.1."""

    def __init__(self, pattern):
        """Initializes the SessionData stream."""
        super().__init__(_reader.Csv(f"{pattern}_2*", columns=["id", "weight", "event"]))
