"""This module defines the schema for the social_02 dataset."""

import aeon.io.reader as _reader
from aeon.schema import core, foraging
from aeon.schema.streams import Stream, StreamGroup


class Environment(StreamGroup):
    def __init__(self, path):
        """Initializes the Environment stream group."""
        super().__init__(path)

    EnvironmentState = core.EnvironmentState

    class BlockState(Stream):
        def __init__(self, path):
            """Initializes the BlockState stream."""
            super().__init__(
                _reader.Csv(
                    f"{path}_BlockState_*",
                    columns=["pellet_ct", "pellet_ct_thresh", "due_time"],
                )
            )

    class LightEvents(Stream):
        def __init__(self, path):
            """Initializes the LightEvents stream."""
            super().__init__(_reader.Csv(f"{path}_LightEvents_*", columns=["channel", "value"]))

    MessageLog = core.MessageLog


class SubjectData(StreamGroup):
    def __init__(self, path):
        """Initializes the SubjectData stream group."""
        super().__init__(path)

    class SubjectState(Stream):
        def __init__(self, path):
            """Initializes the SubjectState stream."""
            super().__init__(_reader.Csv(f"{path}_SubjectState_*", columns=["id", "weight", "type"]))

    class SubjectVisits(Stream):
        def __init__(self, path):
            """Initializes the SubjectVisits stream."""
            super().__init__(_reader.Csv(f"{path}_SubjectVisits_*", columns=["id", "type", "region"]))

    class SubjectWeight(Stream):
        def __init__(self, path):
            """Initializes the SubjectWeight stream."""
            super().__init__(
                _reader.Csv(
                    f"{path}_SubjectWeight_*",
                    columns=["weight", "confidence", "subject_id", "int_id"],
                )
            )


class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        super().__init__(_reader.Pose(f"{path}_test-node1*"))


class WeightRaw(Stream):
    def __init__(self, path):
        """Initialize the WeightRaw stream."""
        super().__init__(_reader.Harp(f"{path}_200_*", ["weight(g)", "stability"]))


class WeightFiltered(Stream):
    def __init__(self, path):
        """Initializes the WeightFiltered stream."""
        super().__init__(_reader.Harp(f"{path}_202_*", ["weight(g)", "stability"]))


class Patch(StreamGroup):
    def __init__(self, path):
        """Initializes the Patch stream group."""
        super().__init__(path)

    class DepletionState(Stream):
        def __init__(self, path):
            """Initializes the DepletionState stream."""
            super().__init__(_reader.Csv(f"{path}_State_*", columns=["threshold", "offset", "rate"]))

    Encoder = core.Encoder

    Feeder = foraging.Feeder

    class ManualDelivery(Stream):
        def __init__(self, path):
            """Initializes the ManualDelivery stream."""
            super().__init__(_reader.Harp(f"{path}_201_*", ["manual_delivery"]))

    class MissedPellet(Stream):
        def __init__(self, path):
            """Initializes the MissedPellet stream."""
            super().__init__(_reader.Harp(f"{path}_202_*", ["missed_pellet"]))

    class RetriedDelivery(Stream):
        def __init__(self, path):
            """Initializes the RetriedDelivery stream."""
            super().__init__(_reader.Harp(f"{path}_203_*", ["retried_delivery"]))


class RfidEvents(Stream):
    def __init__(self, path):
        """Initializes the RfidEvents stream."""
        super().__init__(_reader.Harp(f"{path}_32*", ["rfid"]))
