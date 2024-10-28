""" Octagon schema definition. """

import aeon.io.reader as _reader
from aeon.schema.streams import Stream, StreamGroup


class Photodiode(Stream):
    def __init__(self, path):
        """Initializes the Photodiode stream."""
        super().__init__(_reader.Harp(f"{path}_44_*", columns=["adc", "encoder"]))


class OSC(StreamGroup):
    def __init__(self, path):
        """Initializes the OSC stream group."""
        super().__init__(path)

    class BackgroundColor(Stream):
        def __init__(self, pattern):
            """Initializes the BackgroundColor stream."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_backgroundcolor_*",
                    columns=["typetag", "r", "g", "b", "a"],
                )
            )

    class ChangeSubjectState(Stream):
        def __init__(self, pattern):
            """Initializes the ChangeSubjectState stream."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_changesubjectstate_*",
                    columns=["typetag", "id", "weight", "event"],
                )
            )

    class EndTrial(Stream):
        def __init__(self, pattern):
            """Initialises the EndTrial stream."""
            super().__init__(
                _reader.Csv(f"{pattern}_endtrial_*", columns=["typetag", "value"])
            )

    class Slice(Stream):
        def __init__(self, pattern):
            """Initialises the Slice."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_octagonslice_*",
                    columns=["typetag", "wall_id", "r", "g", "b", "a", "delay"],
                )
            )

    class GratingsSlice(Stream):
        def __init__(self, pattern):
            """Initialises the GratingsSlice stream."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_octagongratingsslice_*",
                    columns=[
                        "typetag",
                        "wall_id",
                        "contrast",
                        "opacity",
                        "spatial_frequency",
                        "temporal_frequency",
                        "angle",
                        "delay",
                    ],
                )
            )

    class Poke(Stream):
        def __init__(self, pattern):
            """Initializes the Poke class."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_poke_*",
                    columns=[
                        "typetag",
                        "wall_id",
                        "poke_id",
                        "reward",
                        "reward_interval",
                        "delay",
                        "led_delay",
                    ],
                )
            )

    class Response(Stream):
        def __init__(self, pattern):
            """Initialises the Response class."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_response_*",
                    columns=["typetag", "wall_id", "poke_id", "response_time"],
                )
            )

    class RunPreTrialNoPoke(Stream):
        def __init__(self, pattern):
            """Initialises the RunPreTrialNoPoke class."""
            super().__init__(
                _reader.Csv(
                    f"{pattern}_run_pre_no_poke_*",
                    columns=[
                        "typetag",
                        "wait_for_poke",
                        "reward_iti",
                        "timeout_iti",
                        "pre_trial_duration",
                        "activity_reset_flag",
                    ],
                )
            )

    class StartNewSession(Stream):
        def __init__(self, pattern):
            """Initializes the StartNewSession class."""
            super().__init__(
                _reader.Csv(f"{pattern}_startnewsession_*", columns=["typetag", "path"])
            )


class TaskLogic(StreamGroup):
    def __init__(self, path):
        """Initialises the TaskLogic stream group."""
        super().__init__(path)

    class TrialInitiation(Stream):
        def __init__(self, pattern):
            """Initializes the TrialInitiation stream."""
            super().__init__(_reader.Harp(f"{pattern}_1_*", columns=["trial_type"]))

    class Response(Stream):
        def __init__(self, pattern):
            """Initializes the Response stream."""
            super().__init__(
                _reader.Harp(f"{pattern}_2_*", columns=["wall_id", "poke_id"])
            )

    class PreTrialState(Stream):
        def __init__(self, pattern):
            """Initializes the PreTrialState stream."""
            super().__init__(_reader.Harp(f"{pattern}_3_*", columns=["state"]))

    class InterTrialInterval(Stream):
        def __init__(self, pattern):
            """Initializes the InterTrialInterval stream."""
            super().__init__(_reader.Harp(f"{pattern}_4_*", columns=["state"]))

    class SliceOnset(Stream):
        def __init__(self, pattern):
            """Initializes the SliceOnset stream."""
            super().__init__(_reader.Harp(f"{pattern}_10_*", columns=["wall_id"]))

    class DrawBackground(Stream):
        def __init__(self, pattern):
            """Initializes the DrawBackground stream."""
            super().__init__(_reader.Harp(f"{pattern}_11_*", columns=["state"]))

    class GratingsSliceOnset(Stream):
        def __init__(self, pattern):
            """Initializes the GratingsSliceOnset stream."""
            super().__init__(_reader.Harp(f"{pattern}_12_*", columns=["wall_id"]))


class Wall(StreamGroup):
    def __init__(self, path):
        """Initialises the Wall stream group."""
        super().__init__(path)

    class BeamBreak0(Stream):
        def __init__(self, pattern):
            """Initialises the BeamBreak0 stream."""
            super().__init__(
                _reader.DigitalBitmask(f"{pattern}_32_*", 0x1, columns=["state"])
            )

    class BeamBreak1(Stream):
        def __init__(self, pattern):
            """Initialises the BeamBreak1 stream."""
            super().__init__(
                _reader.DigitalBitmask(f"{pattern}_32_*", 0x2, columns=["state"])
            )

    class BeamBreak2(Stream):
        def __init__(self, pattern):
            """Initialises the BeamBreak2 stream."""
            super().__init__(
                _reader.DigitalBitmask(f"{pattern}_32_*", 0x4, columns=["state"])
            )

    class SetLed0(Stream):
        def __init__(self, pattern):
            """Initialises the SetLed0 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x1, "Set"))

    class SetLed1(Stream):
        def __init__(self, pattern):
            """Initialises the SetLed1 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x2, "Set"))

    class SetLed2(Stream):
        def __init__(self, pattern):
            """Initialises the SetLed2 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x4, "Set"))

    class SetValve0(Stream):
        def __init__(self, pattern):
            """Initialises the SetValve0 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x8, "Set"))

    class SetValve1(Stream):
        def __init__(self, pattern):
            """Initialises the SetValve1 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x10, "Set"))

    class SetValve2(Stream):
        def __init__(self, pattern):
            """Initialises the SetValve2 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x20, "Set"))

    class ClearLed0(Stream):
        def __init__(self, pattern):
            """Initialises the ClearLed0 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x1, "Clear"))

    class ClearLed1(Stream):
        def __init__(self, pattern):
            """Initializes the ClearLed1 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x2, "Clear"))

    class ClearLed2(Stream):
        def __init__(self, pattern):
            """Initializes the ClearLed2 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x4, "Clear"))

    class ClearValve0(Stream):
        def __init__(self, pattern):
            """Initializes the ClearValve0 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x8, "Clear"))

    class ClearValve1(Stream):
        def __init__(self, pattern):
            """Initializes the ClearValve1 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x10, "Clear"))

    class ClearValve2(Stream):
        def __init__(self, pattern):
            """Initializes the ClearValve2 stream."""
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x20, "Clear"))
