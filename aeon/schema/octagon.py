import aeon.io.reader as _reader
from aeon.schema.streams import Stream, StreamGroup


class Photodiode(Stream):
    def __init__(self, path):
        super().__init__(_reader.Harp(f"{path}_44_*", columns=["adc", "encoder"]))


class OSC(StreamGroup):
    def __init__(self, path):
        super().__init__(path)

    class BackgroundColor(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(f"{pattern}_backgroundcolor_*", columns=["typetag", "r", "g", "b", "a"])
            )

    class ChangeSubjectState(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(f"{pattern}_changesubjectstate_*", columns=["typetag", "id", "weight", "event"])
            )

    class EndTrial(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Csv(f"{pattern}_endtrial_*", columns=["typetag", "value"]))

    class Slice(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(
                    f"{pattern}_octagonslice_*", columns=["typetag", "wall_id", "r", "g", "b", "a", "delay"]
                )
            )

    class GratingsSlice(Stream):
        def __init__(self, pattern):
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
                        "aperture_angle",
                        "delay",
                    ],
                )
            )

    class CheckerboardsSlice(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(
                f"{pattern}_octagoncheckerboardsslice_*",
                columns=[
                    "typetag",
                    "wall_id",
                    "rows",
                    "columns",
                    "contrast",
                    "temporal_frequency",
                    "angle",
                    "aperture_angle",
                    "delay",
                ],
            )
            )

    class Poke(Stream):
        def __init__(self, pattern):
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
            super().__init__(
                _reader.Csv(
                    f"{pattern}_response_*", columns=["typetag", "wall_id", "poke_id", "response_time"]
                )
            )

    class RunPreTrialNoPoke(Stream):
        def __init__(self, pattern):
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
            super().__init__(_reader.Csv(f"{pattern}_startnewsession_*", columns=["typetag", "path"]))

    class TrackingResponse(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(
            f"{pattern}_return_tracking_response_*", columns=[
            'typetag',
            'id',
            'response_position_x_1',
            'response_position_y_1',
            'response_theta_1',
            'response_area_1'])
            )

    class TrackingSliceOnset(Stream):
        def __init__(self, pattern):
            super().__init__(
                _reader.Csv(
                    f"{pattern}_return_tracking_slice_onset_*", columns=[
            'typetag',
            'id',
            'slice_onset_position_x_1',
            'slice_onset_position_y_1',
            'slice_onset_theta_1',
            'slice_onset_area_1'])
            )
        
class TaskLogic(StreamGroup):
    def __init__(self, path):
        super().__init__(path)

    class TrialInitiation(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_1_*", columns=["trial_type"]))

    class Response(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_2_*", columns=["wall_id", "poke_id"]))

    class PreTrialState(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_3_*", columns=["state"]))

    class InterTrialInterval(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_4_*", columns=["state"]))

    class SliceOnset(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_10_*", columns=["wall_id"]))

    class DrawBackground(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_11_*", columns=["state"]))

    class GratingsSliceOnset(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_12_*", columns=["wall_id"]))

    class CheckerboardSliceOnset(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.Harp(f"{pattern}_14_*", columns=["wall_id"]))


class Wall(StreamGroup):
    def __init__(self, path):
        super().__init__(path)

    class BeamBreak0(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.DigitalBitmask(f"{pattern}_32_*", 0x1, columns=["state"]))

    class BeamBreak1(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.DigitalBitmask(f"{pattern}_32_*", 0x2, columns=["state"]))

    class BeamBreak2(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.DigitalBitmask(f"{pattern}_32_*", 0x4, columns=["state"]))

    class SetLed0(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x1, "Set"))

    class SetLed1(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x2, "Set"))

    class SetLed2(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x4, "Set"))

    class SetValve0(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x8, "Set"))

    class SetValve1(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x10, "Set"))

    class SetValve2(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_34_*", 0x20, "Set"))

    class ClearLed0(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x1, "Clear"))

    class ClearLed1(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x2, "Clear"))

    class ClearLed2(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x4, "Clear"))

    class ClearValve0(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x8, "Clear"))

    class ClearValve1(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x10, "Clear"))

    class ClearValve2(Stream):
        def __init__(self, pattern):
            super().__init__(_reader.BitmaskEvent(f"{pattern}_35_*", 0x20, "Clear"))
