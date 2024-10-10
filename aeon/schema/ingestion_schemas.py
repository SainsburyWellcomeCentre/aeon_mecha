"""Aeon experiment schemas for DataJoint database ingestion."""

from os import PathLike

import pandas as pd
from dotmap import DotMap

import aeon.schema.core as stream
from aeon.io import reader
from aeon.io.api import aeon as aeon_time
from aeon.schema import foraging, octagon, social_01, social_02, social_03
from aeon.schema.foraging import DepletionFunction, Feeder
from aeon.schema.streams import Device, Stream, StreamGroup


# Define new readers
class _Encoder(reader.Encoder):
    """A version of the encoder reader that can downsample the data."""

    def read(self, file: PathLike[str], sr_hz: int = 50) -> pd.DataFrame:
        """Reads encoder data from the specified Harp binary file."""
        data = super().read(file)
        data.index = aeon_time(data.index)
        first_index = data.first_valid_index()
        if first_index is not None:
            data = data.resample(f"{1/sr_hz}s").first()  # take first sample in each resampled bin
        return data


class _Video(reader.Video):
    """A version of the video reader that drops columns that can be recreated from data and metadata."""

    def read(self, file: PathLike[str], drop_cols=None) -> pd.DataFrame:
        """Reads video metadata from the specified file."""
        drop_cols = ["hw_counter", "_frame", "_path", "_epoch"] if drop_cols is None else drop_cols
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data.drop(columns=drop_cols, errors="ignore", inplace=True)
        data.set_index("time", inplace=True)
        return data


# Define new streams and stream groups
class Video(Stream):
    """Video frame metadata."""

    def __init__(self, pattern):
        super().__init__(_Video(f"{pattern}_*"))


class Encoder(Stream):
    """Wheel magnetic encoder data."""

    def __init__(self, pattern):
        super().__init__(_Encoder(f"{pattern}_90_*"))


class Patch(StreamGroup):
    """Data streams for a patch."""

    def __init__(self, pattern):
        super().__init__(pattern, DepletionFunction, Encoder, Feeder)


# Define schemas
octagon01 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("CameraTop", Video, stream.Position),
        Device("CameraColorTop", Video),
        Device("ExperimentalMetadata", stream.SubjectState),
        Device("Photodiode", octagon.Photodiode),
        Device("OSC", octagon.OSC),
        Device("TaskLogic", octagon.TaskLogic),
        Device("Wall1", octagon.Wall),
        Device("Wall2", octagon.Wall),
        Device("Wall3", octagon.Wall),
        Device("Wall4", octagon.Wall),
        Device("Wall5", octagon.Wall),
        Device("Wall6", octagon.Wall),
        Device("Wall7", octagon.Wall),
        Device("Wall8", octagon.Wall),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.SessionData),
        Device("FrameTop", Video, stream.Position),
        Device("FrameEast", Video),
        Device("FrameGate", Video),
        Device("FrameNorth", Video),
        Device("FramePatch1", Video),
        Device("FramePatch2", Video),
        Device("FrameSouth", Video),
        Device("FrameWest", Video),
        Device("Patch1", foraging.DepletionFunction, stream.Encoder, foraging.Feeder),
        Device("Patch2", foraging.DepletionFunction, stream.Encoder, foraging.Feeder),
    ]
)

exp02 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("ExperimentalMetadata", stream.Environment, stream.MessageLog),
        Device("CameraTop", Video, stream.Position, foraging.Region),
        Device("CameraEast", Video),
        Device("CameraNest", Video),
        Device("CameraNorth", Video),
        Device("CameraPatch1", Video),
        Device("CameraPatch2", Video),
        Device("CameraSouth", Video),
        Device("CameraWest", Video),
        Device("Nest", foraging.Weight),
        Device("Patch1", Patch),
        Device("Patch2", Patch),
    ]
)

social01 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", Video, social_01.Pose),
        Device("CameraNorth", Video),
        Device("CameraSouth", Video),
        Device("CameraEast", Video),
        Device("CameraWest", Video),
        Device("CameraPatch1", Video),
        Device("CameraPatch2", Video),
        Device("CameraPatch3", Video),
        Device("CameraNest", Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", Patch),
        Device("Patch2", Patch),
        Device("Patch3", Patch),
        Device("RfidGate", social_01.RfidEvents),
        Device("RfidNest1", social_01.RfidEvents),
        Device("RfidNest2", social_01.RfidEvents),
        Device("RfidPatch1", social_01.RfidEvents),
        Device("RfidPatch2", social_01.RfidEvents),
        Device("RfidPatch3", social_01.RfidEvents),
    ]
)


social02 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", Video, social_02.Pose),
        Device("CameraNorth", Video),
        Device("CameraSouth", Video),
        Device("CameraEast", Video),
        Device("CameraWest", Video),
        Device("CameraPatch1", Video),
        Device("CameraPatch2", Video),
        Device("CameraPatch3", Video),
        Device("CameraNest", Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", Patch),
        Device("Patch2", Patch),
        Device("Patch3", Patch),
        Device("GateRfid", social_02.RfidEvents),
        Device("NestRfid1", social_02.RfidEvents),
        Device("NestRfid2", social_02.RfidEvents),
        Device("Patch1Rfid", social_02.RfidEvents),
        Device("Patch2Rfid", social_02.RfidEvents),
        Device("Patch3Rfid", social_02.RfidEvents),
    ]
)


social03 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", Video, social_03.Pose),
        Device("CameraNorth", Video),
        Device("CameraSouth", Video),
        Device("CameraEast", Video),
        Device("CameraWest", Video),
        Device("CameraNest", Video),
        Device("CameraPatch1", Video),
        Device("CameraPatch2", Video),
        Device("CameraPatch3", Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", Patch),
        Device("Patch2", Patch),
        Device("Patch3", Patch),
        Device("PatchDummy1", Patch),
        Device("NestRfid1", social_02.RfidEvents),
        Device("NestRfid2", social_02.RfidEvents),
        Device("GateRfid", social_02.RfidEvents),
        Device("GateEastRfid", social_02.RfidEvents),
        Device("GateWestRfid", social_02.RfidEvents),
        Device("Patch1Rfid", social_02.RfidEvents),
        Device("Patch2Rfid", social_02.RfidEvents),
        Device("Patch3Rfid", social_02.RfidEvents),
        Device("PatchDummy1Rfid", social_02.RfidEvents),
    ]
)


__all__ = ["octagon01", "exp01", "exp02", "social01", "social02", "social03"]
