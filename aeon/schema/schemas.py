""" Schemas for different experiments. """

from dotmap import DotMap

import aeon.schema.core as stream
from aeon.schema import foraging, octagon, social_01, social_02, social_03
from aeon.schema.streams import Device

exp02 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("ExperimentalMetadata", stream.Environment, stream.MessageLog),
        Device("CameraTop", stream.Video, stream.Position, foraging.Region),
        Device("CameraEast", stream.Video),
        Device("CameraNest", stream.Video),
        Device("CameraNorth", stream.Video),
        Device("CameraPatch1", stream.Video),
        Device("CameraPatch2", stream.Video),
        Device("CameraSouth", stream.Video),
        Device("CameraWest", stream.Video),
        Device("Nest", foraging.Weight),
        Device("Patch1", foraging.Patch),
        Device("Patch2", foraging.Patch),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.SessionData),
        Device("FrameTop", stream.Video, stream.Position),
        Device("FrameEast", stream.Video),
        Device("FrameGate", stream.Video),
        Device("FrameNorth", stream.Video),
        Device("FramePatch1", stream.Video),
        Device("FramePatch2", stream.Video),
        Device("FrameSouth", stream.Video),
        Device("FrameWest", stream.Video),
        Device("Patch1", foraging.DepletionFunction, stream.Encoder, foraging.Feeder),
        Device("Patch2", foraging.DepletionFunction, stream.Encoder, foraging.Feeder),
    ]
)

octagon01 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("CameraTop", stream.Video, stream.Position),
        Device("CameraColorTop", stream.Video),
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

social01 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", stream.Video, social_01.Pose),
        Device("CameraNorth", stream.Video),
        Device("CameraSouth", stream.Video),
        Device("CameraEast", stream.Video),
        Device("CameraWest", stream.Video),
        Device("CameraPatch1", stream.Video),
        Device("CameraPatch2", stream.Video),
        Device("CameraPatch3", stream.Video),
        Device("CameraNest", stream.Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", social_02.Patch),
        Device("Patch2", social_02.Patch),
        Device("Patch3", social_02.Patch),
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
        Device("CameraTop", stream.Video, social_02.Pose),
        Device("CameraNorth", stream.Video),
        Device("CameraSouth", stream.Video),
        Device("CameraEast", stream.Video),
        Device("CameraWest", stream.Video),
        Device("CameraPatch1", stream.Video),
        Device("CameraPatch2", stream.Video),
        Device("CameraPatch3", stream.Video),
        Device("CameraNest", stream.Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", social_02.Patch),
        Device("Patch2", social_02.Patch),
        Device("Patch3", social_02.Patch),
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
        Device(
            "Environment",
            social_02.Environment,
            social_02.SubjectData,
            social_03.EnvironmentActiveConfiguration,
        ),
        Device("CameraTop", stream.Video, social_03.Pose),
        Device("CameraNorth", stream.Video),
        Device("CameraSouth", stream.Video),
        Device("CameraEast", stream.Video),
        Device("CameraWest", stream.Video),
        Device("CameraNest", stream.Video),
        Device("CameraPatch1", stream.Video),
        Device("CameraPatch2", stream.Video),
        Device("CameraPatch3", stream.Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", social_02.Patch),
        Device("Patch2", social_02.Patch),
        Device("Patch3", social_02.Patch),
        Device("PatchDummy1", social_02.Patch),
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


social04 = DotMap(
    [
        Device("Metadata", stream.Metadata),
        Device(
            "Environment",
            social_02.Environment,
            social_02.SubjectData,
            social_03.EnvironmentActiveConfiguration,
        ),
        Device("CameraTop", stream.Video, social_03.Pose),
        Device("CameraNorth", stream.Video),
        Device("CameraSouth", stream.Video),
        Device("CameraEast", stream.Video),
        Device("CameraWest", stream.Video),
        Device("CameraNest", stream.Video),
        Device("CameraPatch1", stream.Video),
        Device("CameraPatch2", stream.Video),
        Device("CameraPatch3", stream.Video),
        Device("Nest", social_02.WeightRaw, social_02.WeightFiltered),
        Device("Patch1", social_02.Patch),
        Device("Patch2", social_02.Patch),
        Device("Patch3", social_02.Patch),
        Device("PatchDummy1", social_02.Patch),
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


__all__ = [
    "exp01",
    "exp02",
    "octagon01",
    "social01",
    "social02",
    "social03",
    "social04",
]
