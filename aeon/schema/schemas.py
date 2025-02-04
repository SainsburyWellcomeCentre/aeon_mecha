"""Schemas for different experiments."""

from dotmap import DotMap

from swc.aeon.schema import core
from swc.aeon.schema.streams import Device

from aeon.schema import foraging, octagon, social_01, social_02, social_03


exp02 = DotMap(
    [
        Device("Metadata", core.Metadata),
        Device("ExperimentalMetadata", core.Environment, core.MessageLog),
        Device("CameraTop", core.Video, core.Position, foraging.Region),
        Device("CameraEast", core.Video),
        Device("CameraNest", core.Video),
        Device("CameraNorth", core.Video),
        Device("CameraPatch1", core.Video),
        Device("CameraPatch2", core.Video),
        Device("CameraSouth", core.Video),
        Device("CameraWest", core.Video),
        Device("Nest", foraging.Weight),
        Device("Patch1", foraging.Patch),
        Device("Patch2", foraging.Patch),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.SessionData),
        Device("FrameTop", core.Video, core.Position),
        Device("FrameEast", core.Video),
        Device("FrameGate", core.Video),
        Device("FrameNorth", core.Video),
        Device("FramePatch1", core.Video),
        Device("FramePatch2", core.Video),
        Device("FrameSouth", core.Video),
        Device("FrameWest", core.Video),
        Device("Patch1", foraging.DepletionFunction, core.Encoder, foraging.Feeder),
        Device("Patch2", foraging.DepletionFunction, core.Encoder, foraging.Feeder),
    ]
)

octagon01 = DotMap(
    [
        Device("Metadata", core.Metadata),
        Device("CameraTop", core.Video, core.Position),
        Device("CameraColorTop", core.Video),
        Device("ExperimentalMetadata", core.SubjectState),
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
        Device("Metadata", core.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", core.Video, social_01.Pose),
        Device("CameraNorth", core.Video),
        Device("CameraSouth", core.Video),
        Device("CameraEast", core.Video),
        Device("CameraWest", core.Video),
        Device("CameraPatch1", core.Video),
        Device("CameraPatch2", core.Video),
        Device("CameraPatch3", core.Video),
        Device("CameraNest", core.Video),
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
        Device("Metadata", core.Metadata),
        Device("Environment", social_02.Environment, social_02.SubjectData),
        Device("CameraTop", core.Video, social_02.Pose),
        Device("CameraNorth", core.Video),
        Device("CameraSouth", core.Video),
        Device("CameraEast", core.Video),
        Device("CameraWest", core.Video),
        Device("CameraPatch1", core.Video),
        Device("CameraPatch2", core.Video),
        Device("CameraPatch3", core.Video),
        Device("CameraNest", core.Video),
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
        Device("Metadata", core.Metadata),
        Device(
            "Environment",
            social_02.Environment,
            social_02.SubjectData,
            social_03.EnvironmentActiveConfiguration,
        ),
        Device("CameraTop", core.Video, social_03.Pose),
        Device("CameraNorth", core.Video),
        Device("CameraSouth", core.Video),
        Device("CameraEast", core.Video),
        Device("CameraWest", core.Video),
        Device("CameraNest", core.Video),
        Device("CameraPatch1", core.Video),
        Device("CameraPatch2", core.Video),
        Device("CameraPatch3", core.Video),
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
        Device("Metadata", core.Metadata),
        Device(
            "Environment",
            social_02.Environment,
            social_02.SubjectData,
            social_03.EnvironmentActiveConfiguration,
        ),
        Device("CameraTop", core.Video, social_03.Pose),
        Device("CameraNorth", core.Video),
        Device("CameraSouth", core.Video),
        Device("CameraEast", core.Video),
        Device("CameraWest", core.Video),
        Device("CameraNest", core.Video),
        Device("CameraPatch1", core.Video),
        Device("CameraPatch2", core.Video),
        Device("CameraPatch3", core.Video),
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


__all__ = ["exp01", "exp02", "octagon01", "social01", "social02", "social03", "social04"]
