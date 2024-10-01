from dotmap import DotMap

import aeon.schema.core as stream
from aeon.schema import foraging, octagon
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

octagon02 = DotMap([
    Device("Metadata", stream.metadata),
    Device("CameraTop", stream.video, stream.position),
    Device("CameraColorTop", stream.video),
    Device("ExperimentalMetadata", stream.subject_state),
    Device("Photodiode", octagon.photodiode),
    Device("OSC", octagon.OSC),
    Device("TaskLogic", octagon.TaskLogic),
    Device("Wall1", octagon.Wall),
    Device("Wall2", octagon.Wall),
    Device("Wall3", octagon.Wall),
    Device("Wall4", octagon.Wall),
    Device("Wall5", octagon.Wall),
    Device("Wall6", octagon.Wall),
    Device("Wall7", octagon.Wall),
    Device("Wall8", octagon.Wall)
])
