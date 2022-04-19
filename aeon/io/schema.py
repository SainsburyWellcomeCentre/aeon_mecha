from dotmap import DotMap
import aeon.io.stream as stream
from aeon.io.device import Device

exp02 = DotMap([
    Device("Metadata", stream.metadata),
    Device("ExperimentalMetadata", stream.environment, stream.messageLog),
    Device("CameraTop", stream.video, stream.position, stream.region),
    Device("CameraEast", stream.video),
    Device("CameraNest", stream.video),
    Device("CameraNorth", stream.video),
    Device("CameraPatch1", stream.video),
    Device("CameraPatch2", stream.video),
    Device("CameraSouth", stream.video),
    Device("CameraWest", stream.video),
    Device("Nest", stream.weight),
    Device("Patch1", stream.patch),
    Device("Patch2", stream.patch)
])

exp01 = DotMap([
    Device("SessionData", stream.session),
    Device("FrameTop", stream.video, stream.position),
    Device("FrameEast", stream.video),
    Device("FrameGate", stream.video),
    Device("FrameNorth", stream.video),
    Device("FramePatch1", stream.video),
    Device("FramePatch2", stream.video),
    Device("FrameSouth", stream.video),
    Device("FrameWest", stream.video),
    Device("Patch1", stream.depletionFunction, stream.encoder, stream.feeder),
    Device("Patch2", stream.depletionFunction, stream.encoder, stream.feeder)
])