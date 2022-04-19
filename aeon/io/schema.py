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
    Device("Patch1", stream.depletionFunction, stream.encoder, stream.feeder),
    Device("Patch2", stream.depletionFunction, stream.encoder, stream.feeder)
])