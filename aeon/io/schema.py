from dotmap import DotMap
from aeon.io.devices import *

exp02 = DotMap([
    Device("CameraTop", videoStream, positionStream, regionStream),
    Device("CameraEast", videoStream),
    Device("CameraNest", videoStream),
    Device("CameraNorth", videoStream),
    Device("CameraPatch1", videoStream),
    Device("CameraPatch2", videoStream),
    Device("CameraSouth", videoStream),
    Device("CameraWest", videoStream),
    Device("ExperimentalMetadata", metadataStream, messageLogStream),
    Device("Nest", weightStream),
    Device("Patch1", depletionFunctionStream, encoderStream, feederStream),
    Device("Patch2", depletionFunctionStream, encoderStream, feederStream)
])