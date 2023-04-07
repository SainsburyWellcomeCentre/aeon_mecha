from dotmap import DotMap

import aeon.schema.core as stream
import aeon.schema.foraging as foraging
import aeon.schema.octagon as octagon
from aeon.io.device import Device

exp02 = DotMap(
    [
        Device("Metadata", stream.metadata),
        Device("ExperimentalMetadata", stream.environment, stream.messageLog),
        Device("CameraTop", stream.video, stream.position, foraging.region),
        Device("CameraEast", stream.video),
        Device("CameraNest", stream.video),
        Device("CameraNorth", stream.video),
        Device("CameraPatch1", stream.video),
        Device("CameraPatch2", stream.video),
        Device("CameraSouth", stream.video),
        Device("CameraWest", stream.video),
        Device("Nest", foraging.weight),
        Device("WeightNest", foraging.weight),
        Device("Patch1", foraging.patch),
        Device("Patch2", foraging.patch),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.session),
        Device("FrameTop", stream.video, stream.position),
        Device("FrameEast", stream.video),
        Device("FrameGate", stream.video),
        Device("FrameNorth", stream.video),
        Device("FramePatch1", stream.video),
        Device("FramePatch2", stream.video),
        Device("FrameSouth", stream.video),
        Device("FrameWest", stream.video),
        Device("Patch1", foraging.depletionFunction, stream.encoder, foraging.feeder),
        Device("Patch2", foraging.depletionFunction, stream.encoder, foraging.feeder),
    ]
)

octagon01 = DotMap(
    [
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
        Device("Wall8", octagon.Wall),
    ]
)


def get_device_info(schema: DotMap) -> dict[dict]:
    """
    Read from the above DotMap object and returns device dictionary {device_name: {stream_name: reader}}
    """
    from collections import defaultdict

    device_info = {}

    for device_name in schema:
        if not device_name.startswith("_"):
            device_info[device_name] = defaultdict(list)
            if isinstance(schema[device_name], DotMap):
                for stream_type in schema[device_name].keys():
                    if schema[device_name][stream_type].__class__.__module__ in [
                        "aeon.io.reader",
                        "aeon.schema.foraging",
                        "aeon.schema.octagon",
                    ]:
                        device_info[device_name]["stream_type"].append(stream_type)
                        device_info[device_name]["reader"].append(
                            schema[device_name][stream_type].__class__
                        )
            else:
                stream_type = schema[device_name].__class__.__name__
                device_info[device_name]["stream_type"].append(stream_type)
                device_info[device_name]["reader"].append(schema[device_name].__class__)
    return device_info
