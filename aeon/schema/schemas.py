from dotmap import DotMap
from aeon.io.device import Device
from aeon.schema import core, foraging, octagon, social01

exp02 = DotMap(
    [
        Device("Metadata", core.metadata),
        Device("ExperimentalMetadata", core.environment, core.message_log),
        Device("CameraTop", core.video, core.position, foraging.region),
        Device("CameraEast", core.video),
        Device("CameraNest", core.video),
        Device("CameraNorth", core.video),
        Device("CameraPatch1", core.video),
        Device("CameraPatch2", core.video),
        Device("CameraSouth", core.video),
        Device("CameraWest", core.video),
        Device("Nest", foraging.weight),
        Device("Patch1", foraging.patch),
        Device("Patch2", foraging.patch),
    ]
)

exp01 = DotMap(
    [
        Device("SessionData", foraging.session),
        Device("FrameTop", core.video, core.position),
        Device("FrameEast", core.video),
        Device("FrameGate", core.video),
        Device("FrameNorth", core.video),
        Device("FramePatch1", core.video),
        Device("FramePatch2", core.video),
        Device("FrameSouth", core.video),
        Device("FrameWest", core.video),
        Device("Patch1", foraging.depletion_function, core.encoder, foraging.feeder),
        Device("Patch2", foraging.depletion_function, core.encoder, foraging.feeder),
    ]
)

octagon01 = DotMap(
    [
        Device("Metadata", core.metadata),
        Device("CameraTop", core.video, core.position),
        Device("CameraColorTop", core.video),
        Device("ExperimentalMetadata", core.subject_state),
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

social01 = DotMap(
    [
        Device("Metadata", core.metadata),
        Device("Environment", social01.env_block_state_b, social01.light_events_b, core.message_log),
        Device("Subject", social01.subject_state_b, social01.subject_visits_b, social01.subject_weight_b),
        *social01.camera_devices,
        Device("Nest", social01.weight_raw_b, social01.weight_filtered_b),
        *social01.patch_devices,
        *social01.rfid_devices,
    ]
)
