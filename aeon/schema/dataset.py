from dotmap import DotMap
import aeon.schema.core as stream
import aeon.schema.foraging as foraging
import aeon.schema.octagon as octagon
from aeon.io.device import Device

exp02 = DotMap([
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
    Device("Patch1", foraging.patch),
    Device("Patch2", foraging.patch)
])

exp01 = DotMap([
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
    Device("Patch2", foraging.depletionFunction, stream.encoder, foraging.feeder)
])

octagon01 = DotMap([
    Device("Metadata", stream.metadata),
    Device("CameraTop", stream.video, stream.position),
    Device("CameraColorTop", stream.video),
    Device("ExperimentalMetadata", stream.subject_state),
    Device("Photodiode", octagon.photodiode),
    Device(
        "OSC",
        octagon.OSC.background_color,
        octagon.OSC.slice,
        octagon.OSC.gratings_slice,
        octagon.OSC.poke,
        octagon.OSC.response,
        octagon.OSC.run_pre_trial_no_poke,
        octagon.OSC.start_new_session),
    Device(
        "TaskLogic",
        octagon.TaskLogic.trial_initiation,
        octagon.TaskLogic.response,
        octagon.TaskLogic.pre_trial,
        octagon.TaskLogic.inter_trial_interval,
        octagon.TaskLogic.slice_onset,
        octagon.TaskLogic.draw_background,
        octagon.TaskLogic.gratings_slice_onset),
    Device("Wall1", octagon.Wall),
    Device("Wall2", octagon.Wall),
    Device("Wall3", octagon.Wall),
    Device("Wall4", octagon.Wall),
    Device("Wall5", octagon.Wall),
    Device("Wall6", octagon.Wall),
    Device("Wall7", octagon.Wall),
    Device("Wall8", octagon.Wall)
])