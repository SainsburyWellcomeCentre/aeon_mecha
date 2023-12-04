from dotmap import DotMap

import aeon.io.binder.core as stream
from aeon.io.device import Device
from aeon.io.binder import foraging, octagon

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

# All recorded social01 streams:

# *Note* regiser 8 is always the harp heartbeat for any device that has this stream.

# - Metadata.yml
# - Environment_BlockState
# - Environment_EnvironmentState
# - Environment_LightEvents
# - Environment_MessageLog
# - Environment_SubjectState
# - Environment_SubjectVisits
# - Environment_SubjectWeight
# - CameraTop (200, 201, avi, csv, <model_path>,)
#     - 200: position
#     - 201: region
# - CameraNorth (avi, csv)
# - CameraEast (avi, csv)
# - CameraSouth (avi, csv)
# - CameraWest (avi, csv)
# - CameraPatch1 (avi, csv)
# - CameraPatch2 (avi, csv)
# - CameraPatch3 (avi, csv)
# - CameraNest (avi, csv)
# - ClockSynchronizer (8, 36)
#     - 36: 
# - Nest (200, 201, 202, 203)
#     - 200: weight_raw
#     - 201: weight_tare
#     - 202: weight_filtered
#     - 203: weight_baseline
#     - 204: weight_subject
# - Patch1 (8, 32, 35, 36, 87, 90, 91, 200, 201, 202, 203, State)
#     - 32: beam_break
#     - 35: delivery_set
#     - 36: delivery_clear
#     - 87: expansion_board
#     - 90: enocder_read
#     - 91: encoder_mode
#     - 200: dispenser_state
#     - 201: delivery_manual
#     - 202: missed_pellet
#     - 203: delivery_retry
# - Patch2 (8, 32, 35, 36, 87, 90, 91, State)
# - Patch3 (8, 32, 35, 36, 87, 90, 91, 200, 203, State)
# - RfidEventsGate (8, 32, 35)
#     - 32: entry_id
#     - 35: hardware_notifications
# - RfidEventsNest1 (8, 32, 35)
# - RfidEventsNest2 (8, 32, 35)
# - RfidEventsPatch1 (8, 32, 35)
# - RfidEventsPatch2 (8, 32, 35)
# - RfidEventsPatch3 (8, 32, 35)
# - VideoController (8, 32, 33, 34, 35, 36, 45, 52)
#     - 32: frame_number