import math
import datetime
import numpy as np
import pandas as pd

import aeon.preprocess.api as api

from . import utils


def getMouseTotalTravelledDistanceAcrossSessions(mouse_id, root, patchesIDs):
    sessions_durations = utils.getMouseSessionsStartTimesAndDurations(mouse_id=mouse_id, root=root)
    travelled_distances = np.empty((len(sessions_durations), len(patchesIDs)), dtype=np.double)
    for session_index in range(len(sessions_durations)):
        session_start_time = sessions_durations.index[session_index]
        session_end_time = session_start_time + sessions_durations[session_index]
        for patch_index, patchID  in enumerate(patchesIDs):
            wheel_encoder_vals = api.encoderdata(root, patchID, start=session_start_time, end=session_end_time)
            if len(wheel_encoder_vals)>0:
                travelled_distances[session_index, patch_index] = api.distancetravelled(wheel_encoder_vals.angle)[-1]
            else:
                travelled_distances[session_index, patch_index] = 0.0
    answer = pd.DataFrame(data=travelled_distances, index=sessions_durations.index, columns=patchesIDs)
    return answer

def getTravelledDistanceInBlocksAcrossSession(session_start_time, block_duration, root, patchesIDs):
    session_duration = utils.getSessionsDuration(session_start_time=session_start_time, root=root)
    number_blocks = math.ceil(session_duration/block_duration)
    travelled_distances = np.empty((number_blocks, len(patchesIDs)), dtype=np.double)
    block_end_time = session_start_time
    blocks_start_times = []
    for block_index in range(number_blocks):
        block_start_time = block_end_time
        blocks_start_times.append(block_start_time)
        block_end_time = block_start_time + datetime.timedelta(seconds=block_duration)
        for patch_index, patchID in enumerate(patchesIDs):
            try:
                wheel_encoder_vals = api.encoderdata(root, patchID, start=block_start_time, end=block_end_time)
            except KeyError as e:
                travelled_distances[block_index, patch_index] = 0.0

            if len(wheel_encoder_vals)>0:
                travelled_distances[block_index, patch_index] = api.distancetravelled(wheel_encoder_vals.angle)[-1]
            else:
                travelled_distances[block_index, patch_index] = 0.0
    answer = pd.DataFrame(data=travelled_distances, index=blocks_start_times, columns=patchesIDs)
    return answer
