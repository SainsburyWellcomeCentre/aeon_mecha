
import sys
import time
import numpy as np
import pandas as pd

from . import storageMgr

sys.path.append("../..")
import aeon.preprocess.api as api
import aeon.preprocess.utils

class FilesStorageMgr(storageMgr.StorageMgr):

    def __init__(self, root):
        self._root = root
        self._metadata = api.sessiondata(root)
        self._metadata = self._metadata[self._metadata.id.str.startswith('BAA')]
        self._metadata = aeon.preprocess.utils.getPairedEvents(metadata=self._metadata)
        self._metadata = api.sessionduration(self._metadata)

    def getSessionEndTime(self, session_start_time_str):
        session_start_time = pd.to_datetime(session_start_time_str)
        session_index = np.argmin((pd.DatetimeIndex(self._metadata.start) - session_start_time).total_seconds().to_series().abs())
        session_end_time = self._metadata.iloc[session_index].end
        return session_end_time

    def getSessionPositions(self, session_start_time_str):
        session_end_time = self.getSessionEndTime(session_start_time_str=session_start_time_str)
        session_start_time = pd.to_datetime(session_start_time_str)
        position = api.positiondata(self._root, start=session_start_time,
                                    end=session_end_time)
        return position

    def getWheelAngles(self, start_time_str, end_time_str, patch_label):
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
        wheel_encoder_vals = api.encoderdata(self._root, patch_label,
                                             start=start_time,
                                             end=end_time)
        return wheel_encoder_vals.angle

    def getFoodPatchEventTimes(self, start_time_str, end_time_str, event_label, patch_label):
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
        pellet_vals = api.pelletdata(self._root, patch_label,
                                     start=start_time,
                                     end=end_time)
        pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(event_label,)].index
        return pellets_times

