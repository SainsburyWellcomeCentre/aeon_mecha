
import abc

class StorageMgr(abc.ABC):

    @abc.abstractmethod
    def getSessionPositions(self, session_start_time_str):
        pass

    @abc.abstractmethod
    def getSessionEndTime(self, session_start_time_str):
        pass

    @abc.abstractmethod
    def getWheelAngles(self, start_time_str, end_time_str, patch_label):
        pass

    @abc.abstractmethod
    def getFoodPatchEventTimes(self, start_time_str, end_time_str, event_label, patch_label):
        pass


