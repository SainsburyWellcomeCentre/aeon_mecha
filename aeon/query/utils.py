import aeon.preprocess.api as api
import aeon.preprocess.utils


def getMouseSessionsStartTimesAndDurations(mouse_id, root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    durations = metadata.loc[metadata.id == mouse_id, "duration"]
    return durations


def getAllSessionsStartTimes(root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    answer = metadata.index
    return answer


def getSessionsDuration(session_start_time, root):
    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    duration = metadata.loc[session_start_time, "duration"].total_seconds()
    return duration
