import pdb
import numpy as np

def getPairedEvents(metadata):
    paired_events = None
    i = 0
    while i < (len(metadata)-1):
        if metadata.iloc[i]["event"]=="Start" and metadata.iloc[i+1]["event"]=="End":
            if paired_events is None:
                paired_events = metadata.iloc[i:(i+2)]
            else:
                paired_events = paired_events.append(metadata.iloc[i:(i+2)])
            i += 2
        else:
            i += 1
    return paired_events
