import sys
import pdb
import numpy as np
import pandas as pd
import datetime
import pathlib
import argparse
import plotly.graph_objects as go

sys.path.append("../../")
import aeon.preprocess.api as api
import aeon.preprocess.utils
import aeon.plotting.plot_functions

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--sessionIndex", help="Session index", default=10, type=int)
    parser.add_argument("--patchID", help="Patch ID", default="Patch1")

    args = parser.parse_args()

    root = args.root
    sessionIndex = args.sessionIndex
    patchID = args.patchID

    metadata = api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = api.sessionduration(metadata)
    state = aeon.preprocess.api.patchdata(root, patchID, start=metadata.iloc[sessionIndex].start, end=metadata.iloc[sessionIndex].end)
    thr_changes = state["threshold"].iloc[1:].to_numpy()-state["threshold"].iloc[:-1].to_numpy()
    idx_changes = np.where(thr_changes>0)[0]
    dates_changes = []
    if len(idx_changes)>0:
        dates_changes = state.index[idx_changes+1]
    print(state)
    if len(dates_changes)>0:
        print("Changes in thr in {:s} at dates".format(patchID))
        print(dates_changes)
    else:
        print("No changes in thr detected in {:s}".format(patchID))
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
