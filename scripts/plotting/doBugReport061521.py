import sys
import pdb
import datetime
import argparse

import aeon.preprocess.utils
from aeon.query import exp0_api


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=0.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=14000.0, type=float)
    parser.add_argument("--patch_name", help="Patch name", default="Patch2")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    requested_duration = args.duration
    tf_relative = args.start_time+args.duration
    patch_name = args.patch_name

    metadata = exp0_api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = exp0_api.sessionduration(metadata)

    session_duration_sec = metadata.iloc[session].duration.total_seconds()
    if requested_duration>session_duration_sec:
        raise ValueError("Requested duration {:f} exceeds the session duration {:f}".format(requested_duration, session_duration_sec))
    print("Session duration {:f}".format(session_duration_sec))

    session_start = metadata.index[session]
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)
    pellet_vals = exp0_api.pelletdata(root, patch_name, start=t0_absolute, end=tf_absolute)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
