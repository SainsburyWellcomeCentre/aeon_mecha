import sys
import pdb
import datetime
import argparse

import aeon.preprocess.api
import aeon.preprocess.utils


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Root path for data access", default="/ceph/aeon/test2/experiment0.1")
    parser.add_argument("--session", help="Session index", default=3, type=int)
    parser.add_argument("--start_time", help="Start time (sec)", default=12000.0, type=float)
    parser.add_argument("--duration", help="Duration (sec)", default=600.0, type=float)
    parser.add_argument("--patch_name", help="Patch name", default="Patch2")
    parser.add_argument("--pellet_event_name", help="Pellet event name to display", default="TriggerPellet")

    args = parser.parse_args()

    root = args.root
    session = args.session
    t0_relative = args.start_time
    requested_duration = args.duration
    tf_relative = args.start_time+args.duration
    patch_name = args.patch_name
    pellet_event_name = args.pellet_event_name

    metadata = aeon.preprocess.api.sessiondata(root)
    metadata = metadata[metadata.id.str.startswith('BAA')]
    metadata = aeon.preprocess.utils.getPairedEvents(metadata=metadata)
    metadata = aeon.preprocess.api.sessionduration(metadata)

    session_duration_sec = metadata.iloc[session].duration.total_seconds()
    if requested_duration>session_duration_sec:
        raise ValueError("Requested duration {:f} exceeds the session duration {:f}".format(requested_duration, session_duration_sec))

    session_start = metadata.index[session]
    t0_absolute = session_start + datetime.timedelta(seconds=t0_relative)
    tf_absolute = session_start + datetime.timedelta(seconds=tf_relative)
    pellet_vals = aeon.preprocess.api.pelletdata(root, patch_name, start=t0_absolute, end=tf_absolute)
    pellets_times = pellet_vals[pellet_vals.event == "{:s}".format(pellet_event_name)].index
    print("pellets_times ==", pellets_times)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
