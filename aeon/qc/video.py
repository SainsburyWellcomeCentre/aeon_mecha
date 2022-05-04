import os
import aeon.io.api as aeon
from pathlib import Path

root = '/ceph/aeon/test2/experiment0.1'
qcroot = '/ceph/aeon/aeon/qc/experiment0.1'
devicenames = ['FrameEast','FrameGate','FrameNorth','FramePatch1','FramePatch2','FrameSouth','FrameTop','FrameWest']

for device in devicenames:
    videochunks = aeon.chunkdata(root, device)
    videochunks['epoch'] = videochunks.path.str.rsplit('/', n=3, expand=True)[1]

    stats = []
    frameshifts = []
    for (key, period) in videochunks.groupby(by='epoch'):
        frame_offset = 0
        path = Path(os.path.join(qcroot, key, device))
        path.mkdir(parents=True, exist_ok=True)
        for chunk in period.itertuples():
            outpath = Path(chunk.path.replace(root, qcroot)).with_suffix('.parquet')
            print('[{1}] Analysing {0} {2}... '.format(device, key,chunk.Index),end="")
            data = aeon.videoreader(chunk.path).reset_index()
            deltas = data[data.columns[0:4]].diff()
            deltas.columns = [ 'time_delta', 'frame_delta', 'hw_counter_delta', 'hw_timestamp_delta']
            deltas['frame_offset'] = (deltas.hw_counter_delta - 1).cumsum() + frame_offset
            drop_count = deltas.frame_offset.iloc[-1]
            max_harp_delta = deltas.time_delta.max().total_seconds()
            max_camera_delta = deltas.hw_timestamp_delta.max() / 1e9 # convert nanoseconds to seconds
            print('drops: {0} frameOffset: {1}  maxHarpDelta: {2} s  maxCameraDelta: {3} s'.format(
                drop_count - frame_offset,
                drop_count,
                max_harp_delta,
                max_camera_delta))
            stats.append((drop_count, max_harp_delta, max_camera_delta, chunk.path))
            deltas.set_index(data.time, inplace=True)
            deltas.to_parquet(outpath)
            frameshifts.append(deltas)
            frame_offset = drop_count