import pandas as pd
import aeon.io.api as aeon

root = '/ceph/aeon/test2/experiment0.1'
videobins = aeon.chunkdata(root, 'FrameTop')

stats = []
for timebin in videobins.itertuples():
    print('Analysing {0}... '.format(timebin.Index),end="")
    data = aeon.videoreader(timebin.path).reset_index()
    deltas = data[data.columns[0:4]].diff()
    counter_drops = (deltas.hw_counter - 1).sum()
    max_harp_delta = deltas.time.max().total_seconds()
    max_camera_delta = deltas.hw_timestamp.max() / 1e9 # convert nanoseconds to seconds
    print('drops: {0}  maxHarpDelta: {1} s  maxCameraDelta: {2} s'.format(
        counter_drops,
        max_harp_delta,
        max_camera_delta))
    stats.append((counter_drops, max_harp_delta, max_camera_delta, timebin.path))

stats = pd.DataFrame(stats,
    columns=['drop_frames', 'harp_delta', 'camera_delta', 'path'],
    index=videobins.index)
