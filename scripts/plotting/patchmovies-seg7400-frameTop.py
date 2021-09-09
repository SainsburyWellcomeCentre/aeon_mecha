import sys
sys.path.append("../..")

import cv2
import numpy as np
import pandas as pd
from aeon.analyze.movies import *
from aeon.analyze.patches import *
from aeon.util.plotting import *
import aeon.preprocess.api as aeon
import matplotlib.pyplot as plt

dpi = 300
output = 'figures'
root = '/ceph/aeon/test2/experiment0.1'
data = aeon.sessiondata(root)
data = aeon.sessionduration(data)                                     # compute session duration

session = data.iloc[0]
print('{0} on {1}...'.format(session.id, session.start))          # print progress report
start, end = session.start, session.end                           # grab session start and end time
prefix = '{0}_{1}'.format(session.id,start.date())                # format figure prefix

encoder1 = aeon.encoderdata(root, 'Patch1', start=start, end=end) # get encoder data for patch1 between start and end
encoder2 = aeon.encoderdata(root, 'Patch2', start=start, end=end) # get encoder data for patch2 between start and end
pellets1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)  # get pellet events for patch1 between start and end
pellets2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)  # get pellet events for patch2 between start and end
state1 = aeon.patchdata(root, 'Patch1', start=start, end=end)
state2 = aeon.patchdata(root, 'Patch2', start=start, end=end)

wheel1 = aeon.distancetravelled(encoder1.angle)                   # compute total distance travelled on patch1 wheel
wheel2 = aeon.distancetravelled(encoder2.angle)                   # compute total distance travelled on patch2 wheel
reward1 = pellets1[pellets1.event == 'PelletDetected']            # get timestamps of pellets delivered at patch1
reward2 = pellets2[pellets2.event == 'PelletDetected']            # get timestamps of pellets delivered at patch2
pellets1 = pellets1[pellets1.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch1
pellets2 = pellets2[pellets2.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch2

video_fps = 50 # Hz
pixelscale = 0.00192 # 1 px = 1.92 mm
framesize = (1440, 1080)
video = aeon.videodata(root, 'FrameTop', start=start, end=end)    # get video frame metadata between start and end
position = aeon.positiondata(root, start=start, end=end)          # get position data between start and end
position.x = position.x * pixelscale
position.y = position.y * pixelscale

seg_start = start + pd.Timedelta(7400, 's')                       # define segment starting 7400 seconds from session start
seg_end = start + pd.Timedelta(7800, 's')                         # and ending 7800 seconds from session start
wheel1_seg = wheel1[seg_start:seg_end]                            # get wheel position data for the segment
position_seg = position[seg_start:seg_end]                        # get position data for the segment

video_seg = video[seg_start:seg_end]                              # get video frame metadata for the segment
wheel_seg = wheel1_seg.reindex(video_seg.index, method='nearest') # resample wheel data using video FPS for plotting

# Plot side video synchronized with wheel trace
# Here we are saving the rendition of a plot which has both the frame and the wheel trace around that frame,
# for all frames in the video

# setup figure
plt.rcParams.update({'font.size': 22})
fig = plt.figure()
fig.set_size_inches(19.2, 14.4)
gs = fig.add_gridspec(3, 1)
frame_ax = fig.add_subplot(gs[0:2,0])
wheel_ax = fig.add_subplot(gs[2, 0])

movie = aeon.videoframes(video_seg)
wheel_ax.plot(wheel1_seg)

# setup dynamic plot export
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
writer = cv2.VideoWriter('seg7400-frameTop.avi', fourcc, video_fps, (1920, 1440))

for i, frame in enumerate(movie):
    print(i)
    # clear all axes for next frame
    wheel_ax.clear()
    frame_ax.clear()
    frame_ax.imshow(frame)
    frame_ax.set_axis_off()

    # get data for current frame
    time, sample = wheel_seg.index[i], wheel_seg.iloc[i]
    wheel_ax.plot(wheel1_seg)
    wheel_ax.plot(time, sample, 'r.')
    wheel_ax.set_ylabel('distance travelled (cm)')
    wheel_ax.set_xlabel('time (s)')
    plt.tight_layout()
    fig.subplots_adjust(left=0.2,right=0.8)
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, -1)
    img = cv2.cvtColor(img, cv2.cv2.COLOR_RGBA2BGR)
    writer.write(img)
writer.release()

plt.close(fig)
