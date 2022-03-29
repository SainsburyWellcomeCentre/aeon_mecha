import cv2
import numpy as np
import pandas as pd
from aeon.analyze.movies import *
from aeon.analyze.patches import *
from aeon.util.plotting import *
import aeon.io.api as aeon
import aeon.io.video as video
import matplotlib.pyplot as plt

dpi = 300
output = 'figures'
root = '/ceph/aeon/test2/experiment0.1'
data = aeon.sessiondata(root)
annotations = aeon.annotations(root)
data = aeon.sessionduration(data)                                     # compute session duration
data = data[data.id.str.startswith('BAA')]                            # take only mouse sessions
data = data[(data.duration > pd.Timedelta('1h'))                      # take only sessions with more than 1h
           | data.duration.isna()]                                    # or where duration is NA (no ending)
data = data[data.id == 'BAA-1099793']                                 # take only sessions from specific mouse
data = data[data.start >= pd.Timestamp('20210712')].iloc[[0]]         # take only sessions after the 14th June

for session in data.itertuples():                                     # for all sessions
    print('{0} on {1}...'.format(session.id, session.start))          # print progress report
    start, end = session.start, session.end                           # grab session start and end time
    if end is pd.NaT:
        continue
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

    frequency = 50 # Hz
    pixelscale = 0.00192 # 1 px = 1.92 mm
    framesize = (1440, 1080)
    position = aeon.positiondata(root, start=start, end=end)          # get position data between start and end
    if start > pd.Timestamp('20210621') and \
       start < pd.Timestamp('20210701'):                              # time offset to account for abnormal drop event
        position.index = position.index + pd.Timedelta('22.57966s')   # exact offset extracted from video timestamps
    valid_position = (position.area > 0) & (position.area < 1000)     # filter for objects of the correct size
    position = position[valid_position]
    position.x = position.x * pixelscale
    position.y = position.y * pixelscale

    radius = 0.95 # middle
    inner = 0.93 # inner
    outer = 0.97 # outer
    patchradius = 0.21 # patch radius
    x0, y0 = 1.475, 1.075 # center
    p1x, p1y = 1.13, 1.59 # patch1
    p2x, p2y = 1.19, 0.50 # patch2
    p1x, p1y = p1x-0.02, p1y+0.07 # offset1
    p2x, p2y = p2x-0.04, p2y-0.04 # offset2
    dist0 = distance(position, (x0, y0))
    distp1 = distance(position, (p1x, p1y))
    distp2 = distance(position, (p2x, p2y))
    in_corridor = (dist0 < outer) & (dist0 > inner)
    in_nest = dist0 > outer
    in_patch1 = activepatch(wheel1, distp1 < patchradius, position)
    in_patch2 = activepatch(wheel2, distp2 < patchradius, position)
    in_arena = ~in_corridor & ~in_nest & ~in_patch1 & ~in_patch2
    ethogram = pd.Series('other', index=position.index)
    ethogram[in_corridor] = 'corridor'
    ethogram[in_nest] = 'nest'
    ethogram[in_patch1] = 'patch1'
    ethogram[in_patch2] = 'patch2'
    ethogram[in_arena] = 'arena'

    # Select event of interest (other examples below)
    # events = pellets1
    # events = reward1
    # events = in_arena[in_arena.astype(np.int8).diff() > 0].iloc[5::10]
    events = in_patch1[in_patch1.astype(np.int8).diff() > 0]

    print("Exporting side video...")
    frames = aeon.videodata(root, 'FramePatch1', start=start, end=end)
    clips = triggerclip(frames, events, before='5s', after='10s')
    clips = clips[clips.clip_sequence.isin(list(range(0,5)))]
    movie = collatemovie(clips, lambda f:gridframes(f, 640, 1800, (5, 1)))
    movie = gridmovie(clips, 640, 1800, (5, 1))
    video.export(movie, '{0}-in_patch1.avi'.format(prefix), 60)

    print("Exporting average top video...")
    frames = aeon.videodata(root, 'FrameTop', start=start, end=end)
    clips = triggerclip(frames, events, before='5s', after='10s')
    movie = collatemovie(clips, averageframes)
    video.export(movie, '{0}-in_arena-Top.avi'.format(prefix), 60)

    # Plot wheel trqaces aligned on pellet trigger
    fig, ax = plt.subplots(1, 1)
    pelletangle = wheel1.reindex(pellets1.index, method='pad')
    for trial in pellets1.itertuples():                                                 # for each pellet delivery
        before = trial.Index - pd.to_timedelta(4, 's')                                  # get the previous 4 seconds
        after = trial.Index + pd.to_timedelta(5, 's')                                   # get the subsequent 5 seconds
        value = wheel1.loc[before:after]                                                # get wheel data around trigger
        value -= pelletangle.loc[trial.Index]
        ax.plot((value.index-trial.Index).total_seconds(), value / 100, 'b', alpha=0.2) # plot wheel traces
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance from pellet (m)')
    fig.savefig('wheel.png', dpi=dpi)

    # Plot side video of reward consumption
    frames = aeon.videodata(root, 'FramePatch1', start=start, end=end)
    clips = triggerclip(frames, events, before='5s', after='10s')
    clips = clips[clips.clip_sequence.isin(list(range(0,5)))]
    movie = gridmovie(clips, 640, 1800, (5, 1))
    video.export(movie, '{0}-in_patch1.avi'.format(prefix), 60)

    # Plot side video synchronized with wheel trace
    # This one is different from the above since we are actually saving the final rendition
    # of a plot which has both the frame and the wheel trace around that frame, for all frames
    # in the video
    events = in_patch1[in_patch1.astype(np.int8).diff() > 0]
    plt.rcParams.update({'font.size': 22})

    fig = plt.figure()
    fig.set_size_inches(19.2, 14.4)
    gs = fig.add_gridspec(3, 1)
    frame_ax = fig.add_subplot(gs[0:2,0])
    wheel_ax = fig.add_subplot(gs[2, 0])
    frames = aeon.videodata(root, 'FramePatch1', start=start, end=end)
    clips = triggerclip(frames, events, before='6s', after='11s')
    clips = clips[clips.clip_sequence == 5]
    movie = video.frames(clips)
    motion = (wheel1.diff().rolling('8ms').mean()).reindex(clips.index, method='pad') / 0.008
    wheel_ax.plot((motion.index-motion.index[0]).total_seconds(), motion, 'b')
    ymin, ymax = wheel_ax.get_ylim()
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    writer = cv2.VideoWriter('{0}-in_patch1-single-trial.avi'.format(prefix),fourcc,15, (1920, 1440))
    motion_start = motion.index[550]
    for i, frame in enumerate(movie):
        print(i)
        if i < 550 or i > len(clips)-125:
            continue
        wheel_ax.clear()
        frame_ax.clear()
        frame_ax.imshow(frame)
        frame_ax.set_axis_off()
        sample = motion.iloc[i-62:i+63]
        wheel_ax.plot((sample.index-motion_start).total_seconds(), sample, 'b')
        wheel_ax.vlines((motion.index[i]-motion_start).total_seconds(), ymin, ymax, 'k')
        wheel_ax.set_ylim(ymin,ymax)
        wheel_ax.set_ylabel('wheel speed (cm / s)')
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
    break