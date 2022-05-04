import pandas as pd
from colorline import colorline
from aeon.analyze.patches import *
from aeon.util.plotting import *
import aeon.io.api as aeon
import matplotlib.pyplot as plt

dpi = 300
output = 'figures'
root = '/ceph/aeon/test2/experiment0.1'
data = aeon.sessiondata(root)
annotations = aeon.annotations(root)
data = aeon.sessionduration(data)                                     # compute session duration
data = data[data.id.str.startswith('BAA')]                            # take only mouse sessions
data = data[data.start >= pd.Timestamp('20210621')]                   # take only sessions after the 14th June
data = data[(data.duration > pd.Timedelta('1h'))                      # take only sessions with more than 1h
           | data.duration.isna()]                                    # or where duration is NA (no ending)
data = data[data.id == 'BAA-1099793']                                 # take only sessions from specific mouse

for session in data.itertuples():                                     # for all sessions
    print('{0} on {1}...'.format(session.id, session.start))          # print progress report
    start, end = session.start, session.end                           # grab session start and end time
    if end is pd.NaT:                                                 # ignore invalid sessions
        continue
    prefix = '{0}_{1}'.format(session.id,start.date())                # format figure prefix

    encoder1 = aeon.encoderdata(root, 'Patch1', start=start, end=end) # get encoder data for patch1 between start and end
    encoder2 = aeon.encoderdata(root, 'Patch2', start=start, end=end) # get encoder data for patch2 between start and end
    pellets1 = aeon.pelletdata(root, 'Patch1', start=start, end=end)  # get pellet events for patch1 between start and end
    pellets2 = aeon.pelletdata(root, 'Patch2', start=start, end=end)  # get pellet events for patch2 between start and end
    state1 = aeon.patchdata(root, 'Patch1', start=start, end=end)     # get patch state for patch1 between start and end
    state2 = aeon.patchdata(root, 'Patch2', start=start, end=end)     # get patch state for patch2 between start and end

    wheel1 = aeon.distancetravelled(encoder1.angle)                   # compute total distance travelled on patch1 wheel
    wheel2 = aeon.distancetravelled(encoder2.angle)                   # compute total distance travelled on patch2 wheel
    pellets1 = pellets1[pellets1.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch1
    pellets2 = pellets2[pellets2.event == 'TriggerPellet']            # get timestamps of pellets delivered at patch2

    frequency = 50                                                    # frame rate in Hz
    pixelscale = 0.00192                                              # 1 px = 1.92 mm
    positionrange = [[0,1440*pixelscale], [0,1080*pixelscale]]        # frame position range in metric units
    position = aeon.positiondata(root, start=start, end=end)          # get position data between start and end
    if start > pd.Timestamp('20210621') and \
       start < pd.Timestamp('20210701'):                              # time offset to account for abnormal drop event
        position.index = position.index + pd.Timedelta('22.57966s')   # exact offset extracted from video timestamps
    valid_position = (position.area > 0) & (position.area < 1000)     # filter for objects of the correct size
    position = position[valid_position]                               # get only valid positions
    position.x = position.x * pixelscale                              # scale position data to metric units
    position.y = position.y * pixelscale

    # compute ethogram based on distance to patches, nest and corridor
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
    in_patch1 = activepatch(wheel1, distp1 < patchradius)
    in_patch2 = activepatch(wheel2, distp2 < patchradius)
    in_arena = ~in_corridor & ~in_nest & ~in_patch1 & ~in_patch2
    ethogram = pd.Series('other', index=position.index)
    ethogram[in_corridor] = 'corridor'
    ethogram[in_nest] = 'nest'
    ethogram[in_patch1] = 'patch1'
    ethogram[in_patch2] = 'patch2'
    ethogram[in_arena] = 'arena'

    # plot heatmap of time spent in the arena
    fig, ax = plt.subplots(1, 1)
    heatmap(position, frequency, bins=500, range=positionrange, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.savefig('{0}/heatmap/{1}-heatmap.png'.format(output, prefix), dpi=dpi)
    plt.close(fig)

    # plot foraging trajectories overlaid on top of heatmap
    fig, ax = plt.subplots(1, 1)
    heatmap(position, frequency, bins=500, range=positionrange, ax=ax, alpha=0.5)
    circle(p1x, p1y, patchradius, 'b', linewidth=1, ax=ax)
    circle(p2x, p2y, patchradius, 'r', linewidth=1, ax=ax)
    forage = position.reindex(pellets1.index, method='pad')           # get position data when a pellet is delivered at patch1
    for trial in pellets1.itertuples():                               # for each pellet delivery
        before = trial.Index - pd.to_timedelta(15, 's')               # get the previous 10 seconds
        path = position.loc[before:trial.Index]                       # get position data in the time before pellet delivery
        colorline(path.x, path.y, cmap=plt.cm.Oranges, linewidth=1)   # plot path traces preceding pellet delivery
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    fig.savefig('{0}/forage/{1}-forage.png'.format(output, prefix), dpi=dpi)
    plt.close(fig)

    # plot patch activity summary including foraging rate, travelled distance and ethogram
    fig = plt.figure()
    rate_ax = fig.add_subplot(211)
    distance_ax = fig.add_subplot(212)
    ethogram_ax = fig.add_subplot(20,1,20)
    rateplot(pellets1,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color='b', label='Patch 1', ax=rate_ax)
    rateplot(pellets2,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color='r', label='Patch 2', ax=rate_ax)
    distance_ax.plot(sessiontime(wheel1.index), wheel1 / 100, 'b')  # plot position data as a path trajectory
    distance_ax.plot(sessiontime(wheel2.index), wheel2 / 100, 'r')  # plot position data as a path trajectory

    # plot vertical line indicating change of patch state, e.g. threshold
    change1 = state1[state1.threshold.diff().abs() > 0]
    change2 = state2[state2.threshold.diff().abs() > 0]
    change = pd.concat([change1, change2])
    if len(change) > 0:
        ymin, ymax = distance_ax.get_ylim()
        distance_ax.vlines(sessiontime(change.index, start), ymin, ymax, linewidth=1, color='k')

    # plot ethogram
    consecutive = (ethogram != ethogram.shift()).cumsum()
    ethogram_colors = {
        'patch1' : 'blue',
        'patch2' : 'red',
        'arena': 'green',
        'corridor' : 'black',
        'nest' : 'black' }
    ethogram_offsets = {
        'patch1' : [0,0.2],
        'patch2' : [0.2,0.2],
        'arena': [0.4,0.2],
        'corridor' : [0.6,0.2],
        'nest' : [0.6,0.2] }
    ethogram_ranges = ethogram.groupby(by=[ethogram, consecutive]).apply(lambda x:[
        sessiontime(x.index[0],start),
        sessiontime(x.index[-1],x.index[0])])
    for key,ranges in ethogram_ranges.groupby(level=0):
        color = ethogram_colors[key]
        offsets = ethogram_offsets[key]
        ethogram_ax.broken_barh(ranges,offsets,color=color)

    rate_ax.legend()
    rate_ax.sharex(distance_ax)
    rate_ax.tick_params(bottom=False, labelbottom=False)
    fig.subplots_adjust(hspace = 0.1)
    rate_ax.set_ylabel('pellets / min')
    rate_ax.set_title('foraging rate (bin size = 10 min)')
    distance_ax.set_xlabel('time (min)')
    distance_ax.set_ylabel('distance travelled (m)')
    set_ymargin(distance_ax, 0.2, 0.1)
    rate_ax.spines['top'].set_visible(False)
    rate_ax.spines['right'].set_visible(False)
    rate_ax.spines['bottom'].set_visible(False)
    distance_ax.spines['top'].set_visible(False)
    distance_ax.spines['right'].set_visible(False)
    ethogram_ax.set_axis_off()
    fig.savefig('{0}/ethogram/{1}-ethogram.png'.format(output, prefix), dpi=dpi)
    fig.savefig('{0}/ethogram-svg/{1}-ethogram.svg'.format(output, prefix), dpi=dpi)
    plt.close(fig)
    break