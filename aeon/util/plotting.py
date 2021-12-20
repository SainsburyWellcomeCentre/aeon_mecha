import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from aeon.analyze.patches import *

def heatmap(position, frequency, ax=None, **kwargs):
    '''
    Draw a heatmap of time spent in each location from specified position data and sampling frequency.

    :param Series position: A series of position data containing x and y coordinates.
    :param number frequency: The sampling frequency for the position data.
    :param Axes, optional ax: The Axes on which to draw the heatmap.
    '''
    if ax is None:
        ax = plt.gca()
    _, _, _, mesh = ax.hist2d(
        position.x, position.y,
        weights = np.ones(len(position)) / frequency,
        norm=colors.LogNorm(),
        **kwargs)
    ax.invert_yaxis()
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('time (s)')
    return mesh, cbar

def circle(x, y, radius, fmt=None, ax=None, **kwargs):
    '''
    Plot a circle centered at the given x, y position with the specified radius.

    :param number x: The x-component of the circle center.
    :param number y: The y-component of the circle center.
    :param number radius: The radius of the circle.
    :param str, optional fmt: The format used to plot the circle line.
    :param Axes, optional ax: The Axes on which to draw the circle.
    '''
    if ax is None:
        ax = plt.gca()
    points = pd.DataFrame(np.linspace(0,2 * math.pi, 360),columns=['angle'])
    points['x'] = radius * np.cos(points.angle) + x
    points['y'] = radius * np.sin(points.angle) + y
    ax.plot(points.x, points.y, fmt, **kwargs)

def rateplot(events, window, frequency, weight=1, start=None, end=None, smooth=None, center=True, ax=None, **kwargs):
    '''
    Plot the continuous event rate and raster of a discrete event sequence, given the specified
    window size and sampling frequency.

    :param Series events: The discrete sequence of events.
    :param offset window: The time period of each window used to compute the rate.
    :param DateOffset, Timedelta or str frequency: The sampling frequency for the continuous rate.
    :param number, optional weight: A weight used to scale the continuous rate of each window.
    :param datetime, optional start: The left bound of the time range for the continuous rate.
    :param datetime, optional end: The right bound of the time range for the continuous rate.
    :param datetime, optional smooth: The size of the smoothing kernel applied to the continuous rate output.
    :param DateOffset, Timedelta or str, optional smooth:
    The size of the smoothing kernel applied to the continuous rate output.
    :param bool, optional center: Specifies whether to center the convolution kernels.
    :param Axes, optional ax: The Axes on which to draw the rate plot and raster.
    '''
    label = kwargs.pop('label', None)
    eventrate = rate(events, window, frequency, weight, start, end, smooth=smooth, center=center)
    if ax is None:
        ax = plt.gca()
    ax.plot((eventrate.index-eventrate.index[0]).total_seconds() / 60, eventrate, label=label, **kwargs)
    ax.vlines(sessiontime(events.index, eventrate.index[0]), -0.2, -0.1, linewidth=1, **kwargs)

def set_ymargin(ax, bottom, top):
    '''
    Set the vertical margins of the specified Axes.

    :param Axes ax: The Axes for which to specify the vertical margin.
    :param number bottom: The size of the bottom margin.
    :param number top: The size of the top margins.
    '''
    ax.set_ymargin(0)
    ax.autoscale_view()
    ylim = ax.get_ylim()
    delta = ylim[1] - ylim[0]
    bottom = ylim[0] - delta * bottom
    top = ylim[1] + delta * top
    ax.set_ylim(bottom, top)

def colorline(x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), ax=None, **kwargs):
    '''
    Plot a dynamically colored line on the specified Axes.

    :param array-like x, y: The horizontal / vertical coordinates of the data points.
    :param array-like, optional z:
    The dynamic variable used to color each data point by indexing the color map.
    :param str or ~matplotlib.colors.Colormap, optional cmap:
    The colormap used to map normalized data values to RGBA colors.
    :param matplotlib.colors.Normalize, optional norm:
    The normalizing object used to scale data to the range [0, 1] for indexing the color map.
    :param Axes, optional ax: The Axes on which to draw the colored line.
    '''
    if ax is None:
        ax = plt.gca()
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    z = np.asarray(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, array=z, cmap=cmap, norm=norm, **kwargs)
    ax.add_collection(lines)
    return lines




def positionmap(position, positionrange, frequency=50, bins=500, title_str="", fig=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1)
    heatmap(position, frequency, bins=500, range=positionrange, ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title_str)
    ax.set_ylim([0.4, 1.8])
    ax.set_aspect('equal')
    return (fig, ax)

def plotWheelData():
    pellets1, pellets2, state1, state2, wheel1, wheel2 = D
    fig = plt.figure()
    rate_ax = fig.add_subplot(211)
    distance_ax = fig.add_subplot(212)
    ethogram_ax = fig.add_subplot(20,1,20)
    plorateplot(pellets1,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color='b', label='Patch 1', ax=rate_ax)
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
