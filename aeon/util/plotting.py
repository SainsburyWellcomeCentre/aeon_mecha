import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from aeon.analyze.patches import *
from os import path

PATCH1_COLOR = 'b'
PATCH2_COLOR = 'r'

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

def positionTitle(session, start, end, t1, t2):
    pass


# def patchplot_v1(*
#     patchdf, start, end,
#      # The rest of the inputs need to be specified as keywords
#     savepath = None
#     ):
#     if savepath:
#         rate_ax = fig.add_subplot(211)
#         distance_ax = fig.add_subplot(212)
#         rateplot(pellets1,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color='b', label='Patch 1', ax=rate_ax)
#         rateplot(pellets2,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color='r', label='Patch 2', ax=rate_ax)
#         distance_ax.plot(sessiontime(wheel1.index), wheel1 / 100, 'b')  # plot position data as a path trajectory
#         distance_ax.plot(sessiontime(wheel2.index), wheel2 / 100, 'r')  # plot position data as a path trajectory
#         change1 = state1[state1.threshold.diff().abs() > 0]
#         change2 = state2[state2.threshold.diff().abs() > 0]
#         change = pd.concat([change1, change2])
#         if len(change) > 0:
#             ymin, ymax = distance_ax.get_ylim()
#             distance_ax.vlines(sessiontime(change.index, start), ymin, ymax, linewidth=1, color='k')


def positionmap(position, positionrange, frequency=50, bins=250, title_str="", fig=None, ax=None,**kwargs):
    """
    Input:
    =====
    position        a dataframe with x,y
    positionrange   sets the x and y lim (i think)
    frequency       [50Hz], the sampling rate of the position data
    bins            [500], the spatial resolution 
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)
    heatmap(position, frequency, bins=bins, range=positionrange, ax=ax,**kwargs)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title_str)
    ax.set_ylim([0.4, 1.8])
    ax.set_aspect('equal')
    return (fig, ax)

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



def plotFileName(fpath, plottype, meta, type='png'):
    sessid = meta['session'].id.split('/')[0].replace(';','.').replace(' ','')
    filename = f'{sessid}_{meta["session"].start:%m%d}_{plottype}.{type}'
    return path.join(fpath, filename)

def wheelTitle(meta):
    s = meta['session']
    return f"{s.id.split('/')[0]} {s.start:%m/%d %H:%M}"



def plotWheelData(*,
        pellets1, pellets2, state1, state2, wheel1, wheel2,
        savepath=None,
        meta=None,
        title_str=None,
        filename = None,
        force = False,
        ax = None,
        change_in_red = False,
        forceshow = None,
        total_dur = None,
        dpi=350,
        type='png'):
    """
    e.g. 
    `plotWheelData(helpers.getWheelData(root, start, end))`
    or 
    ```
    data = helpers.getWheelData(root, start, end)
    data['filename'] = 'path_to_figs/patch_2020.png'
    plotWheelData(**data)
    ```
    """
    if savepath and meta and not filename:
        filename = plotFileName(savepath,'patch',meta,type=type)

    if not(force) and filename and path.exists(filename):
        print(f'{filename} already exists. Set `force=True` to overwrite')
        if forceshow:
            img = mpimg.imread(filename)
            return plt.imshow(img)

    if not(forceshow) and filename:
        forceshow = False

    if not title_str:
        title_str = wheelTitle(meta)

    start = meta['session'].start
    end = meta['session'].end
    if not ax:
        fig = plt.figure()
        rate_ax = fig.add_subplot(211)
        distance_ax = fig.add_subplot(212)
    else:
        rate_ax, distance_ax = ax


    change1 = state1[state1.threshold.diff().abs() > 0]
    change2 = state2[state2.threshold.diff().abs() > 0]
    change = pd.concat([change1, change2])
    
    # if len(change) > 0 and total_dur:
    #     start = change.index - pd.DateOffset(minutes=(total_dur/2))
    #     end = change.index + pd.DateOffset(minutes=(total_dur/2))

    
    patch1_clr = PATCH1_COLOR
    patch2_clr = PATCH2_COLOR
    if len(change) > 0:
        if len(change1) > 0 and change_in_red:
            patch1_clr, patch2_clr = patch2_clr, patch1_clr
        
    if total_dur:
        end = start + pd.DateOffset(minutes=total_dur)

    for x in [wheel1, wheel2, pellets1, pellets2]:
        x.drop(x.loc[x.index > end].index, inplace=True)

    rateplot(pellets1,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color=patch1_clr, label='Patch 1', ax=rate_ax)
    rateplot(pellets2,'600s',frequency=500,weight=0.1,start=start,end=end,smooth='120s',color=patch2_clr, label='Patch 2', ax=rate_ax)
    distance_ax.plot(sessiontime(wheel1.index), wheel1 / 100, patch1_clr)  # plot position data as a path trajectory
    distance_ax.plot(sessiontime(wheel2.index), wheel2 / 100, patch2_clr)  # plot position data as a path trajectory

    # plot vertical line indicating change of patch state, e.g. threshold
    
    rate_ax.legend()
    rate_ax.sharex(distance_ax)
    rate_ax.tick_params(bottom=False, labelbottom=False)
    fig.subplots_adjust(hspace = 0.1)
    rate_ax.set_ylabel('pellets / min')
    rate_ax.set_title(title_str)
    distance_ax.set_xlabel('time (min)')
    distance_ax.set_ylabel('distance travelled (m)')
    set_ymargin(distance_ax, 0.2, 0.1)
    rate_ax.spines['top'].set_visible(False)
    rate_ax.spines['right'].set_visible(False)
    rate_ax.spines['bottom'].set_visible(False)
    distance_ax.spines['top'].set_visible(False)
    distance_ax.spines['right'].set_visible(False)
    
    if len(change) > 0:
        ymin, ymax = distance_ax.get_ylim()
        distance_ax.vlines(sessiontime(change.index, start), ymin, ymax, linewidth=1, color='k')


    fig.patch.set_facecolor('white')
    if filename:
        fig.savefig(filename, dpi=dpi)
    if forceshow:
        return fig
    else:
        plt.close(fig)
        return None