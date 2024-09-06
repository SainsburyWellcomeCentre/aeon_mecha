import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.collections import LineCollection

from aeon.analysis.utils import rate, sessiontime


def heatmap(position, frequency, ax=None, **kwargs):
    """Draw a heatmap of time spent in each location from specified position data and sampling frequency.

    :param Series position: A series of position data containing x and y coordinates.
    :param number frequency: The sampling frequency for the position data.
    :param Axes, optional ax: The Axes on which to draw the heatmap.
    """
    if ax is None:
        ax = plt.gca()
    _, _, _, mesh = ax.hist2d(
        position.x,
        position.y,
        weights=np.ones(len(position)) / frequency,
        norm=colors.LogNorm(),
        **kwargs,
    )
    ax.invert_yaxis()
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("time (s)")
    return mesh, cbar


def circle(x, y, radius, fmt=None, ax=None, **kwargs):
    """Plot a circle centered at the given x, y position with the specified radius.

    :param number x: The x-component of the circle center.
    :param number y: The y-component of the circle center.
    :param number radius: The radius of the circle.
    :param str, optional fmt: The format used to plot the circle line.
    :param Axes, optional ax: The Axes on which to draw the circle.
    """
    if ax is None:
        ax = plt.gca()
    points = pd.DataFrame(np.linspace(0, 2 * math.pi, 360), columns=["angle"])
    points["x"] = radius * np.cos(points.angle) + x
    points["y"] = radius * np.sin(points.angle) + y
    ax.plot(points.x, points.y, fmt, **kwargs)


def rateplot(
    events,
    window,
    frequency,
    weight=1,
    start=None,
    end=None,
    smooth=None,
    center=True,
    ax=None,
    **kwargs,
):
    """Plot the continuous event rate and raster of a discrete event sequence.

    The window size and sampling frequency can be specified.

    :param Series events: The discrete sequence of events.
    :param offset window: The time period of each window used to compute the rate.
    :param DateOffset, Timedelta or str frequency: The sampling frequency for the continuous rate.
    :param number, optional weight: A weight used to scale the continuous rate of each window.
    :param datetime, optional start: The left bound of the time range for the continuous rate.
    :param datetime, optional end: The right bound of the time range for the continuous rate.
    :param datetime, optional smooth: The size of the smoothing kernel applied to the rate output.
    :param DateOffset, Timedelta or str, optional smooth:
    The size of the smoothing kernel applied to the continuous rate output.
    :param bool, optional center: Specifies whether to center the convolution kernels.
    :param Axes, optional ax: The Axes on which to draw the rate plot and raster.
    """
    label = kwargs.pop("label", None)
    eventrate = rate(events, window, frequency, weight, start, end, smooth=smooth, center=center)
    if ax is None:
        ax = plt.gca()
    ax.plot(
        (eventrate.index - eventrate.index[0]).total_seconds() / 60,
        eventrate,
        label=label,
        **kwargs,
    )
    ax.vlines(sessiontime(events.index, eventrate.index[0]), -0.2, -0.1, linewidth=1, **kwargs)


def set_ymargin(ax, bottom, top):
    """Set the vertical margins of the specified Axes.

    :param Axes ax: The Axes for which to specify the vertical margin.
    :param number bottom: The size of the bottom margin.
    :param number top: The size of the top margins.
    """
    ax.set_ymargin(0)
    ax.autoscale_view()
    ylim = ax.get_ylim()
    delta = ylim[1] - ylim[0]
    bottom = ylim[0] - delta * bottom
    top = ylim[1] + delta * top
    ax.set_ylim(bottom, top)


def colorline(
    x,
    y,
    z=None,
    cmap=None,
    norm=None,
    ax=None,
    **kwargs,
):
    """Plot a dynamically colored line on the specified Axes.

    :param array-like x, y: The horizontal / vertical coordinates of the data points.
    :param array-like, optional z:
    The dynamic variable used to color each data point by indexing the color map.
    :param str or ~matplotlib.colors.Colormap, optional cmap:
    The colormap used to map normalized data values to RGBA colors.
    :param matplotlib.colors.Normalize, optional norm:
    The normalizing object used to scale data to the range [0, 1] for indexing the color map.
    :param Axes, optional ax: The Axes on which to draw the colored line.
    """
    if ax is None:
        ax = plt.gca()
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    if cmap is None:
        cmap = plt.get_cmap("copper")
    if norm is None:
        norm = plt.Normalize(0.0, 1.0)
    z = np.asarray(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, array=z, cmap=cmap, norm=norm, **kwargs)
    ax.add_collection(lines)
    return lines
