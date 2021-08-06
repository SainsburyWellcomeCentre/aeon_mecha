import numpy as np
import pandas as pd

def rate(events, window, frequency, weight=1, start=None, end=None, smooth=None, center=False):
    '''
    Computes the continuous event rate from a discrete event sequence, given the specified
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
    :return: A Series containing the continuous event rate over time.
    '''
    counts = pd.Series(weight, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts = counts.resample(pd.Timedelta(1 / frequency, 's')).sum()
    rate = counts.rolling(window,center=center).sum()
    return rate.rolling(window if smooth is None else smooth,center=center).mean()

def sessiontime(index, start=None):
    """Converts absolute to relative time, with optional reference starting time."""
    if (start is None):
        start = index[0]
    return (index-start).total_seconds() / 60

def distance(position, target):
    """Computes the euclidean distance to a specified target."""
    return np.sqrt(np.square(position[['x','y']] - target).sum(axis=1))

def activepatch(wheel, in_patch):
    '''
    Computes a decision boundary for when a patch is active based on wheel movement.
    
    :param Series wheel: A pandas Series containing the cumulative distance travelled on the wheel.
    :param Series in_patch: A Series of type bool containing whether the specified patch may be active.
    :return: A pandas Series specifying for each timepoint whether the patch is active.
    '''
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel.diff().rolling('1s').sum() > 1).reindex(in_patch.index, method='pad')
    epochs = exit_patch.cumsum()
    return in_wheel.groupby(epochs).apply(lambda x:x.cumsum()) > 0