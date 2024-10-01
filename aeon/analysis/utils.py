import numpy as np
import pandas as pd


def distancetravelled(angle, radius=4.0):
    """Calculates the total distance travelled on the wheel.

    Takes into account the wheel radius and the total number of turns in both directions across time.

    :param Series angle: A series of magnetic encoder measurements.
    :param float radius: The radius of the wheel, in metric units.
    :return: The total distance travelled on the wheel, in metric units.
    """
    maxvalue = int(np.iinfo(np.uint16).max >> 2)
    jumpthreshold = maxvalue // 2
    turns = angle.astype(int).diff()
    clickup = (turns < -jumpthreshold).astype(int)
    clickdown = (turns > jumpthreshold).astype(int) * -1
    turns = (clickup + clickdown).cumsum()
    distance = 2 * np.pi * radius * (turns + angle / maxvalue)
    distance = distance - distance[0]
    return distance


def visits(data, onset="Enter", offset="Exit"):
    """Computes duration, onset and offset times from paired events.

    Allows for missing data by trying to match event onset times with subsequent offset times.
    If the match fails, event offset metadata is filled with NaN. Any additional metadata columns
    in the data frame will be paired and included in the output.

    :param DataFrame data: A pandas data frame containing visit onset and offset events.
    :param str, optional onset: The label used to identify event onsets.
    :param str, optional offset: The label used to identify event offsets.
    :return: A pandas data frame containing duration and metadata for each visit.
    """
    lonset = onset.lower()
    loffset = offset.lower()
    lsuffix = f"_{lonset}"
    rsuffix = f"_{loffset}"
    id_onset = "id" + lsuffix
    event_onset = "event" + lsuffix
    event_offset = "event" + rsuffix
    time_onset = "time" + lsuffix
    time_offset = "time" + rsuffix

    # find all possible onset / offset pairs
    data = data.reset_index()
    data_onset = data[data.event == onset]
    data_offset = data[data.event == offset]
    data = pd.merge(data_onset, data_offset, on="id", how="left", suffixes=(lsuffix, rsuffix))

    # valid pairings have the smallest positive duration
    data["duration"] = data[time_offset] - data[time_onset]
    valid_visits = data[data.duration >= pd.Timedelta(0)]
    data = data.iloc[valid_visits.groupby([time_onset, "id"]).duration.idxmin()]
    data = data[data.duration > pd.Timedelta(0)]

    # duplicate offsets indicate missing data from previous pairing
    missing_data = data.duplicated(subset=time_offset, keep="last")
    if missing_data.any():
        data.loc[missing_data, ["duration"] + [name for name in data.columns if rsuffix in name]] = pd.NA

    # rename columns and sort data
    data.rename({time_onset: lonset, id_onset: "id", time_offset: loffset}, axis=1, inplace=True)
    data = data[["id"] + [name for name in data.columns if "_" in name] + [lonset, loffset, "duration"]]
    data.drop([event_onset, event_offset], axis=1, inplace=True)
    data.sort_index(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def rate(events, window, frequency, weight=1, start=None, end=None, smooth=None, center=False):
    """Computes the continuous event rate from a discrete event sequence.

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
    :return: A Series containing the continuous event rate over time.
    """
    counts = pd.Series(weight, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts = counts.resample(pd.Timedelta(1 / frequency, "s")).sum()
    rate = counts.rolling(window, center=center).sum()
    return rate.rolling(window if smooth is None else smooth, center=center).mean()


def get_events_rates(
    events, window_len_sec, frequency, unit_len_sec=60, start=None, end=None, smooth=None, center=False
):
    """Computes the event rate from a sequence of events over a specified window."""
    # events is an array with the time (in seconds) of event occurence
    # window_len_sec is the size of the window over which the event rate is estimated
    # unit_len_sec is the length of one sample point
    window_len_sec_str = f"{window_len_sec:d}S"
    counts = pd.Series(1.0, events.index)
    if start is not None and start < events.index[0]:
        counts.loc[start] = 0
    if end is not None and end > events.index[-1]:
        counts.loc[end] = 0
    counts.sort_index(inplace=True)
    counts_resampled = counts.resample(frequency).sum()
    counts_rolled = (
        counts_resampled.rolling(window_len_sec_str, center=center).sum() * unit_len_sec / window_len_sec
    )
    counts_rolled_smoothed = counts_rolled.rolling(
        window_len_sec_str if smooth is None else smooth, center=center
    ).mean()
    return counts_rolled_smoothed


def sessiontime(index, start=None):
    """Converts absolute to relative time, with optional reference starting time."""
    if start is None:
        start = index[0]
    return (index - start).total_seconds() / 60


def distance(position, target):
    """Computes the euclidean distance to a specified target."""
    return np.sqrt(np.square(position[["x", "y"]] - target).sum(axis=1))


def activepatch(wheel, in_patch):
    """Computes a decision boundary for when a patch is active based on wheel movement.

    :param Series wheel: A pandas Series containing the cumulative distance travelled on the wheel.
    :param Series in_patch: A Series of type bool containing whether the specified patch may be active.
    :return: A pandas Series specifying for each timepoint whether the patch is active.
    """
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel.diff().rolling("1s").sum() > 1).reindex(in_patch.index, method="pad")
    epochs = exit_patch.cumsum()
    return in_wheel.groupby(epochs).apply(lambda x: x.cumsum()) > 0
