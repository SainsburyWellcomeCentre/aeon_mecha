"""Preprocesses aeon data. Typically called on data loaded from `aeon.io.api.load()`"""
import numpy as np
from dotmap import DotMap


HARP_EVENT_BITMASK = DotMap(
    {
        "pellet_trigger": 0x80,
        "pellet_detected_in": 0x20,
        "pellet_detected_out": 0x22,
    }
)


def apply_bitmask(df, bitmask):
    """Filters a dataframe (`df`) by a Harp bitmask (`bitmask`)"""
    return df.iloc[np.where(df == bitmask)[0]]


def calc_wheel_cum_dist(angle, enc_res=2**14, radius=0.04):
    """Calculates cumulative wheel move distance from a Series containing encoder angle
    data (`angle`), the encoder bit res (`enc_res`), and wheel radius in m (`radius`)"""
    # Algo: Compute number of wheel turns (overflows - underflows) cumulatively at
    # each datapoint, then use this to compute cum dist at each datapoint
    jump_thresh = enc_res // 2  # if diff in data > jump_thresh, assume over/underflow
    angle_diff = angle.diff()
    overflow = (angle_diff < -jump_thresh).astype(int)
    underflow = (angle_diff > jump_thresh).astype(int)
    turns = (overflow - underflow).cumsum()
    # cum_dist = circumference of wheel * fractional number of turns
    cum_dist = 2 * np.pi * radius * (turns + (angle / (enc_res - 1)))
    return cum_dist - cum_dist[0]
