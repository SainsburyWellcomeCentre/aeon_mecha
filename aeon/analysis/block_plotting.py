import os
import pathlib
from colorsys import hls_to_rgb, rgb_to_hls

import numpy as np
from numpy.lib.stride_tricks import as_strided


def gen_hex_grad(hex_col, vals, min_l=0.3):
    """Generates an array of hex color values based on a gradient defined by unit-normalized values."""
    # Convert hex to rgb to hls
    h, l, s = rgb_to_hls(
        *[int(hex_col.lstrip("#")[i: i + 2], 16) / 255 for i in (0, 2, 4)]
    )
    grad = np.empty(shape=(len(vals),), dtype="<U10")  # init grad
    for i, val in enumerate(vals):
        cur_l = (l * val) + (
            min_l * (1 - val)
        )  # get cur lightness relative to `hex_col`
        cur_l = max(min(cur_l, l), min_l)  # set min, max bounds
        cur_rgb_col = hls_to_rgb(h, cur_l, s)  # convert to rgb
        cur_hex_col = "#%02x%02x%02x" % tuple(
            int(c * 255) for c in cur_rgb_col
        )  # convert to hex
        grad[i] = cur_hex_col

    return grad
