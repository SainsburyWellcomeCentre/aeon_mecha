from colorsys import hls_to_rgb, rgb_to_hls

import numpy as np
import plotly

"""Standardize subject colors, patch colors, and markers."""

subject_colors = plotly.colors.qualitative.Plotly
subject_colors_dict = {
    "BAA-1104045": subject_colors[0],
    "BAA-1104047": subject_colors[1],
    "BAA-1104048": subject_colors[2],
    "BAA-1104049": subject_colors[3],
}
patch_colors = plotly.colors.qualitative.Dark2
patch_markers = [
    "circle",
    "bowtie",
    "square",
    "hourglass",
    "diamond",
    "cross",
    "x",
    "triangle",
    "star",
]
patch_markers_symbols = ["●", "⧓", "■", "⧗", "♦", "✖", "×", "▲", "★"]
patch_markers_dict = dict(zip(patch_markers, patch_markers_symbols, strict=False))
patch_markers_linestyles = ["solid", "dash", "dot", "dashdot", "longdashdot"]


def gen_hex_grad(hex_col, vals, min_l=0.3):
    """Generates an array of hex color values based on a gradient defined by unit-normalized values."""
    # Convert hex to rgb to hls
    h, l, s = rgb_to_hls(*[int(hex_col.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)])
    grad = np.empty(shape=(len(vals),), dtype="<U10")  # init grad
    for i, val in enumerate(vals):
        cur_l = (l * val) + (min_l * (1 - val))  # get cur lightness relative to `hex_col`
        cur_l = max(min(cur_l, l), min_l)  # set min, max bounds
        cur_rgb_col = hls_to_rgb(h, cur_l, s)  # convert to rgb
        cur_hex_col = "#{:02x}{:02x}{:02x}".format(
            *tuple(int(c * 255) for c in cur_rgb_col)
        )  # convert to hex
        grad[i] = cur_hex_col

    return grad
