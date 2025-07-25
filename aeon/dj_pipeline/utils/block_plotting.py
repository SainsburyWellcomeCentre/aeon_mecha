"""Helper functions for plotting block data."""

from colorsys import hls_to_rgb, rgb_to_hls

import numpy as np
import plotly
from numpy.lib.stride_tricks import as_strided

"""Standardize subject colors, patch colors, and markers."""

subject_colors = plotly.colors.qualitative.Plotly
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


def gen_hex_grad(hex_col, vals, min_lightness=0.3):
    """Generates an array of hex color values based on a gradient defined by unit-normalized values."""
    # Convert hex to rgb to hls
    hue, lightness, saturation = rgb_to_hls(
        *[int(hex_col.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    )
    grad = np.empty(shape=(len(vals),), dtype="<U10")  # init grad
    for i, val in enumerate(vals):
        cur_lightness = (lightness * val) + (
            min_lightness * (1 - val)
        )  # get cur lightness relative to `hex_col`
        cur_lightness = max(min(cur_lightness, lightness), min_lightness)  # set min, max bounds
        cur_rgb_col = hls_to_rgb(hue, cur_lightness, saturation)  # convert to rgb
        cur_hex_col = "#{:02x}{:02x}{:02x}".format(
            *tuple(int(c * 255) for c in cur_rgb_col)
        )  # convert to hex
        grad[i] = cur_hex_col

    return grad


def conv2d(arr, kernel):
    """Performs "valid" 2d convolution using numpy `as_strided` and `einsum`."""
    out_shape = tuple(np.subtract(arr.shape, kernel.shape) + 1)
    sub_mat_shape = kernel.shape + out_shape
    # Create "new view" of `arr` as submatrices at which kernel will be applied
    sub_mats = as_strided(arr, shape=sub_mat_shape, strides=(arr.strides * 2))
    out = np.einsum("ij, ijkl -> kl", kernel, sub_mats)
    return out


def gen_subject_colors_dict(subject_names):
    """Generates a dictionary of subject colors based on a list of subjects."""
    return dict(zip(subject_names, subject_colors, strict=False))


def gen_patch_style_dict(patch_names):
    """Generates a dictionary of patch styles given a list of patch_names.

    The dictionary contains dictionaries which map patch names to their respective styles.
    Below are the keys for each nested dictionary and their contents:

    - colors: patch name to color
    - markers: patch name to marker
    - symbols: patch name to symbol
    - linestyles: patch name to linestyle
    """
    return {
        "colors": dict(zip(patch_names, patch_colors, strict=False)),
        "markers": dict(zip(patch_names, patch_markers, strict=False)),
        "symbols": dict(zip(patch_names, patch_markers_symbols, strict=False)),
        "linestyles": dict(zip(patch_names, patch_markers_linestyles, strict=False)),
    }
