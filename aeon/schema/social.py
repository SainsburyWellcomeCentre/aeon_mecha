"""Readers for data relevant to Social experiments."""

import json

import aeon.io.reader as _reader


class Sleap(_reader.Harp):
    def __init__(self, model_path):
        """Initializes a reader for Harp-binarized Sleap tracking data given a Sleap network model path."""
        columns = ["class", "class_likelihood"]
        # @todo Get bodyparts from Sleap config.
        bodyparts = []
        for part in bodyparts:
            columns.extend([f"{part}_x", f"{part}_y", f"{part}_likelihood"])
        super().__init__(pattern="", columns=columns)
        