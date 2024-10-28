""" This module contains the schema for the social_03 dataset. """

import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        super().__init__(_reader.Pose(f"{path}_202_*"))


class EnvironmentActiveConfiguration(Stream):

    def __init__(self, path):
        """Initializes the EnvironmentActiveConfiguration stream."""
        super().__init__(
            _reader.JsonList(f"{path}_ActiveConfiguration_*", columns=["name"])
        )
