"""Schema definition for social_03 experiments-specific data streams."""

from swc.aeon.io import reader
from swc.aeon.schema.streams import Stream

class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        super().__init__(reader.Pose(f"{path}_202_*"))


class EnvironmentActiveConfiguration(Stream):
    def __init__(self, path):
        """Initializes the EnvironmentActiveConfiguration stream."""
        super().__init__(reader.JsonList(f"{path}_ActiveConfiguration_*", columns=["name"]))
