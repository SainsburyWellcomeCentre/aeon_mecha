"""Schema definition for social_01 experiments-specific data streams."""

from swc.aeon.io import reader
from swc.aeon.schema import Stream


class RfidEvents(Stream):
    def __init__(self, path):
        """Initializes the RfidEvents stream."""
        path = path.replace("Rfid", "")
        if path.startswith("Events"):
            path = path.replace("Events", "")

        super().__init__(reader.Harp(f"RfidEvents{path}_32*", ["rfid"]))


class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        super().__init__(reader.Pose(f"{path}_node-0*", "/ceph/aeon/aeon/data/processed"))
