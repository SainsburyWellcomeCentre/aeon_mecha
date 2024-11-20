"""Schema definition for social_01 experiments-specific data streams."""

import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class RfidEvents(Stream):
    def __init__(self, path):
        """Initializes the RfidEvents stream."""
        path = path.replace("Rfid", "")
        if path.startswith("Events"):
            path = path.replace("Events", "")

        super().__init__(_reader.Harp(f"RfidEvents{path}_32*", ["rfid"]))


class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        super().__init__(_reader.Pose(f"{path}_node-0*"))
