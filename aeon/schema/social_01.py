""" This module contains the schema for the social_01 dataset. """

import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class RfidEvents(Stream):
    def __init__(self, path):
        path = path.replace("Rfid", "")
        if path.startswith("Events"):
            path = path.replace("Events", "")

        super().__init__(_reader.Harp(f"RfidEvents{path}_32*", ["rfid"]))


class Pose(Stream):
    def __init__(self, path):
        super().__init__(_reader.Pose(f"{path}_node-0*"))
