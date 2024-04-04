import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class RfidEvents(Stream):

    def __init__(self, path):
        path = path.replace("Rfid", "")
        if path.startswith("Events"):
            path = path.replace("Events", "")

        super().__init__(_reader.Harp(f"RfidEvents{path}_32*", ["rfid"]))
