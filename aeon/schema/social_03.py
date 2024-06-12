import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class Pose(Stream):

    def __init__(self, path):
        super().__init__(_reader.Pose(f"{path}_202_*"))
