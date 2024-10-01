import json
import pandas as pd
import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class Pose(Stream):
    def __init__(self, path):
        super().__init__(_reader.Pose(f"{path}_202_*"))


class EnvironmentActiveConfiguration(Stream):

    def __init__(self, path):
        super().__init__(_reader.JsonList(f"{path}_ActiveConfiguration_*", columns=["name"]))
