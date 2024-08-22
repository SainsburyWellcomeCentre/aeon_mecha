import json
import pandas as pd
import aeon.io.reader as _reader
from aeon.schema.streams import Stream


class Pose(Stream):

    def __init__(self, path):
        super().__init__(_reader.Pose(f"{path}_202_*"))


class EnvActiveConfigReader(_reader.Reader):
    def __init__(self, pattern, columns=["name", "value"], extension="jsonl"):
        super().__init__(pattern, columns, extension)

    def read(self, file):
        """Reads data from the specified jsonl file."""
        with open(file, "r") as f:
            df = pd.read_json(f, lines=True)
        df["name"] = df["value"].apply(lambda x: x["name"])
        df["time"] = pd.to_datetime(df["seconds"], unit="s")
        df.set_index("time", inplace=True)
        return df[self.columns]


class ActiveConfiguration(Stream):

    def __init__(self, path):
        super().__init__(EnvActiveConfigReader(f"{path}_ActiveConfiguration_*"))
