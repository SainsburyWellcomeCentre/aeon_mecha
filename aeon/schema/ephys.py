import numpy as np
import pandas as pd
from datetime import datetime
from aeon.schema.streams import Stream, StreamGroup
from sklearn.linear_model import LinearRegression
from aeon.io.reader import Csv
import aeon.io.reader as _reader


class HarpSync(Stream):
    class Reader(Csv):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_HarpSync_*", columns=["clock", "hub_clock", "harp_time"])

    def __init__(self, pattern):
        super().__init__(HarpSync.Reader(pattern))


class HarpSyncModel(Stream):
    class Reader(HarpSync.Reader):
        def __init__(self, pattern):
            super().__init__(pattern)

        def read(self, file):
            data = super().read(file).dropna()
            onix_clock = data.clock.values.reshape(-1, 1)
            harp_time = data.harp_time.values.reshape(-1, 1)

            model = LinearRegression().fit(onix_clock, harp_time)
            r2 = model.score(onix_clock, harp_time)
            chunk_info = file.name.split("_")[-1]
            epoch = datetime.strptime(chunk_info, "%Y-%m-%dT%H-%M-%S.csv")
            return pd.DataFrame(
                index=[epoch],
                data={
                    "clock_start": onix_clock[0],
                    "clock_end": onix_clock[-1],
                    "model": [model],
                    "r2": [r2],
                },
            )

    def __init__(self, pattern):
        super().__init__(HarpSyncModel.Reader(pattern))


class OnixClock(Stream):
    def __init__(self, pattern):
        super().__init__(_reader.Binary(pattern, dtype=np.uint64, columns=["clock"]))


class Bno055(StreamGroup):
    def __init__(self, path):
        super().__init__(path)

    class Bno055Clock(OnixClock):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_Bno055_Clock_*")


class NeuropixelsV2Beta(StreamGroup):
    def __init__(self, path):
        super().__init__(path, HarpSync, HarpSyncModel)
