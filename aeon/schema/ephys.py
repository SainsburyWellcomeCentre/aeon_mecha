import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

from swc.aeon.io import reader as _reader
from swc.aeon.schema import Stream, StreamGroup


# -- Ephys streams for HarpSync and OnixClock data

class Binary(_reader.Reader):
    """Extracts data from raw flat binary files without timestamp information."""

    def __init__(self, pattern, dtype, columns, extension="bin"):
        super().__init__(pattern, columns, extension)
        self.dtype = dtype

    def read(self, file):
        """Reads data from the specified flat binary file."""
        data = np.fromfile(file, dtype=self.dtype)
        data = data.reshape((-1, len(self.columns)))
        return pd.DataFrame(data, columns=self.columns)


class HarpSync(Stream):
    class Reader(_reader.Csv):
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
        super().__init__(Binary(pattern, dtype=np.uint64, columns=["clock"]))


class Bno055(StreamGroup):
    def __init__(self, path):
        super().__init__(path)

    class Bno055Clock(OnixClock):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_Bno055_Clock_*")


class NeuropixelsV2Beta(StreamGroup):
    def __init__(self, path):
        super().__init__(path, HarpSync, HarpSyncModel)

# ----

from dotmap import DotMap
from swc.aeon.schema import Device

social_ephys = DotMap(
    [
        Device("NeuropixelsV2Beta", NeuropixelsV2Beta, Bno055)
    ])
