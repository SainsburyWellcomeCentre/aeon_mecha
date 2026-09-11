import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

from swc.aeon.io import reader as _reader
from swc.aeon.schema.streams import Stream, StreamGroup


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
            
            # ONIX records the raw seconds value arriving from the HARP master
            # clock as "Value.HarpTime". In the harp world, this timestamp 
            # corresponds to the second that is about to end in less than a
            # millisecond, therefore it is lagging by 1 second.
            # The index colums (Seconds) is the actual time that corresponds most
            # closely to the actual harp time when the ONIX clock was recorded
            harp_time = data.index.values.reshape(-1, 1)

            model = LinearRegression().fit(onix_clock, harp_time)
            r2 = model.score(onix_clock, harp_time)
            # Chunk timestamp lives in the filename suffix. Two conventions
            # coexist: the compact UTC form ("...T090000Z.csv", emitted by the
            # current acquisition software) and the older dashed form
            # ("...T09-00-00.csv", still used by the test fixtures).
            chunk_info = file.name.split("_")[-1].removesuffix(".csv")
            for _fmt in ("%Y-%m-%dT%H%M%SZ", "%Y-%m-%dT%H-%M-%S"):
                try:
                    epoch = datetime.strptime(chunk_info, _fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(
                    f"Unrecognized HarpSync chunk timestamp: {chunk_info!r}"
                )
            return pd.DataFrame(
                index=[epoch],
                data={
                    "clock_start": onix_clock[0],
                    "clock_end": onix_clock[-1],
                    "harp_start": harp_time[0],
                    "harp_end": harp_time[-1],
                    "n_samples": len(data),
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

    class Bno055Euler(Stream):
        def __init__(self, pattern):
            super().__init__(Binary(f"{pattern}_Bno055_Euler_*", dtype=np.float32, columns=['x', 'y', 'z']))

    class Bno055GravityVector(Stream):
        def __init__(self, pattern):
            super().__init__(Binary(f"{pattern}_Bno055_GravityVector_*", dtype=np.float32, columns=['x', 'y', 'z']))

    class Bno055LinearAcceleration(Stream):
        def __init__(self, pattern):
            super().__init__(Binary(f"{pattern}_Bno055_LinearAcceleration_*", dtype=np.float32, columns=['x', 'y', 'z']))

    class Bno055Quaternion(Stream):
        def __init__(self, pattern):
            super().__init__(Binary(f"{pattern}_Bno055_Quaternion_*", dtype=np.float32, columns=['w', 'x', 'y', 'z']))


class NeuropixelsV2Beta(StreamGroup):
    def __init__(self, path):
        super().__init__(path, HarpSync, HarpSyncModel)


class NeuropixelsV2(StreamGroup):
    def __init__(self, path):
        super().__init__(path, HarpSync, HarpSyncModel)

# ----

from dotmap import DotMap
from swc.aeon.schema.streams import Device

social_ephys = DotMap(
    [
        Device("NeuropixelsV2Beta", NeuropixelsV2Beta, Bno055),
        Device("NeuropixelsV2", NeuropixelsV2, Bno055),
    ])
