import datajoint as dj
import pandas as pd
import numpy as np

from aeon.io import api as aeon_api
from aeon.util import plotting as aeon_plotting

from . import acquisition
from . import get_schema_name


schema = dj.schema(get_schema_name('qc'))

# -------------- Quality Control ---------------------


@schema
class QCCode(dj.Lookup):
    definition = """
    qc_code: int
    ---
    qc_code_description: varchar(255)
    """


@schema
class QCRoutine(dj.Lookup):
    definition = """
    qc_routine: varchar(24)  # name of this quality control evaluation - e.g. drop_frame
    ---
    qc_routine_order: int    # the order in which this qc routine is executed
    qc_routine_description: varchar(255)  # description of this QC routine
    qc_module: varchar(64)     # the module where the qc_function can be imported from - e.g. aeon.analysis.quality_control
    qc_function: varchar(64)   # the function used to evaluate this QC - e.g. check_drop_frame
    """


# -------------- Data stream level Quality Control ---------------------


@schema
class CameraQC(dj.Imported):
    definition = """ # Quality controls performed on a particular camera for a particular acquisition chunk
    -> acquisition.Chunk
    -> acquisition.ExperimentCamera
    ---
    drop_count=null: int
    max_harp_delta: float    # (s)
    max_camera_delta: float  # (s)
    timestamps: longblob
    time_delta: longblob
    frame_delta: longblob
    hw_counter_delta: longblob
    hw_timestamp_delta: longblob
    frame_offset: longblob
    """

    @property
    def key_source(self):
        return (acquisition.Chunk
                * acquisition.ExperimentCamera.join(acquisition.ExperimentCamera.RemovalTime, left=True)
                & 'chunk_start >= camera_install_time'
                & 'chunk_start < IFNULL(camera_remove_time, "2200-01-01")')

    def make(self, key):
        qc_dir = acquisition.Experiment.get_data_directory(
            key, directory_type='quality-control', as_posix=True)

        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1('chunk_start', 'chunk_end')

        camera = (acquisition.ExperimentCamera & key).fetch1('camera_description')

        deltas = aeon_api.load(qc_dir, lambda f: pd.read_parquet(f),
                               camera, extension='*.parquet',
                               start=pd.Timestamp(chunk_start),
                               end=pd.Timestamp(chunk_end))

        self.insert1({**key,
                      'drop_count': deltas.frame_offset.iloc[-1],
                      'max_harp_delta': deltas.time_delta.max().total_seconds(),
                      'max_camera_delta': deltas.hw_timestamp_delta.max() / 1e9,  # convert to seconds
                      'timestamps': deltas.index.to_pydatetime(),
                      'time_delta': deltas.time_delta.values / np.timedelta64(1, 's'), # convert to seconds
                      'frame_delta': deltas.frame_delta.values,
                      'hw_counter_delta': deltas.hw_counter_delta.values,
                      'hw_timestamp_delta': deltas.hw_timestamp_delta.values,
                      'frame_offset': deltas.frame_offset.values})

