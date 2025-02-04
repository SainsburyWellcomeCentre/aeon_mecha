"""DataJoint schema for the quality control pipeline."""

import datajoint as dj
import numpy as np
import pandas as pd

from aeon.dj_pipeline import acquisition, get_schema_name, streams
from swc.aeon.io import api as io_api

schema = dj.schema(get_schema_name("qc"))
logger = dj.logger

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
    qc_module: varchar(64)     # module path, e.g., aeon.analysis.quality_control
    qc_function: varchar(64)   # the function used to evaluate this QC - e.g. check_drop_frame
    """


# -------------- Data stream level Quality Control ---------------------


@schema
class CameraQC(dj.Imported):
    definition = """ # Quality controls performed on a particular camera for one acquisition chunk
    -> acquisition.Chunk
    -> streams.SpinnakerVideoSource
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
        """Return the keys for the CameraQC table."""
        return (
            acquisition.Chunk
            * (
                streams.SpinnakerVideoSource.join(streams.SpinnakerVideoSource.RemovalTime, left=True)
                & "spinnaker_video_source_name='CameraTop'"
            )
            & "chunk_start >= spinnaker_video_source_install_time"
            & 'chunk_start < IFNULL(spinnaker_video_source_removal_time, "2200-01-01")'
        )  # CameraTop

    def make(self, key):
        """Perform quality control checks on the CameraTop stream."""
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

        device_name = (streams.SpinnakerVideoSource & key).fetch1("spinnaker_video_source_name")
        data_dirs = acquisition.Experiment.get_data_directories(key)

        devices_schema = getattr(
            acquisition.aeon_schemas,
            (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                "devices_schema_name"
            ),
        )
        stream_reader = getattr(devices_schema, device_name).Video

        videodata = io_api.load(
            root=data_dirs,
            reader=stream_reader,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        ).reset_index()

        deltas = videodata[videodata.columns[0:4]].diff()
        deltas.columns = [
            "time_delta",
            "hw_counter_delta",
            "hw_timestamp_delta",
            "frame_delta",
        ]
        deltas["frame_offset"] = (deltas.hw_counter_delta - 1).cumsum()

        videodata.set_index("time", inplace=True)

        self.insert1(
            {
                **key,
                "drop_count": deltas.frame_offset.iloc[-1],
                "max_harp_delta": deltas.time_delta.max().total_seconds(),
                "max_camera_delta": deltas.hw_timestamp_delta.max() / 1e9,  # convert to seconds
                "timestamps": videodata.index.values,
                "time_delta": deltas.time_delta.values / np.timedelta64(1, "s"),  # convert to seconds
                "frame_delta": deltas.frame_delta.values,
                "hw_counter_delta": deltas.hw_counter_delta.values,
                "hw_timestamp_delta": deltas.hw_timestamp_delta.values,
                "frame_offset": deltas.frame_offset.values,
            }
        )
