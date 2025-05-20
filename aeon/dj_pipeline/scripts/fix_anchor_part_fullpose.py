"""Script to fix the anchor part of the fullpose SLEAP entries.

See this commit: https://github.com/SainsburyWellcomeCentre/aeon_mecha/commit/8358ce4b6923918920efb77d09adc769721dbb9b

Last run: ---
"""

import pandas as pd
from tqdm import tqdm

from aeon.dj_pipeline import acquisition, streams, tracking

aeon_schemas = acquisition.aeon_schemas
logger = acquisition.logger
io_api = acquisition.io_api


def update_anchor_part(key):
    """Update the anchor part of the fullpose SLEAP entries for one SLEAPTracking `key`."""
    chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

    data_dirs = acquisition.Experiment.get_data_directories(key)

    device_name = (streams.SpinnakerVideoSource & key).fetch1("spinnaker_video_source_name")

    devices_schema = getattr(
        aeon_schemas,
        (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
            "devices_schema_name"
        ),
    )

    stream_reader = getattr(devices_schema, device_name).Pose

    # special ingestion case for social0.2 full-pose data (using Pose reader from social03)
    # fullpose for social0.2 has a different "pattern" for non-fullpose, hence the Pose03 reader
    if key["experiment_name"].startswith("social0.2"):
        from swc.aeon.io import reader as io_reader

        stream_reader = getattr(devices_schema, device_name).Pose03
        if not isinstance(stream_reader, io_reader.Pose):
            raise TypeError("Pose03 is not a Pose reader")
        data_dirs = [acquisition.Experiment.get_data_directory(key, "processed")]

    pose_data = io_api.load(
        root=data_dirs,
        reader=stream_reader,
        start=pd.Timestamp(chunk_start),
        end=pd.Timestamp(chunk_end),
    )

    if not len(pose_data):
        raise ValueError(f"No SLEAP data found for {key['experiment_name']} - {device_name}")

    # get anchor part
    anchor_part = next(v.replace("_x", "") for v in stream_reader.columns if v.endswith("_x"))

    # update anchor part
    for entry in tracking.SLEAPTracking.PoseIdentity.fetch("KEY"):
        entry["anchor_part"] = anchor_part
        tracking.SLEAPTracking.PoseIdentity.update1(entry)

    logger.info(f"Anchor part updated to {anchor_part} for {key}")


def main():
    """Calling `update_anchor_part` for all SLEAPTracking entries."""
    keys = tracking.SLEAPTracking.fetch("KEY")
    for key in tqdm(keys):
        update_anchor_part(key)
