"""Functions to find and delete orphaned epochs that have been ingested but are no longer valid."""

from datetime import datetime

from aeon.dj_pipeline import acquisition, tracking

aeon_schemas = acquisition.aeon_schemas
logger = acquisition.logger

exp_key = {"experiment_name": "social0.2-aeon4"}


def find_chunks_to_reingest(exp_key, delete_not_fullpose=False):
    """Find chunks with newly available full pose data to reingest.

    If available, fullpose data can be found in `processed` folder
    """
    device_name = "CameraTop"

    devices_schema = getattr(
        aeon_schemas,
        (acquisition.Experiment.DevicesSchema & {"experiment_name": exp_key["experiment_name"]}).fetch1(
            "devices_schema_name"
        ),
    )
    stream_reader = getattr(devices_schema, device_name).Pose

    # special ingestion case for social0.2 full-pose data (using Pose reader from social03)
    if exp_key["experiment_name"].startswith("social0.2"):
        from swc.aeon.io import reader as io_reader
        stream_reader = getattr(devices_schema, device_name).Pose03
        if not isinstance(stream_reader, io_reader.Pose):
            raise TypeError("Pose03 is not a Pose reader")

    # find processed path for exp_key
    processed_dir = acquisition.Experiment.get_data_directory(exp_key, "processed")

    files = sorted(f.stem for f in processed_dir.rglob(f"{stream_reader.pattern}.bin") if f.is_file())
    # extract timestamps from the file names & convert to datetime
    file_times = [datetime.strptime(f.split("_")[-1], "%Y-%m-%dT%H-%M-%S") for f in files]

    # sleap query with files in processed dir
    query = acquisition.Chunk & exp_key & [{"chunk_start": t} for t in file_times]
    epochs = acquisition.Epoch & query.proj("epoch_start")
    sleap_query = tracking.SLEAPTracking & (acquisition.Chunk & epochs.proj("epoch_start"))

    fullpose, not_fullpose = [], []
    for key in sleap_query.fetch("KEY"):
        identity_count = len(tracking.SLEAPTracking.PoseIdentity & key)
        part_count = len(tracking.SLEAPTracking.Part & key)
        if part_count <= identity_count:
            not_fullpose.append(key)
        else:
            fullpose.append(key)

    logger.info(f"Fullpose: {len(fullpose)} | Not fullpose: {len(not_fullpose)}")

    if delete_not_fullpose:
        (tracking.SLEAPTracking & not_fullpose).delete()

    return fullpose, not_fullpose
