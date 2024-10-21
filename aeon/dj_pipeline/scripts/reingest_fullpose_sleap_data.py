from datetime import datetime
from aeon.dj_pipeline import acquisition, tracking


exp_key = {"experiment_name": "social0.2-aeon4"}


def find_chunks_to_reingest(exp_key, delete_not_fullpose=False):
    # find processed path for exp_key
    processed_dir = acquisition.Experiment.get_data_directory(exp_key, "processed")

    files = sorted(f.stem for f in processed_dir.rglob("CameraTop_202_*.bin") if f.is_file())
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

    if delete_not_fullpose:
        (tracking.SLEAPTracking & not_fullpose).delete()

    return fullpose, not_fullpose

