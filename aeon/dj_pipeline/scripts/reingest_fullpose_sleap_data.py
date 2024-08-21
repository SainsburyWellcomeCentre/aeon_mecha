from datetime import datetime
from pipeline import acquisition, tracking


exp_key = {"experiment_name": "social0.3-aeon4"}


def find_chunks_to_reingest(exp_key):
    # find processed path for exp_key
    processed_dir = acquisition.Experiment.get_data_directory(exp_key, "processed")

    files = sorted(f.stem for f in processed_dir.rglob("CameraTop*.bin") if f.is_file())
    # extract timestamps from the file names & convert to datetime
    file_times = [datetime.strptime(f.split("_")[-1], "%Y-%m-%dT%H-%M-%S") for f in files]

    query = acquisition.Chunk & exp_key & [{"chunk_start": t} for t in file_times]
    epochs = acquisition.Epoch & query.proj("epoch_start")
    sleap_query = tracking.SLEAPTracking & (acquisition.Chunk & epochs.proj("epoch_start"))
    # sleap_query.delete()
    return sleap_query

