"""01a -- Arrow Register basic Experiment.

Get the experiment,subject, probe types set up

    uv run python docs/ephys_runbooks/step01a_goldenarrow.py
"""

# existing functions / utils

import json
from datetime import datetime
from pathlib import Path

from aeon.dj_pipeline import acquisition, lab
from aeon.dj_pipeline import subject as subject_mod
from aeon.dj_pipeline.utils.paths import get_repository_path

EXPERIMENT_NAME = "golden_arrow_01"

RAW_BEHAVIOR_DIR = f"/ceph/aeon/aeon/data/raw/AEON3/{EXPERIMENT_NAME}"
RAW_EPHYS_DIR = f"/ceph/aeon/aeon/data/raw/AEONX1/{EXPERIMENT_NAME}"

# The ephys metadata contains all subject data, which we can use to ingest automatically
# There might be a better way to do this...
one_metadata_path = list(Path(RAW_EPHYS_DIR).glob("*/Metadata.json"))[0]
with open(one_metadata_path) as f:
    metadata = json.load(f)

# register lab + subject
lab.Location.insert1(
    {
        "lab": "SWC",
        "location": "AEONX1",
        "location_description": "acquisition machine AEONX1",
    },
    skip_duplicates=True,
)

subject_metadata = metadata["subject"]
subject_id = subject_metadata["Subject"]

subject_mod.Subject.insert1(
    {
        "subject": subject_id,
        "sex": "U",
        "subject_birth_date": "2026-05-01",
        "subject_description": "Ephys pilot subject",
    },
    skip_duplicates=True,
)

subject_key = {"subject": subject_id}

# --- Experiment ---

# Derive experiment_start_time from the first epoch directory
raw_path = Path(RAW_EPHYS_DIR)
epoch_dirs = sorted(
    d.name for d in raw_path.iterdir() if d.is_dir() and "T" in d.name and not d.name.startswith(".")
)

# convert epoch dir to standard datetime(6) string
dt = datetime.strptime(epoch_dirs[0], "%Y-%m-%dT%H%M%S%z")
experiment_start_time = dt.strftime("%Y-%m-%d %H:%M:%S")

acquisition.Experiment.insert1(
    {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_start_time": experiment_start_time,
        "experiment_description": "Golden arrow baseline dataset - abc ephys (AEON3 + AEONX1)",
        "arena_name": "circle-2m",
        "lab": "SWC",
        "location": "AEONX1",
        "experiment_type": "foraging",
    },
    skip_duplicates=True,
)

exp_key = {"experiment_name": EXPERIMENT_NAME}

ceph_root = get_repository_path("ceph_aeon")
behavior_dir_path = Path(RAW_BEHAVIOR_DIR).relative_to(ceph_root).as_posix()
ephys_dir_path = Path(RAW_EPHYS_DIR).relative_to(ceph_root).as_posix()

acquisition.Experiment.Directory.insert(
    [
        {
            "experiment_name": EXPERIMENT_NAME,
            "directory_type": "raw",
            "repository_name": "ceph_aeon",
            "directory_path": behavior_dir_path,
            "load_order": 0,
        },
        {
            "experiment_name": EXPERIMENT_NAME,
            "directory_type": "raw-ephys",
            "repository_name": "ceph_aeon",
            "directory_path": ephys_dir_path,
            "load_order": 1,
        },
    ],
    skip_duplicates=True,
)

# Add subject to experiment
acquisition.Experiment.Subject.insert1(
    {
        "experiment_name": EXPERIMENT_NAME,
        "subject": subject_id,
    },
    skip_duplicates=True,
)
