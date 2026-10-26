"""01 -- Arrow Register Experiment.

For now, the ingestion code for raw data is quite different
for arrow and not-arrow formatted data.

Hence, we keep two ingsetion scripts until the auto-ingestion
is merged into the source of aeon_mecha.

    uv run python docs/ephys_runbooks/step01_goldenarrow.py
"""

# existing functions / utils

import json
from datetime import datetime
from pathlib import Path

from step01_register_experiment import create_electrode_config

from aeon.dj_pipeline import acquisition, lab
from aeon.dj_pipeline import subject as subject_mod
from aeon.dj_pipeline.ephys import (
    EphysChunk,
    EphysEpoch,
    EphysEpochConfig,
    EphysSyncModel,
    Probe,
    ProbeInsertion,
    ProbeType,
    create_probe_type,
)
from aeon.dj_pipeline.utils.paths import get_repository_path

EXPERIMENT_NAME = "golden_arrow_01"

RAW_BEHAVIOR_DIR = f"/ceph/aeon/aeon/data/raw/AEON3/{EXPERIMENT_NAME}"
RAW_EPHYS_DIR = f"/ceph/aeon/aeon/data/raw/AEONX1/{EXPERIMENT_NAME}"

PROBE_TYPE = "neuropixels2.0"

if ProbeType & {"probe_type": PROBE_TYPE}:
    n_electrodes = len(ProbeType.Electrode & {"probe_type": PROBE_TYPE})
    print(f"ProbeType already exists: {PROBE_TYPE} ({n_electrodes} electrodes)")
else:
    create_probe_type(PROBE_TYPE, "imec", "NP2014")

# The ephys metadata contains all subject data, which we can use to ingest automatically
# There might be a better way to do this...
one_metadata_path = Path(RAW_EPHYS_DIR) / "2026-10-13T094255Z/Metadata.json"
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

# ------- probe


# Extract probes metadata from metadata file
def get_probes_metadata_from_epoch_metadata(epoch_metadata):
    """Get probe metadata for each epoch."""
    headstage_metadata = epoch_metadata["ephys"]["Headstage"]
    implants_metadata = epoch_metadata["Subject"]["Implants"]

    probes = {}
    for key, values in headstage_metadata.items():
        if "Probe" in key:
            probe = {}
            gain_filename = Path(values["ProbeConfiguration"]["GainCalibrationFileName"])
            serial_number = gain_filename.name.split("_")[0]
            probe["serial_number"] = serial_number
            probe_filename = Path(values["ProbeConfiguration"]["ProbeInterfaceFileName"])
            probe["channel_map_file"] = probe_filename.name
            probes[key] = probe

    for implant_metadata in implants_metadata:
        probe_name = implant_metadata["Probe"]
        for property in ["InsertionNumber", "TargetArea", "ImplantationDate"]:
            probes[probe_name][property] = implant_metadata[property]

    return probes


def register_probes(probes, exp_key, subject_key):
    """Ingest probe info for all probes.

    Note that this only needs to be run once for each probe,
    but since the active probes in each epoch _could_ be
    difference, we run it every epoch.
    """
    for probe_metadata in probes.values():
        create_electrode_config(
            RAW_EPHYS_DIR, str(Path("NeuropixelsV2") / probe_metadata["channel_map_file"]), PROBE_TYPE
        )
        Probe.insert1(
            {
                "probe": probe_metadata["serial_number"],
                "probe_type": PROBE_TYPE,
            },
            skip_duplicates=True,
        )
        ProbeInsertion.insert1(
            {
                "probe": probe_metadata["serial_number"],
                "insertion_number": probe_metadata["InsertionNumber"],
                "implantation_date": probe_metadata["ImplantationDate"],
                **exp_key,
                **subject_key,
            },
            skip_duplicates=True,
        )


# Ingest actual ephys data - Epochs and Chunks

EphysEpoch.ingest_epochs(EXPERIMENT_NAME)

for epoch in EphysEpoch:
    epoch_key = {
        "epoch_start": epoch["epoch_start"],
        "experiment_name": epoch["experiment_name"],
    }

    # This code should probably go in EphysEpochConfig.make
    # For now, we have all the probe data here already.

    epoch_dir = Path(epoch["epoch_dir"])
    epoch_metadata_path = Path(RAW_EPHYS_DIR) / epoch_dir / "Metadata.json"
    with open(epoch_metadata_path) as f:
        epoch_metadata = json.load(f)

    epoch_probes = get_probes_metadata_from_epoch_metadata(epoch_metadata)
    register_probes(epoch_probes, exp_key, subject_key)

    EphysEpochConfig.insert1(
        {
            **epoch_key,
            "n_probes": len(epoch_probes),
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    for probe_name, probe_metadata in epoch_probes.items():
        EphysEpochConfig.Insertion.insert1(
            {
                **epoch_key,
                "insertion_number": probe_metadata["InsertionNumber"],
                **subject_key,
                "probe_type": PROBE_TYPE,
                "probe_label": probe_name,
                "config_file_name": probe_metadata["channel_map_file"],
                "electrode_config_name": probe_metadata["channel_map_file"].split(".")[0],
            },
            skip_duplicates=True,
            allow_direct_insert=True,
        )

EphysSyncModel.ingest(EXPERIMENT_NAME)
EphysChunk.ingest_chunks(EXPERIMENT_NAME)
