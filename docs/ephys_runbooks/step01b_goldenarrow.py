"""01b -- Ingest actual ephys data, such as chunks.

During an experiment, you would run this file e.g. daily
to ingest all the new ephys data.

    uv run python docs/ephys_runbooks/step01b_goldenarrow.py
"""

# existing functions / utils

import json
from pathlib import Path

from step01_register_experiment import create_electrode_config

from aeon.dj_pipeline import acquisition
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

EXPERIMENT_NAME = "abcEphys02"
subject_id = "IAA-1150342"

exp_key = {"experiment_name": EXPERIMENT_NAME}
subject_key = {"subject": subject_id}

RAW_EPHYS_DIR = acquisition.Experiment.get_data_directory(exp_key, directory_type="raw-ephys")

PROBE_TYPE = "neuropixels2.0"

# ------- probe

if ProbeType & {"probe_type": PROBE_TYPE}:
    n_electrodes = len(ProbeType.Electrode & {"probe_type": PROBE_TYPE})
    print(f"ProbeType already exists: {PROBE_TYPE} ({n_electrodes} electrodes)")
else:
    create_probe_type(PROBE_TYPE, "imec", "NP2014")

# Extract probes metadata from metadata file
def get_probes_metadata_from_epoch_metadata(epoch_metadata):
    """Get probe metadata for each epoch."""
    headstage_metadata = epoch_metadata["ephys"]["Headstage"]
    implants_metadata = epoch_metadata["subject"]["Implants"]

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
