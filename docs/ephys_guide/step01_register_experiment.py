"""
01 -- Register Experiment
========================
Register the experiment, subject, probe type, electrode configuration,
and probe assignments needed before any ephys data can be ingested.

This script sets up everything the pipeline requires to know about your
experiment: what it is called, where the raw data lives, which subject
was recorded, what kind of probe was used, and which electrodes were
active. It then ingests epochs and chunks so the data is ready for
block definition and spike sorting.

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_guide/step01_register_experiment.py
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
import json
import uuid
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration -- edit these for your experiment
# --------------------------------------------------------------------------

# Name that identifies this experiment in the database.
# Convention: <experiment_tag>-<arena>, e.g. "abcEphysPilot02-aeonx1".
EXPERIMENT_NAME = "abcEphysPilot02-aeonx1"

# Absolute path to the raw data on Ceph.
RAW_DATA_DIR = "/ceph/aeon/aeon/data/raw/AEONX1/abcEphysPilot02"

# SWC subject ID. Replace with the actual ID Adrian provides.
SUBJECT = "BAA-XXXXXXX"

# Serial number of the physical probe (from Metadata.yml / GainCalibrationFileName).
PROBE_SERIAL = "23299108854"

# Probe type string used by the pipeline. The device directory on Ceph is
# called "NeuropixelsV2", and the pipeline maps that to "neuropixels2.0"
# via DEVICE_PROBE_TYPE_MAP in ephys_utils.py. The ProbeType entry we
# create here must match that string exactly.
PROBE_TYPE = "neuropixels2.0"

# Path to the channel mapping JSON (probeinterface format), relative to
# the first epoch directory. This defines the 384 active electrodes.
CHANNEL_MAP_FILE = "recording_configurations/M81_ProbeB_4Shanks_1000_to_1700_um.json"


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def register_experiment(experiment_name, raw_data_dir, subject):
    """Insert experiment, subject, and data directory into the database."""
    # Deferred imports so there are no DB side effects at module load time.
    from aeon.dj_pipeline import acquisition, lab, subject as subject_mod

    # --- Subject ---
    # The Subject table holds basic animal metadata. It must exist before we
    # can add the subject to an experiment.
    subject_mod.Subject.insert1(
        {
            "subject": subject,
            "sex": "U",
            "subject_birth_date": "2024-01-01",
            "subject_description": "Ephys pilot subject",
        },
        skip_duplicates=True,
    )
    print(f"Subject: {subject}")

    # --- Location ---
    # Location is a Lookup table seeded with known rigs. If your arena is not
    # in the default list you need to insert it. "AEONX1" is the acquisition
    # machine for this experiment.
    lab.Location.insert1(
        {
            "lab": "SWC",
            "location": "AEONX1",
            "location_description": "acquisition machine AEONX1",
        },
        skip_duplicates=True,
    )
    print("Location: AEONX1")

    # --- Experiment ---
    # The Experiment table is the root of the pipeline. Every downstream table
    # references it. Fields:
    #   experiment_name       -- unique identifier (PK)
    #   experiment_start_time -- when recording began (first epoch timestamp)
    #   experiment_description
    #   arena_name            -- FK to lab.Arena (physical arena geometry)
    #   lab                   -- FK to lab.Lab
    #   location              -- FK to lab.Location (which rig / room)
    #   experiment_type       -- FK to acquisition.ExperimentType
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": "2026-05-05 15:15:51",
            "experiment_description": "Ephys pilot experiment 02 - AEONX1",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": "AEONX1",
            "experiment_type": "foraging",
        },
        skip_duplicates=True,
    )
    print(f"Experiment: {experiment_name}")

    # --- Experiment.Directory ---
    # Tells the pipeline where the raw data lives on Ceph. The directory_path
    # is *relative* to the ceph_aeon repository root (/ceph/aeon/). The
    # pipeline resolves the full path at runtime using:
    #   get_repository_path("ceph_aeon") / directory_path
    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": experiment_name,
                "directory_type": "raw",
                "repository_name": "ceph_aeon",
                "directory_path": "aeon/data/raw/AEONX1/abcEphysPilot02",
                "load_order": 0,
            },
        ],
        skip_duplicates=True,
    )
    print("Directory: raw -> aeon/data/raw/AEONX1/abcEphysPilot02")

    # --- Experiment.Subject ---
    # Links the subject to this experiment. A Part table of Experiment.
    acquisition.Experiment.Subject.insert1(
        {
            "experiment_name": experiment_name,
            "subject": subject,
        },
        skip_duplicates=True,
    )
    print(f"Experiment.Subject: {subject}")


def ensure_probe_type(probe_type):
    """Create the ProbeType entry (with electrode geometry) if it does not exist."""
    from aeon.dj_pipeline.ephys import ProbeType, create_probe_type

    if ProbeType & {"probe_type": probe_type}:
        n_electrodes = len(ProbeType.Electrode & {"probe_type": probe_type})
        print(f"ProbeType already exists: {probe_type} ({n_electrodes} electrodes)")
        return

    # ProbeType stores the physical geometry of the probe: every electrode's
    # (x, y) position and shank assignment. The pipeline needs this to build
    # electrode configurations and to map channels to electrodes.
    #
    # create_probe_type() uses the `probeinterface` library to fetch the
    # geometry from its built-in probe library.
    #
    # For a 4-shank Neuropixels 2.0 probe the correct probeinterface name
    # is "NP2014" (not "NP2004", which is the 1-shank variant). Verify this
    # matches your hardware before running.
    #
    # NOTE: probeinterface requires internet access to download probe
    # geometry data. On the HPC, internet may not be available from compute
    # nodes. If this step fails with a connection error, run it from the
    # gateway node instead:
    #     ssh aeon-hpc
    #     module load uv
    #     uv run python -c "
    #         from aeon.dj_pipeline.ephys import create_probe_type
    #         create_probe_type('neuropixels2.0', 'neuropixels', 'NP2014')
    #     "
    create_probe_type(probe_type, "neuropixels", "NP2014")
    n_electrodes = len(ProbeType.Electrode & {"probe_type": probe_type})
    print(f"ProbeType created: {probe_type} ({n_electrodes} electrodes)")


def create_electrode_config(raw_data_dir, channel_map_file, probe_type):
    """Create ElectrodeConfig + Electrode entries from the channel mapping JSON."""
    from aeon.dj_pipeline.ephys import ElectrodeConfig

    # The channel mapping JSON (probeinterface format) lives inside the first
    # epoch directory on Ceph. It defines which 384 electrodes out of the full
    # probe were active during recording.
    #
    # EphysChunk.ingest_chunks() requires at least one ElectrodeConfig entry
    # for the probe_type. If none exists, it raises a ValueError. So we must
    # create this before ingesting chunks.

    # Find the first epoch directory (they are named like "2026-05-05T15-15-51")
    raw_path = Path(raw_data_dir)
    epoch_dirs = sorted(
        d for d in raw_path.iterdir()
        if d.is_dir() and "T" in d.name and not d.name.startswith(".")
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {raw_data_dir}")
    first_epoch = epoch_dirs[0]

    json_path = first_epoch / channel_map_file
    if not json_path.exists():
        raise FileNotFoundError(
            f"Channel map file not found: {json_path}\n"
            f"Expected a probeinterface JSON at this path."
        )

    # Parse the probeinterface JSON to extract active electrode IDs.
    # The format has a "probes" array; each probe has "contact_ids" listing
    # the electrode indices that were active during recording.
    with open(json_path) as f:
        pi_data = json.load(f)

    contact_ids = []
    for probe in pi_data.get("probes", []):
        ids = probe.get("contact_ids", [])
        contact_ids.extend(int(cid) for cid in ids)

    if not contact_ids:
        raise ValueError(
            f"No contact_ids found in {json_path}. "
            f"Check that the file is in probeinterface format."
        )
    contact_ids = sorted(set(contact_ids))
    n_electrodes = len(contact_ids)

    # Build an electrode_config_name from the electrode range.
    # For 384 consecutive electrodes starting at 0, this gives "0-383".
    electrode_config_name = f"{contact_ids[0]}-{contact_ids[-1]}"

    electrode_config_key = {
        "probe_type": probe_type,
        "electrode_config_name": electrode_config_name,
    }

    # Check if already exists
    if ElectrodeConfig & electrode_config_key:
        existing = len(ElectrodeConfig.Electrode & electrode_config_key)
        print(
            f"ElectrodeConfig already exists: {electrode_config_name} "
            f"({existing} electrodes)"
        )
        return

    # Insert the config. The hash is a random UUID -- it just needs to be
    # unique. The pipeline uses the (probe_type, electrode_config_name)
    # composite key to look up configs, not the hash.
    ElectrodeConfig.insert1(
        {
            **electrode_config_key,
            "electrode_config_description": (
                f"Active electrodes from {channel_map_file} "
                f"({n_electrodes} channels)"
            ),
            "electrode_config_hash": uuid.uuid4(),
        },
        skip_duplicates=True,
    )

    # Each Electrode entry links a (probe_type, electrode_config_name) to a
    # specific electrode on the ProbeType.Electrode table.
    ElectrodeConfig.Electrode.insert(
        ({**electrode_config_key, "electrode": e} for e in contact_ids),
        skip_duplicates=True,
    )

    print(
        f"ElectrodeConfig created: {electrode_config_name} "
        f"({n_electrodes} electrodes from {json_path.name})"
    )


def create_probe_assignments(raw_data_dir, probe_serial, subject):
    """Write probe_assignments.json to the first epoch directory."""
    # The probe assignments file is the one thing the pipeline cannot
    # auto-discover: which animal a given probe is implanted in. The pipeline
    # reads this file during EphysEpoch.populate() to create ProbeInsertion
    # entries linking each probe serial number to a subject.
    #
    # You only need to place this file in the *first* epoch directory.
    # Subsequent epochs inherit the mapping automatically (carry-forward
    # behavior in read_probe_assignments).

    raw_path = Path(raw_data_dir)

    # Find epoch directories (named like "2026-05-05T15-15-51")
    epoch_dirs = sorted(
        d for d in raw_path.iterdir()
        if d.is_dir() and "T" in d.name and not d.name.startswith(".")
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {raw_data_dir}")
    first_epoch = epoch_dirs[0]

    json_path = first_epoch / "probe_assignments.json"

    # Safety: do not overwrite an existing file.
    if json_path.exists():
        print(f"probe_assignments.json already exists: {json_path}")
        print("Skipping to avoid overwriting. Delete the file first if you need to regenerate.")
        return

    # The JSON maps probe serial numbers to subjects. If you have multiple
    # probes, add one entry per serial.
    assignments = {
        "version": 1,
        "probe_assignments": {
            probe_serial: {
                "subject": subject,
            },
        },
    }

    with open(json_path, "w") as f:
        json.dump(assignments, f, indent=2)

    print(f"Wrote probe_assignments.json: {json_path}")
    print(f"  probe {probe_serial} -> subject {subject}")


def register_epochs(experiment_name):
    """Ingest acquisition epochs and populate EphysEpoch."""
    from aeon.dj_pipeline import acquisition
    from aeon.dj_pipeline.ephys import EphysEpoch

    # Step 1: Ingest acquisition Epochs.
    # Epoch.ingest_epochs() scans the raw data directory for epoch directories
    # (each one is a continuous recording period) and inserts them into the
    # Epoch table. It discovers epochs by looking for chunk files from a
    # reference device (typically CameraTop).
    print("Ingesting acquisition epochs...")
    acquisition.Epoch.ingest_epochs(experiment_name)

    epoch_count = len(acquisition.Epoch & {"experiment_name": experiment_name})
    print(f"Epochs in database: {epoch_count}")

    # Step 2: Populate EphysEpoch.
    # For each Epoch, EphysEpoch.make() checks whether the epoch directory
    # contains ephys data (a NeuropixelsV2 or NeuropixelsV2Beta subdirectory).
    # If it does, it:
    #   1. Discovers probes from the binary filenames (e.g. ProbeA, ProbeB)
    #   2. Reads Metadata.yml to get probe serial numbers
    #   3. Auto-creates Probe entries in the Probe table
    #   4. Reads probe_assignments.json (or carries forward from a previous
    #      epoch) to find the subject-probe mapping
    #   5. Creates ProbeInsertion entries linking each probe to a subject
    #   6. Inserts EphysEpoch.Insertion Part rows
    print("Populating EphysEpoch...")
    EphysEpoch.populate(display_progress=True)

    # Report results
    total = len(EphysEpoch & {"experiment_name": experiment_name})
    with_ephys = len(
        EphysEpoch & {"experiment_name": experiment_name, "has_ephys": True}
    )
    print(f"EphysEpoch total: {total}, with ephys data: {with_ephys}")


def ingest_chunks(experiment_name):
    """Ingest ephys recording chunks for all probes."""
    from aeon.dj_pipeline.ephys import EphysChunk

    # EphysChunk.ingest_chunks() is a CLASS METHOD on a Manual table (not an
    # Imported table with .populate()). It scans the raw data directory for
    # all *_AmplifierData*.bin files, resolves each file's ProbeInsertion via
    # the EphysEpoch.Insertion table, builds a sync model to convert ONIX
    # timestamps to HARP clock, and inserts one chunk entry per file.
    #
    # Each chunk corresponds to approximately one hour of recording. The
    # chunk_start and chunk_end times are in the synced HARP clock.
    print("Ingesting ephys chunks...")
    EphysChunk.ingest_chunks(experiment_name)

    chunks = (EphysChunk & {"experiment_name": experiment_name}).to_dicts()
    print(f"EphysChunk entries: {len(chunks)}")

    if chunks:
        starts = [c["chunk_start"] for c in chunks]
        ends = [c["chunk_end"] for c in chunks]
        print(f"Time range: {min(starts)} to {max(ends)}")


def verify_registration(experiment_name):
    """Print a summary of everything registered for this experiment."""
    from aeon.dj_pipeline import acquisition
    from aeon.dj_pipeline.ephys import (
        EphysChunk, EphysEpoch, Probe, ProbeInsertion,
    )

    print(f"Experiment: {experiment_name}")

    # Epochs
    epoch_count = len(acquisition.Epoch & {"experiment_name": experiment_name})
    ephys_epoch_count = len(
        EphysEpoch & {"experiment_name": experiment_name, "has_ephys": True}
    )
    print(f"Epochs: {epoch_count} total, {ephys_epoch_count} with ephys data")

    # Probe insertions
    insertions = (
        ProbeInsertion & {"experiment_name": experiment_name}
    ).to_dicts()
    print(f"ProbeInsertions: {len(insertions)}")
    for pi in insertions:
        probe_type = (Probe & {"probe": pi["probe"]}).fetch1("probe_type")
        print(
            f"  insertion {pi['insertion_number']}: "
            f"subject={pi['subject']}, probe={pi['probe']} ({probe_type})"
        )

    # Chunks per insertion
    for pi in insertions:
        pi_key = {
            "experiment_name": pi["experiment_name"],
            "subject": pi["subject"],
            "insertion_number": pi["insertion_number"],
        }
        chunk_count = len(EphysChunk & pi_key)
        if chunk_count > 0:
            chunk_starts = (EphysChunk & pi_key).to_arrays("chunk_start")
            chunk_ends = (EphysChunk & pi_key).to_arrays("chunk_end")
            print(
                f"  -> {chunk_count} chunks: "
                f"{min(chunk_starts)} to {max(chunk_ends)}"
            )
        else:
            print(f"  -> 0 chunks")


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1: Register Experiment")
    print("=" * 60)

    print("\n--- 1/7: Register experiment, subject, and directory ---")
    register_experiment(EXPERIMENT_NAME, RAW_DATA_DIR, SUBJECT)

    print("\n--- 2/7: Ensure ProbeType exists ---")
    ensure_probe_type(PROBE_TYPE)

    print("\n--- 3/7: Create electrode configuration ---")
    create_electrode_config(RAW_DATA_DIR, CHANNEL_MAP_FILE, PROBE_TYPE)

    print("\n--- 4/7: Create probe assignments JSON ---")
    create_probe_assignments(RAW_DATA_DIR, PROBE_SERIAL, SUBJECT)

    print("\n--- 5/7: Register epochs ---")
    register_epochs(EXPERIMENT_NAME)

    print("\n--- 6/7: Ingest ephys chunks ---")
    ingest_chunks(EXPERIMENT_NAME)

    print("\n--- 7/7: Verify registration ---")
    verify_registration(EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("  Step 1 complete. Ready for block definition (Step 2).")
    print("=" * 60)
