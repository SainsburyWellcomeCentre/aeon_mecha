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
# Convention: <experiment_tag>-<arena>, e.g. "abcGolden01-aeonx1".
EXPERIMENT_NAME = "abcGolden01-aeonx1"

# Absolute paths to raw data on Ceph.
# abc experiments record behavior and ephys on separate acquisition machines:
#   - AEON3 captures behavior (CameraTop, etc.)
#   - AEONX1 captures ephys (NeuropixelsV2)
# Each is registered as a separate directory type so the pipeline knows where
# to look for each kind of data.
RAW_BEHAVIOR_DIR = "/ceph/aeon/aeon/data/raw/AEON3/abcGolden01"
RAW_EPHYS_DIR = "/ceph/aeon/aeon/data/raw/AEONX1/abcGolden01"

# SWC subject ID.
SUBJECT = "IAA-1147881"

# Serial number of the physical probe (from Metadata.yml / GainCalibrationFileName).
PROBE_SERIAL = "23299108854"

# Probe type string used by the pipeline. The device directory on Ceph is
# called "NeuropixelsV2", and the pipeline maps that to "neuropixels2.0"
# via DEVICE_PROBE_TYPE_MAP in ephys_utils.py. The ProbeType entry we
# create here must match that string exactly.
PROBE_TYPE = "neuropixels2.0"

# Path to the channel mapping JSON (probeinterface format), relative to
# the first epoch directory. This defines the 384 active electrodes.
CHANNEL_MAP_FILE = "M81_ProbeB_4Shanks_1000_to_1700_um.json"

# Local directory for probe_assignments.json when Ceph is read-only.
# The file will be written here if the epoch directory on Ceph can't be
# written to. EphysEpoch.populate() checks this location automatically.
PROBE_ASSIGNMENTS_DIR = Path.home() / ".aeon_probe_assignments"


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def register_experiment(experiment_name, raw_behavior_dir, raw_ephys_dir, subject):
    """Insert experiment, subject, and data directories into the database.

    For abc experiments, behavior and ephys data live on separate machines
    (AEON3 and AEONX1 respectively). We register both directories:
    - "raw"       -> behavior data (CameraTop, etc.)
    - "raw-ephys" -> ephys data (NeuropixelsV2)
    """
    # Deferred imports so there are no DB side effects at module load time.
    from aeon.dj_pipeline import acquisition, lab, subject as subject_mod

    # --- Subject ---
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
    lab.Location.insert1(
        {
            "lab": "SWC",
            "location": "AEONX1",
            "location_description": "acquisition machine AEONX1",
        },
        skip_duplicates=True,
    )
    print("Location: AEONX1")

    # --- Derive experiment_start_time from the first epoch directory ---
    # Use the ephys directory to find the experiment start time, since that
    # is the primary data source for this guide.
    raw_path = Path(raw_ephys_dir)
    epoch_dirs = sorted(
        d.name for d in raw_path.iterdir()
        if d.is_dir() and "T" in d.name and not d.name.startswith(".")
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {raw_ephys_dir}")
    date_part, time_part = epoch_dirs[0].split("T")
    experiment_start_time = f"{date_part} {time_part.replace('-', ':')}"

    # --- Derive directory_path relative to ceph_aeon root ---
    # The pipeline resolves full paths as:
    #   get_repository_path("ceph_aeon") / directory_path
    # where the ceph_aeon root is /ceph/aeon/. Strip that prefix.
    ceph_root = "/ceph/aeon/"

    def _relative_to_ceph(abs_path):
        if abs_path.startswith(ceph_root):
            return abs_path[len(ceph_root):]
        return abs_path.lstrip("/")

    behavior_dir_path = _relative_to_ceph(raw_behavior_dir)
    ephys_dir_path = _relative_to_ceph(raw_ephys_dir)

    # --- Experiment ---
    acquisition.Experiment.insert1(
        {
            "experiment_name": experiment_name,
            "experiment_start_time": experiment_start_time,
            "experiment_description": "Golden baseline dataset - abc ephys (AEON3 + AEONX1)",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": "AEONX1",
            "experiment_type": "foraging",
        },
        skip_duplicates=True,
    )
    print(f"Experiment: {experiment_name} (start: {experiment_start_time})")

    # --- Experiment.Directory ---
    # Register both directories. "raw" is for behavior data (CameraTop on
    # AEON3), "raw-ephys" is for ephys data (NeuropixelsV2 on AEONX1).
    # Epoch.ingest_epochs() automatically discovers epochs from each.
    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": experiment_name,
                "directory_type": "raw",
                "repository_name": "ceph_aeon",
                "directory_path": behavior_dir_path,
                "load_order": 0,
            },
            {
                "experiment_name": experiment_name,
                "directory_type": "raw-ephys",
                "repository_name": "ceph_aeon",
                "directory_path": ephys_dir_path,
                "load_order": 1,
            },
        ],
        skip_duplicates=True,
    )
    print(f"Directory: raw       -> {behavior_dir_path}")
    print(f"Directory: raw-ephys -> {ephys_dir_path}")

    # --- Experiment.Subject ---
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
    create_probe_type(probe_type, "imec", "NP2014")
    n_electrodes = len(ProbeType.Electrode & {"probe_type": probe_type})
    print(f"ProbeType created: {probe_type} ({n_electrodes} electrodes)")


def create_electrode_config(raw_ephys_dir, channel_map_file, probe_type):
    """Create ElectrodeConfig + Electrode entries from the channel mapping JSON.

    The probeinterface JSON describes the full probe geometry (e.g. 5120
    contacts for a 4-shank NP2.0), but only a subset are actively recorded.
    The ``device_channel_indices`` array marks active contacts (value >= 0
    gives the raw channel index) and inactive ones (value == -1).

    This function filters to active contacts only and creates an
    ElectrodeConfig with one Electrode entry per active channel.
    """
    from aeon.dj_pipeline.ephys import ElectrodeConfig

    # The channel mapping JSON (probeinterface format) lives inside the first
    # epoch directory on Ceph.
    #
    # EphysChunk.ingest_chunks() requires at least one ElectrodeConfig entry
    # for the probe_type. If none exists, it raises a ValueError. So we must
    # create this before ingesting chunks.

    # Find the first epoch directory (they are named like "2026-05-05T15-15-51")
    raw_path = Path(raw_ephys_dir)
    epoch_dirs = sorted(
        d for d in raw_path.iterdir()
        if d.is_dir() and "T" in d.name and not d.name.startswith(".")
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {raw_ephys_dir}")
    first_epoch = epoch_dirs[0]

    json_path = first_epoch / channel_map_file
    if not json_path.exists():
        raise FileNotFoundError(
            f"Channel map file not found: {json_path}\n"
            f"Expected a probeinterface JSON at this path."
        )

    with open(json_path) as f:
        pi_data = json.load(f)

    # Extract active electrodes using device_channel_indices.
    # Each probe in the JSON has:
    #   contact_ids: electrode site indices on the physical probe (0-5119)
    #   device_channel_indices: raw channel index for each contact, or -1
    #     if the contact is not being recorded
    active_electrodes = []  # (electrode_site_id, raw_channel_idx)
    for probe in pi_data.get("probes", []):
        contact_ids = probe.get("contact_ids", [])
        dci = probe.get("device_channel_indices")
        if dci is None:
            raise ValueError(
                f"No device_channel_indices in {json_path}. "
                f"Cannot determine which contacts are active."
            )
        for cid, ch_idx in zip(contact_ids, dci):
            if int(ch_idx) >= 0:
                active_electrodes.append((int(cid), int(ch_idx)))

    if not active_electrodes:
        raise ValueError(
            f"No active contacts (device_channel_indices >= 0) in {json_path}."
        )

    # Sort by raw channel index for readable output.
    active_electrodes.sort(key=lambda x: x[1])
    active_site_ids = [e[0] for e in active_electrodes]
    n_channels = len(active_electrodes)

    # Config name uses channel range (0-383), not site range (114-4049).
    electrode_config_name = f"0-{n_channels - 1}"

    electrode_config_key = {
        "probe_type": probe_type,
        "electrode_config_name": electrode_config_name,
    }

    if ElectrodeConfig & electrode_config_key:
        existing = len(ElectrodeConfig.Electrode & electrode_config_key)
        print(
            f"ElectrodeConfig already exists: {electrode_config_name} "
            f"({existing} electrodes)"
        )
        return

    ElectrodeConfig.insert1(
        {
            **electrode_config_key,
            "electrode_config_description": (
                f"{n_channels} active channels from {channel_map_file}"
            ),
            "electrode_config_hash": uuid.uuid4(),
        },
        skip_duplicates=True,
    )

    # Each Electrode entry links a (probe_type, electrode_config_name) to a
    # specific electrode on the ProbeType.Electrode table.
    ElectrodeConfig.Electrode.insert(
        ({**electrode_config_key, "electrode": site_id}
         for site_id in active_site_ids),
        skip_duplicates=True,
    )

    print(
        f"ElectrodeConfig created: {electrode_config_name} "
        f"({n_channels} active channels from {json_path.name})"
    )


def create_probe_assignments(raw_ephys_dir, probe_serial, subject):
    """Write probe_assignments.json to the epoch directory (or local fallback).

    The probe assignments file tells the pipeline which animal a given probe
    is implanted in. You only need one file — subsequent epochs inherit it
    automatically (carry-forward in read_probe_assignments).

    If the epoch directory on Ceph is read-only, the file is written to a
    local override directory instead (~/.aeon_probe_assignments/<experiment>/).
    Set PROBE_ASSIGNMENTS_DIR at the top of this script to use a custom path.
    """
    raw_path = Path(raw_ephys_dir)

    # Find epoch directories (named like "2026-05-05T15-15-51")
    epoch_dirs = sorted(
        d for d in raw_path.iterdir()
        if d.is_dir() and "T" in d.name and not d.name.startswith(".")
    )
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch directories found in {raw_ephys_dir}")
    first_epoch = epoch_dirs[0]

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

    # Try writing to the epoch directory on Ceph first.
    json_path = first_epoch / "probe_assignments.json"
    if json_path.exists():
        print(f"probe_assignments.json already exists: {json_path}")
        return str(json_path.parent)

    try:
        with open(json_path, "w") as f:
            json.dump(assignments, f, indent=2)
        print(f"Wrote probe_assignments.json: {json_path}")
        print(f"  probe {probe_serial} -> subject {subject}")
        return str(json_path.parent)
    except PermissionError:
        pass

    # Ceph is read-only — write to a local override directory instead.
    override_dir = PROBE_ASSIGNMENTS_DIR / Path(raw_ephys_dir).name
    override_dir.mkdir(parents=True, exist_ok=True)
    override_path = override_dir / "probe_assignments.json"

    if override_path.exists():
        print(f"probe_assignments.json already exists (local): {override_path}")
        return str(override_dir)

    with open(override_path, "w") as f:
        json.dump(assignments, f, indent=2)
    print(f"Ceph is read-only. Wrote probe_assignments.json to local override:")
    print(f"  {override_path}")
    print(f"  probe {probe_serial} -> subject {subject}")
    return str(override_dir)


def register_epochs(experiment_name, probe_assignments_dir=None):
    """Ingest acquisition epochs and populate EphysEpoch."""
    from aeon.dj_pipeline import acquisition
    from aeon.dj_pipeline.ephys import EphysEpoch

    # Epoch.ingest_epochs() automatically handles both directory types:
    #   - "raw": scans for CameraTop CSV chunks (behavior epochs)
    #   - "raw-ephys": enumerates dirs with NeuropixelsV2 subdirs (ephys epochs)
    # All discovered epochs go into the same Epoch table, each tagged with
    # its directory_type so downstream tables resolve the correct path.
    print("Ingesting acquisition epochs...")
    acquisition.Epoch.ingest_epochs(experiment_name)

    epoch_count = len(acquisition.Epoch & {"experiment_name": experiment_name})
    behavior_epochs = len(
        acquisition.Epoch & {"experiment_name": experiment_name, "directory_type": "raw"}
    )
    ephys_epochs = len(
        acquisition.Epoch & {"experiment_name": experiment_name, "directory_type": "raw-ephys"}
    )
    print(f"Epochs in database: {epoch_count} ({behavior_epochs} behavior, {ephys_epochs} ephys)")

    if epoch_count == 0:
        raise RuntimeError(
            f"No epochs found for '{experiment_name}'. Check that:\n"
            f"  - The 'raw' directory has CameraTop data (behavior)\n"
            f"  - The 'raw-ephys' directory has NeuropixelsV2 subdirs (ephys)\n"
            f"Registered directories:\n"
            + "\n".join(
                f"  {d['directory_type']}: {d['directory_path']}"
                for d in (acquisition.Experiment.Directory & {"experiment_name": experiment_name}).to_dicts()
            )
        )

    # Populate EphysEpoch. If probe_assignments.json was written to a local
    # override directory (because Ceph is read-only), tell EphysEpoch where
    # to find it.
    if probe_assignments_dir is not None:
        EphysEpoch.probe_assignments_override_dir = probe_assignments_dir
    print("Populating EphysEpoch...")
    EphysEpoch.populate(display_progress=True)

    # Report results
    total = len(EphysEpoch & {"experiment_name": experiment_name})
    with_ephys = len(
        EphysEpoch & {"experiment_name": experiment_name, "has_ephys": True}
    )
    print(f"EphysEpoch total: {total}, with ephys data: {with_ephys}")


def ingest_sync_models(experiment_name):
    """Build ONIX-to-HARP sync models from HarpSync CSV files.

    Each epoch's NeuropixelsV2 directory contains HarpSync CSV files that
    record paired ONIX/HARP timestamps. EphysSyncModel.ingest() fits a
    linear regression to each CSV, producing a model that converts ONIX
    hardware timestamps to the HARP master clock.

    These sync models MUST exist before ingest_chunks() can run — chunks
    need them to compute their start/end times in the HARP clock.
    """
    from aeon.dj_pipeline.ephys import EphysSyncModel

    print("Ingesting sync models (ONIX → HARP)...")
    EphysSyncModel.ingest(experiment_name)

    sync_count = len(EphysSyncModel & {"experiment_name": experiment_name})
    print(f"EphysSyncModel entries: {sync_count}")


def ingest_chunks(experiment_name):
    """Ingest ephys recording chunks for all probes."""
    from aeon.dj_pipeline.ephys import EphysChunk

    # EphysChunk.ingest_chunks() is a CLASS METHOD on a Manual table (not an
    # Imported table with .populate()). It scans the raw data directory for
    # all *_AmplifierData*.bin files, resolves each file's ProbeInsertion via
    # the EphysEpoch.Insertion table, and creates one chunk entry per file.
    # Chunk start/end times are converted from ONIX timestamps to HARP clock
    # using EphysSyncModel entries (which must be ingested first).
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

    print("\n--- 1/8: Register experiment, subject, and directories ---")
    register_experiment(EXPERIMENT_NAME, RAW_BEHAVIOR_DIR, RAW_EPHYS_DIR, SUBJECT)

    print("\n--- 2/8: Ensure ProbeType exists ---")
    ensure_probe_type(PROBE_TYPE)

    print("\n--- 3/8: Create electrode configuration ---")
    create_electrode_config(RAW_EPHYS_DIR, CHANNEL_MAP_FILE, PROBE_TYPE)

    print("\n--- 4/8: Create probe assignments JSON ---")
    assignments_dir = create_probe_assignments(RAW_EPHYS_DIR, PROBE_SERIAL, SUBJECT)

    print("\n--- 5/8: Register epochs ---")
    register_epochs(EXPERIMENT_NAME, probe_assignments_dir=assignments_dir)

    print("\n--- 6/8: Ingest sync models ---")
    ingest_sync_models(EXPERIMENT_NAME)

    print("\n--- 7/8: Ingest ephys chunks ---")
    ingest_chunks(EXPERIMENT_NAME)

    print("\n--- 8/8: Verify registration ---")
    verify_registration(EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("  Step 1 complete. Ready for block definition (Step 2).")
    print("=" * 60)
