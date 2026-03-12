"""Utility functions for the ephys pipeline.

Helpers for probe discovery, subject-probe mapping, sync model processing,
and probe type creation. Used by ephys.py table classes.
"""

import json
import re
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

import datajoint as dj

from swc.aeon.io import api as io_api
from aeon.schema.ephys import social_ephys

logger = dj.logger

# Mapping from device directory name to probe_type
DEVICE_PROBE_TYPE_MAP = {
    "NeuropixelsV2Beta": "neuropixels2.0_beta",
    "NeuropixelsV2": "neuropixels2.0",
}


def get_probe_id(metadata: dict | None, device_name: str, probe_label: str) -> str:
    """Extract probe identifier from metadata.

    For V2Beta hardware (no serial numbers): probe ID = "{device_name}_{label}"
    For V2 hardware: probe ID = serial number from GainCalibrationFileName

    Args:
        metadata: Parsed Metadata.yml dict, or None
        device_name: Hardware device name (e.g., "NeuropixelsV2Beta", "NeuropixelsV2")
        probe_label: Probe label from filename (e.g., "ProbeA", "ProbeB")

    Returns:
        Probe identifier string
    """
    default_id = f"{device_name}_{probe_label}"

    if metadata is None:
        return default_id

    # V2 hardware: try to extract serial number
    if device_name == "NeuropixelsV2":
        v2e_config = metadata.get("NeuropixelsV2e")
        if v2e_config:
            config_key_map = {
                "ProbeA": "ProbeConfigurationA",
                "ProbeB": "ProbeConfigurationB",
            }
            config_key = config_key_map.get(probe_label)
            if config_key and config_key in v2e_config:
                gain_cal = v2e_config[config_key].get("GainCalibrationFileName")
                if gain_cal:
                    try:
                        serial = Path(gain_cal).parent.name
                        if serial and serial.isdigit():
                            return serial
                    except Exception:
                        pass

    return default_id


def find_or_create_probe_insertion(
    experiment_name: str, subject: str, probe_id: str,
    probe_insertion_table, probe_table,
) -> dict:
    """Find existing or create new ProbeInsertion for this (experiment, subject, probe).

    Returns the ProbeInsertion key: {experiment_name, subject, insertion_number}.

    Args:
        experiment_name: Experiment name
        subject: Subject name
        probe_id: Probe identifier (serial number)
        probe_insertion_table: The ProbeInsertion table class
        probe_table: The Probe table class

    Returns:
        Dict with ProbeInsertion PK fields
    """
    # Check if a ProbeInsertion already exists for this probe + subject + experiment
    existing = (
        probe_insertion_table
        & {"experiment_name": experiment_name, "subject": subject, "probe": probe_id}
    ).fetch(as_dict=True)
    if existing:
        row = existing[0]
        return {
            "experiment_name": row["experiment_name"],
            "subject": row["subject"],
            "insertion_number": row["insertion_number"],
        }

    # Create new ProbeInsertion with auto-assigned insertion_number
    existing_nums = (
        probe_insertion_table
        & {"experiment_name": experiment_name, "subject": subject}
    ).fetch("insertion_number")
    new_num = int(existing_nums.max()) + 1 if len(existing_nums) > 0 else 1

    probe_insertion_table.insert1({
        "experiment_name": experiment_name,
        "subject": subject,
        "insertion_number": new_num,
        "probe": probe_id,
    })
    logger.info(
        f"Created ProbeInsertion: experiment={experiment_name}, "
        f"subject={subject}, insertion_number={new_num}, probe={probe_id}"
    )
    return {
        "experiment_name": experiment_name,
        "subject": subject,
        "insertion_number": new_num,
    }


def read_probe_assignments(
    key: dict, epoch_path: Path, probe_labels: list[str],
    insertion_table,
) -> dict:
    """Read subject-probe mapping from file or carry forward from previous epoch.

    Priority:
    1. probe_assignments.json in epoch directory
    2. Carry-forward from most recent EphysEpoch with .Insertion entries
    3. Pre-existing ProbeInsertion matched by probe serial (error if not found)

    Args:
        key: Epoch key (experiment_name, epoch_start)
        epoch_path: Path to epoch directory
        probe_labels: Sorted list of probe labels discovered in this epoch
        insertion_table: The EphysEpoch.Insertion Part table class

    Returns:
        Dict mapping probe_label -> {"subject": str, ...}
    """
    raise NotImplementedError(
        "Probe-subject assignment resolution is not yet implemented. "
        "The exact file format and carry-forward logic will be determined "
        "once the experimental data conventions are finalized."
    )


def discover_epoch_probes(epoch_path: Path) -> tuple[str | None, Path | None, list[str]]:
    """Discover ephys device and probe labels in an epoch directory.

    Looks for known device subdirectories (NeuropixelsV2Beta, NeuropixelsV2)
    and parses probe labels from amplifier binary filenames.

    Args:
        epoch_path: Path to the epoch directory

    Returns:
        Tuple of (device_name, device_dir, sorted_probe_labels).
        If no ephys data found, returns (None, None, []).
    """
    device_name = None
    device_dir = None
    for subdir_name in ("NeuropixelsV2Beta", "NeuropixelsV2"):
        candidate = epoch_path / subdir_name
        if candidate.exists() and candidate.is_dir():
            device_name = subdir_name
            device_dir = candidate
            break

    if device_name is None:
        return None, None, []

    amplifier_files = sorted(device_dir.glob(f"{device_name}_Probe*_AmplifierData_*.bin"))
    if not amplifier_files:
        return device_name, device_dir, []

    # e.g., "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin" -> "ProbeA"
    probe_labels = set()
    for f in amplifier_files:
        match = re.search(rf"{device_name}_(Probe[A-Z])_AmplifierData", f.name)
        if match:
            probe_labels.add(match.group(1))

    return device_name, device_dir, sorted(probe_labels)


def parse_epoch_metadata(epoch_path: Path) -> dict | None:
    """Parse Metadata.yml (JSON format) from an epoch directory.

    Args:
        epoch_path: Path to the epoch directory

    Returns:
        Parsed metadata dict, or None if not found or unparseable.
    """
    metadata_path = epoch_path / "Metadata.yml"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path) as f:
            return json.load(f)  # JSON despite .yml extension
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to parse Metadata.yml at {metadata_path}: {e}")
        return None


def process_ephys_file(
    ephys_file: Path,
    raw_dir: Path,
    insertion_key: dict,
    epoch_start: datetime,
    probe_label: str,
    probe_type: str,
    electrode_config_name: str,
    sync_models: dict,
    chunk_table,
) -> None:
    """Process a single ephys binary file into a chunk entry with sync model.

    Args:
        ephys_file: Path to the amplifier data binary file
        raw_dir: Root raw data directory
        insertion_key: ProbeInsertion key (experiment_name, subject, insertion_number)
        epoch_start: Epoch start datetime for this file's epoch
        probe_label: File label (e.g., "ProbeA")
        probe_type: Probe type string
        electrode_config_name: Electrode configuration name
        sync_models: Mutable dict cache for sync models (shared across files)
        chunk_table: The EphysChunk table class (for inserts)
    """
    clock_file = ephys_file.with_name(
        ephys_file.name.replace("AmplifierData", "Clock")
    )
    onix_ts = np.memmap(clock_file, mode="r", dtype=np.uint64)

    model_parent_dir = raw_dir / ephys_file.relative_to(raw_dir).parents[-2]
    if model_parent_dir.as_posix() not in sync_models:
        device_prefix = ephys_file.name.split(f"_{probe_label}_")[0]
        device_reader = getattr(social_ephys, device_prefix, None)
        if device_reader is None:
            raise ValueError(
                f"No sync reader found for device '{device_prefix}'. "
                f"Available devices: {list(social_ephys.keys())}"
            )
        sync_models[model_parent_dir.as_posix()] = io_api.load(
            model_parent_dir,
            device_reader.HarpSyncModel,
        )

    sync_model = sync_models[model_parent_dir.as_posix()]
    matched_sync = sync_model.query(
        f"(clock_start <= {onix_ts[0]} <= clock_end)"
        f" | "
        f"(clock_start <= {onix_ts[-1]} <= clock_end)"
    )

    sync_entries = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (_, r) in enumerate(matched_sync.iterrows()):
            if idx == 0:
                chunk_start = r.model.predict(
                    np.array(onix_ts[0]).reshape(-1, 1)
                ).flatten()[0]
                chunk_start = io_api.to_datetime(chunk_start)
                if hasattr(chunk_start, "tz_localize"):
                    chunk_start = chunk_start.tz_localize(None)
            if idx == len(matched_sync) - 1:
                chunk_end = r.model.predict(
                    np.array(onix_ts[-1]).reshape(-1, 1)
                ).flatten()[0]
                chunk_end = io_api.to_datetime(chunk_end)
                if hasattr(chunk_end, "tz_localize"):
                    chunk_end = chunk_end.tz_localize(None)

            model_path = Path(tmpdir) / (
                ephys_file.stem + f"_{r.clock_start}.joblib"
            )
            joblib.dump(r.model, model_path)

            harp_start = r.name
            if hasattr(harp_start, "tz_localize"):
                harp_start = harp_start.tz_localize(None)
            sync_entries.append(
                {
                    "onix_ts_start": r.clock_start,
                    "onix_ts_end": r.clock_end,
                    "sync_model": model_path,
                    "harp_start": harp_start,
                }
            )

        chunk_entry = {
            **insertion_key,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "epoch_start": epoch_start,
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
        }
        chunk_table.insert1(chunk_entry)
        chunk_table.File.insert(
            [
                dict(
                    **chunk_entry,
                    directory_type="raw",
                    file_name=f.name,
                    file_path=f.relative_to(raw_dir).as_posix(),
                )
                for f in (ephys_file, clock_file)
            ],
            ignore_extra_fields=True,
        )
        chunk_table.SyncModel.insert(
            [{**chunk_entry, **sync_entry} for sync_entry in sync_entries],
            ignore_extra_fields=True,
        )


def create_probe_type(
    probe_type: str, manufacturer: str, probe_name: str,
    probe_type_table,
) -> None:
    """Create a new probe type with electrode geometry from probeinterface.

    Args:
        probe_type: Unique identifier for the probe type
        manufacturer: Probe manufacturer (e.g., "neuropixels")
        probe_name: Specific probe model name (e.g., "NP2004")
        probe_type_table: The ProbeType table class
    """
    import probeinterface as pi

    electrode_df = pi.get_probe(
        manufacturer=manufacturer, probe_name=probe_name
    ).to_dataframe()
    electrode_df.rename(
        columns={
            "contact_ids": "electrode_name",
            "shank_ids": "shank",
            "x": "x_coord",
            "y": "y_coord",
        },
        inplace=True,
    )
    electrode_df.shank = electrode_df.shank.apply(lambda x: x or 0)
    electrode_df["probe_type"] = probe_type
    electrode_df["electrode"] = electrode_df.index

    with probe_type_table.connection.transaction:
        probe_type_table.insert1(dict(probe_type=probe_type))
        probe_type_table.Electrode.insert(electrode_df, ignore_extra_fields=True)
