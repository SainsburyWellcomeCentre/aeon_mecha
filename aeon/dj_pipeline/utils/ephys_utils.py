"""Utility functions for the ephys pipeline.

Helpers for probe discovery, subject-probe mapping, probe type creation,
and ONIX/HARP timestamp resolution.
Used by ephys.py table classes.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import datajoint as dj

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
                    except Exception:  # noqa: BLE001
                        logger.debug(f"Failed to extract serial from GainCalibrationFileName: {gain_cal}")

    return default_id


def find_or_create_probe_insertion(
    experiment_name: str,
    subject: str,
    probe_id: str,
    probe_insertion_table,
    probe_table,
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
        probe_insertion_table & {"experiment_name": experiment_name, "subject": subject, "probe": probe_id}
    ).to_dicts()
    if existing:
        row = existing[0]
        return {
            "experiment_name": row["experiment_name"],
            "subject": row["subject"],
            "insertion_number": row["insertion_number"],
        }

    # Create new ProbeInsertion with auto-assigned insertion_number
    existing_nums = (
        probe_insertion_table & {"experiment_name": experiment_name, "subject": subject}
    ).to_arrays("insertion_number")
    new_num = int(existing_nums.max()) + 1 if len(existing_nums) > 0 else 1

    probe_insertion_table.insert1(
        {
            "experiment_name": experiment_name,
            "subject": subject,
            "insertion_number": new_num,
            "probe": probe_id,
        }
    )
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
    key: dict,
    epoch_path: Path,
    probe_labels: list[str],
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

    if device_name is None or device_dir is None:
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
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse Metadata.yml at {metadata_path}: {e}")
        return None


def create_probe_type(
    probe_type: str,
    manufacturer: str,
    probe_name: str,
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

    electrode_df = pi.get_probe(manufacturer=manufacturer, probe_name=probe_name).to_dataframe()
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
        probe_type_table.insert1({"probe_type": probe_type})
        probe_type_table.Electrode.insert(electrode_df, ignore_extra_fields=True)


# ---------------------------------------------------------------------------
# ONIX/HARP timestamp resolution helpers
# (shared by EphysSyncModel.ingest and EphysChunk.ingest_chunks in ephys.py)
# ---------------------------------------------------------------------------


def resolve_raw_dir_and_epochs(
    experiment_name: str,
) -> "tuple[Path, dict[str, datetime]] | None":
    """Return (raw_dir, {epoch_dir_name: epoch_start}) or None if unavailable.

    Centralises the filesystem + DB lookup that both ingest paths need.
    """
    from aeon.dj_pipeline import acquisition

    exp_key = {"experiment_name": experiment_name}
    raw_dir_result = acquisition.Experiment.get_data_directory(exp_key, directory_type="raw")
    if raw_dir_result is None:
        logger.error(f"Raw data directory not found for {experiment_name}")
        return None
    raw_dir = Path(raw_dir_result)

    epoch_dir_to_start: dict[str, datetime] = {}
    for ep in (acquisition.Epoch & exp_key).proj("epoch_start", "epoch_dir").to_dicts():
        if ep["epoch_dir"]:
            top_dir = Path(ep["epoch_dir"]).parts[0]
            epoch_dir_to_start[top_dir] = ep["epoch_start"]

    return raw_dir, epoch_dir_to_start


def harp_to_naive(seconds: float) -> datetime:
    """Convert HARP seconds-since-1904 to a timezone-naive datetime (DJ-compatible)."""
    from swc.aeon.io import api as io_api

    dt = io_api.to_datetime(float(seconds))
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def resolve_harp(sync_row: dict, onix_ts: int, _model_cache: "dict | None" = None) -> datetime:
    """Compute the HARP equivalent of an ONIX timestamp from a SyncModel row.

    Fast path: exact match against observed onix_ts_start/onix_ts_end boundaries
    — return the stored harp datetime directly (no model load required).
    Slow path: download the attached model bytes and call LinearRegression.predict.
    Caches loaded models in ``_model_cache`` if provided, keyed on
    ``sync_row["sync_start"]``, to avoid reloading the same model across multiple
    calls within a single ingestion iteration.

    Args:
        sync_row: A dict from EphysSyncModel.to_dicts() with all column values.
        onix_ts: ONIX hardware timestamp to convert.
        _model_cache: Optional dict for caching loaded joblib models within one
            ingestion iteration. Keyed on sync_row["sync_start"].

    Returns:
        Timezone-naive datetime representing the HARP equivalent.
    """
    import joblib
    import numpy as np

    if onix_ts == int(sync_row["onix_ts_start"]):
        harp_dt = sync_row["sync_start"]
        return harp_dt.replace(tzinfo=None) if getattr(harp_dt, "tzinfo", None) else harp_dt
    if onix_ts == int(sync_row["onix_ts_end"]):
        harp_dt = sync_row["sync_end"]
        return harp_dt.replace(tzinfo=None) if getattr(harp_dt, "tzinfo", None) else harp_dt

    cache_key = sync_row["sync_start"]
    if _model_cache is not None and cache_key in _model_cache:
        model = _model_cache[cache_key]
    else:
        model = joblib.load(sync_row["sync_model"])
        if _model_cache is not None:
            _model_cache[cache_key] = model

    harp_seconds = float(model.predict(np.array([[onix_ts]])).flatten()[0])
    return harp_to_naive(harp_seconds)
