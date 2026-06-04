"""Utility functions for the ephys pipeline.

Helpers for probe discovery, subject-probe mapping, probe type creation,
and ONIX/HARP timestamp resolution.
Used by ephys.py table classes.
"""

import json
import re
from datetime import datetime
from pathlib import Path, PureWindowsPath

import datajoint as dj

logger = dj.logger

# Mapping from device directory name to probe_type
DEVICE_PROBE_TYPE_MAP = {
    "NeuropixelsV2Beta": "neuropixels2.0_beta",
    "NeuropixelsV2": "neuropixels2.0-multishank",
}


def get_probe_id(metadata: dict | None, device_name: str, probe_label: str) -> str | None:
    """Extract probe identifier from metadata.

    For V2Beta hardware (no serial numbers): probe ID = "{device_name}_{label}"
    For V2 hardware: probe ID = serial number from GainCalibrationFileName.
        Returns None if the probe is disabled (Devices.ProbeX = "false").

    Metadata structure (V2):
        metadata["Devices"]["NeuropixelsV2e"]["ConfigurationA/B"]["GainCalibrationFileName"]
        metadata["Devices"]["ProbeA/B"] = "true"/"false" (enable flag)

    Args:
        metadata: Parsed Metadata.yml dict, or None
        device_name: Hardware device name (e.g., "NeuropixelsV2Beta", "NeuropixelsV2")
        probe_label: Probe label from filename (e.g., "ProbeA", "ProbeB")

    Returns:
        Probe identifier string, or None for disabled probes.
    """
    default_id = f"{device_name}_{probe_label}"

    if metadata is None:
        return default_id

    devices = metadata.get("Devices", {})

    # V2 hardware: check enable flag, then extract serial from calibration path
    if device_name == "NeuropixelsV2":
        # Check if probe is enabled (Devices.ProbeA/B = "true"/"false")
        enabled = devices.get(probe_label, "true")
        if isinstance(enabled, str) and enabled.lower() == "false":
            return None

        v2e_config = devices.get("NeuropixelsV2e", {})
        config_key_map = {
            "ProbeA": "ConfigurationA",
            "ProbeB": "ConfigurationB",
        }
        config_key = config_key_map.get(probe_label)
        if config_key and config_key in v2e_config:
            probe_config = v2e_config[config_key]
            gain_cal = probe_config.get("GainCalibrationFileName")
            if gain_cal:
                try:
                    serial = PureWindowsPath(gain_cal).parent.name
                    if serial and serial.isdigit():
                        return serial
                except Exception:  # noqa: BLE001
                    logger.debug(f"Failed to extract serial from GainCalibrationFileName: {gain_cal}")

        return default_id

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
    probe_info: dict[str, str],
    *,
    override_dir: "Path | None" = None,
) -> dict:
    """Read subject-probe mapping from file or carry forward from previous epoch.

    Priority:
    1. probe_assignments.json in ``override_dir`` (if provided)
    2. probe_assignments.json in the epoch directory on Ceph
    3. Carry-forward from most recent EphysEpoch (same experiment) with Insertion entries

    Carry-forward is scoped to the current experiment (experiment_name in key).
    It cannot cross experiment boundaries.

    Args:
        key: Epoch key (experiment_name, epoch_start)
        epoch_path: Path to epoch directory on Ceph
        probe_labels: Sorted list of probe labels discovered in this epoch
        insertion_table: The EphysEpoch.Insertion Part table class
        probe_info: Dict mapping probe_label -> probe_serial (built by caller
            from get_probe_id()). Used to translate serial-keyed JSON to
            label-keyed dict.
        override_dir: Optional local directory to check first. Useful when
            the epoch directory is read-only (e.g. Ceph without write perms).

    Returns:
        Dict mapping probe_label -> {"subject": str}
    """
    # Priority 1: override directory (local fallback for read-only Ceph)
    if override_dir is not None:
        override_path = Path(override_dir) / "probe_assignments.json"
        if override_path.exists():
            logger.info(f"Reading probe assignments from override: {override_path}")
            return _parse_probe_assignments_file(override_path, probe_info)

    # Priority 2: JSON file in epoch directory
    json_path = epoch_path / "probe_assignments.json"
    if json_path.exists():
        return _parse_probe_assignments_file(json_path, probe_info)

    # Priority 3: Carry-forward from most recent epoch in same experiment
    previous_insertions = (
        insertion_table
        & {"experiment_name": key["experiment_name"]}
        & f'epoch_start < "{key["epoch_start"]}"'
    ).fetch(
        "epoch_start", "probe_label", "subject",
        as_dict=True, order_by="epoch_start DESC",
    )

    if previous_insertions:
        # Use only entries from the most recent epoch (first row, ordered DESC)
        latest_start = previous_insertions[0]["epoch_start"]
        carried = {
            row["probe_label"]: {"subject": row["subject"]}
            for row in previous_insertions
            if row["epoch_start"] == latest_start
        }

        if carried:
            logger.info(
                f"Carried forward probe assignments from epoch {latest_start} "
                f"for {key['experiment_name']}: {carried}"
            )
            return carried

    # Nothing found
    raise FileNotFoundError(
        f"No probe_assignments.json found in {epoch_path} and no previous "
        f"EphysEpoch with probe insertions found for experiment "
        f"'{key['experiment_name']}'. Create a probe_assignments.json file "
        f"in the first epoch directory with format:\n"
        f'{{\n  "version": 1,\n  "probe_assignments": {{\n'
        f'    "<probe_serial>": {{"subject": "<subject_name>"}}\n  }}\n}}'
    )


def _parse_probe_assignments_file(json_path: Path, probe_info: dict[str, str]) -> dict:
    """Parse probe_assignments.json and translate serial-keyed entries to label-keyed.

    Args:
        json_path: Path to the JSON file
        probe_info: Dict mapping probe_label -> probe_serial (from EphysEpoch.make)

    Returns:
        Dict mapping probe_label -> {"subject": str}
    """
    data = json.loads(json_path.read_text())

    if "version" not in data:
        raise ValueError(
            f"probe_assignments.json at {json_path} is missing 'version' field. "
            f'Expected format: {{"version": 1, "probe_assignments": {{...}}}}'
        )
    if data["version"] != 1:
        raise ValueError(
            f"probe_assignments.json at {json_path} has unsupported version "
            f"{data['version']}. Only version 1 is supported."
        )
    if "probe_assignments" not in data:
        raise ValueError(
            f"probe_assignments.json at {json_path} is missing 'probe_assignments' "
            f'field. Expected format: {{"version": 1, "probe_assignments": {{...}}}}'
        )

    assignments_by_serial = data["probe_assignments"]

    # Translate serial -> label using probe_info (label -> serial)
    result = {}
    for label, serial in probe_info.items():
        if serial not in assignments_by_serial:
            raise ValueError(
                f"Probe serial '{serial}' (label '{label}') not found in "
                f"{json_path}. Available serials: {list(assignments_by_serial.keys())}. "
                f"Add an entry for this serial in the probe_assignments section."
            )
        entry = assignments_by_serial[serial]
        if "subject" not in entry:
            raise ValueError(
                f"Entry for serial '{serial}' in {json_path} is missing 'subject' field."
            )
        result[label] = {"subject": entry["subject"]}

    return result


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


def parse_metadata_probe_configs(epoch_path: Path) -> dict[str, str | None]:
    """Extract per-probe config-file basenames from an epoch's Metadata.yml.

    Reads ``Devices.NeuropixelsV2e.ConfigurationA/B.ProbeInterfaceFileName``
    and converts each entry to a (probe_label, basename) pair.

    Args:
        epoch_path: Path to the epoch directory containing Metadata.yml.

    Returns:
        Dict mapping ``"ProbeA" | "ProbeB" | ...`` to JSON basename or None.
        None indicates a disabled/spoofed probe (ProbeInterfaceFileName is
        null in the metadata).

    Raises:
        FileNotFoundError: If Metadata.yml doesn't exist.
    """
    metadata_path = Path(epoch_path) / "Metadata.yml"
    with open(metadata_path) as f:
        data = json.load(f)  # JSON despite .yml extension

    npx = data.get("Devices", {}).get("NeuropixelsV2e", {})
    result: dict[str, str | None] = {}
    for cfg_key, cfg_value in npx.items():
        if not cfg_key.startswith("Configuration"):
            continue
        suffix = cfg_key[len("Configuration"):]  # "A", "B", ...
        probe_label = f"Probe{suffix}"
        if not isinstance(cfg_value, dict):
            result[probe_label] = None
            continue
        pifn = cfg_value.get("ProbeInterfaceFileName")
        if pifn is None or pifn == "":
            result[probe_label] = None
        else:
            # Path may be Windows-style; extract basename robustly
            basename = PureWindowsPath(pifn).name or Path(pifn).name
            result[probe_label] = basename
    return result


def resolve_probe_config_path(raw_ephys_dir, config_file_name: str) -> Path:
    """Resolve the absolute path to a per-epoch probeinterface JSON.

    The JSON lives in the rig's ``recording_configurations/`` directory,
    a sibling of the per-epoch dirs under ``raw_ephys_dir``.

    Args:
        raw_ephys_dir: The rig-level raw-ephys directory
            (e.g. ``/ceph/aeon/data/raw/AEONX1``). Accepts str or Path.
        config_file_name: Basename like
            ``"M81_ProbeB_4Shanks_1000_to_1700_um.json"``.

    Returns:
        Resolved Path. Existence is NOT checked here — callers decide whether
        to raise.
    """
    return Path(raw_ephys_dir) / "recording_configurations" / config_file_name


def load_device_channel_map(json_path: Path) -> dict[int, int]:
    """Read a probeinterface JSON and return the hardware channel mapping.

    The probeinterface JSON describes the full probe geometry (e.g. 5120
    contacts for a 4-shank NP2.0). The ``device_channel_indices`` array marks
    which contacts are actively recorded: values >= 0 give the raw binary
    column index (0-383 for 384-channel recordings), and -1 means inactive.

    Args:
        json_path: Path to the probeinterface JSON file (typically resolved
            from ``ElectrodeConfig.config_file_name`` under the rig's
            ``recording_configurations/`` directory).

    Returns:
        Dict mapping ``{electrode_site_id: raw_channel_idx}`` for all active
        contacts. For example, ``{3954: 0, 114: 210, ...}`` means raw binary
        column 0 records from electrode site 3954, and column 210 records
        from site 114.

    Raises:
        FileNotFoundError: If json_path doesn't exist.
        ValueError: If the JSON has no ``device_channel_indices`` or no
            active contacts.
    """
    with open(json_path) as f:
        pi_data = json.load(f)

    channel_map: dict[int, int] = {}
    for probe in pi_data.get("probes", []):
        contact_ids = probe.get("contact_ids", [])
        dci = probe.get("device_channel_indices")
        if dci is None:
            raise ValueError(
                f"No device_channel_indices in {json_path}. "
                f"Cannot determine hardware channel mapping."
            )
        for cid, ch_idx_raw in zip(contact_ids, dci, strict=False):
            ch_idx = int(ch_idx_raw)
            if ch_idx >= 0:
                channel_map[int(cid)] = ch_idx

    if not channel_map:
        raise ValueError(
            f"No active contacts (device_channel_indices >= 0) in {json_path}."
        )

    return channel_map


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


def create_electrode_config(
    json_path,
    probe_type_table,
    electrode_config_table,
    *,
    config_name: str | None = None,
    probe_type_name: str | None = None,
) -> tuple[str, str]:
    """Populate ProbeType + ElectrodeConfig from a per-epoch probeinterface JSON.

    Reads a per-epoch probe configuration JSON (probeinterface schema). Populates:

      - ProbeType + ProbeType.Electrode with the full contact geometry (e.g.
        5120 contacts for Neuropixels 2.0 multishank).
      - ElectrodeConfig + ElectrodeConfig.Electrode with the SUBSET of contacts
        where ``device_channel_indices != -1`` (the actively-recorded
        electrodes — 384 for a standard NP 2.0 single-bank readout).

    Idempotent: re-running with the same JSON is a no-op (skip_duplicates=True
    on all inserts). Mirrors create_probe_type's style — caller passes the
    table classes explicitly.

    Args:
        json_path: Path to the per-epoch probeinterface JSON file.
        probe_type_table: ProbeType table class.
        electrode_config_table: ElectrodeConfig table class.
        config_name: Override electrode_config_name. Defaults to json_path.stem.
        probe_type_name: Override probe_type. Default: canonical-cased version
            of ``annotations["name"]`` (e.g. "Neuropixels 2.0 - multishank" →
            "neuropixels2.0-multishank").

    Returns:
        (probe_type, electrode_config_name) — keys identifying the inserted rows.
    """
    import uuid

    import probeinterface as pi

    json_path = Path(json_path)
    probe_group = pi.read_probeinterface(str(json_path))
    if len(probe_group.probes) != 1:
        raise ValueError(
            f"Expected exactly one probe in {json_path}, got {len(probe_group.probes)}."
        )
    probe = probe_group.probes[0]

    if probe_type_name is None:
        raw_name = probe.annotations.get("name", "")
        probe_type_name = raw_name.lower().replace(" - ", "-").replace(" ", "")
    if not probe_type_name:
        raise ValueError(f"Cannot derive probe_type from {json_path} annotations")

    # Build electrode geometry dataframe for ProbeType.Electrode
    electrode_df = probe.to_dataframe()
    electrode_df.rename(
        columns={
            "contact_ids": "electrode_name",
            "shank_ids": "shank",
            "x": "x_coord",
            "y": "y_coord",
        },
        inplace=True,
    )
    electrode_df["shank"] = electrode_df["shank"].apply(lambda x: x if x else 0)
    electrode_df["probe_type"] = probe_type_name
    electrode_df["electrode"] = electrode_df.index

    with probe_type_table.connection.transaction:
        probe_type_table.insert1({"probe_type": probe_type_name}, skip_duplicates=True)
        probe_type_table.Electrode.insert(
            electrode_df, ignore_extra_fields=True, skip_duplicates=True
        )

    # ElectrodeConfig: subset where device_channel_indices != -1
    if config_name is None:
        config_name = json_path.stem

    dci = probe.device_channel_indices
    active_electrode_ids = [i for i, ch in enumerate(dci) if ch != -1]

    electrode_config_key = {
        "probe_type": probe_type_name,
        "electrode_config_name": config_name,
    }
    with electrode_config_table.connection.transaction:
        electrode_config_table.insert1(
            {
                **electrode_config_key,
                "electrode_config_description": f"From {json_path.name}",
                "electrode_config_hash": uuid.uuid4(),
                "config_file_name": json_path.name,
            },
            skip_duplicates=True,
        )
        electrode_config_table.Electrode.insert(
            ({**electrode_config_key, "electrode": int(e)} for e in active_electrode_ids),
            skip_duplicates=True,
        )

    return probe_type_name, config_name


# ---------------------------------------------------------------------------
# ONIX/HARP timestamp resolution helpers
# (shared by EphysSyncModel.ingest and EphysChunk.ingest_chunks in ephys.py)
# ---------------------------------------------------------------------------


def resolve_raw_dir_and_epochs(
    experiment_name: str,
) -> "tuple[Path, dict[str, datetime]] | None":
    """Return (raw_ephys_dir, {epoch_dir_name: epoch_start}) or None.

    Looks up the "raw-ephys" directory and returns only epochs with confirmed
    ephys data (``EphysEpochConfig.has_ephys == True``). Reads from
    ``ephys.EphysEpoch`` (peer of ``acquisition.Epoch`` post-#583).
    """
    from aeon.dj_pipeline import acquisition
    from aeon.dj_pipeline.ephys import EphysEpoch, EphysEpochConfig

    exp_key = {"experiment_name": experiment_name}
    raw_dir_result = acquisition.Experiment.get_data_directory(
        exp_key, directory_type="raw-ephys"
    )
    if raw_dir_result is None:
        logger.error(f"raw-ephys data directory not found for {experiment_name}")
        return None
    raw_dir = Path(raw_dir_result)

    # Only include epochs where EphysEpochConfig confirmed ephys data exists.
    ephys_epochs = (
        EphysEpoch & exp_key & (EphysEpochConfig & {"has_ephys": True})
    ).proj("epoch_dir").to_dicts()

    epoch_dir_to_start: dict[str, datetime] = {}
    for ep in ephys_epochs:
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
