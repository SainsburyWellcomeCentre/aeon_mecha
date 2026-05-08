"""DataJoint schema for the ephys pipeline."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import datajoint as dj

from aeon.dj_pipeline import acquisition, get_schema_name
from aeon.dj_pipeline.utils.ephys_utils import (
    DEVICE_PROBE_TYPE_MAP,
    discover_epoch_probes,
    find_or_create_probe_insertion,
    get_probe_id,
    parse_epoch_metadata,
    read_probe_assignments,
)
from aeon.dj_pipeline.utils.ephys_utils import (
    create_probe_type as _create_probe_type,
)

schema = dj.Schema(get_schema_name("ephys"))
logger = dj.logger


# ---------------------------------------------------------------------------
# Module-level helpers (used by EphysSyncModel.ingest and EphysChunk.ingest_chunks)
# ---------------------------------------------------------------------------


def _resolve_raw_dir_and_epochs(
    experiment_name: str,
) -> "tuple[Path, dict[str, datetime]] | None":
    """Return (raw_dir, {epoch_dir_name: epoch_start}) or None if unavailable.

    Centralises the filesystem + DB lookup that both ingest paths need.
    """
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


def _harp_to_naive(seconds: float) -> datetime:
    """Convert HARP seconds-since-1904 to a timezone-naive datetime (DJ-compatible)."""
    from swc.aeon.io import api as io_api

    dt = io_api.to_datetime(float(seconds))
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) else dt


def _resolve_harp(sync_row: dict, onix_ts: int) -> datetime:
    """Compute the HARP equivalent of an ONIX timestamp from a SyncModel row.

    Fast path: exact match against observed onix_ts_start/onix_ts_end boundaries
    — return the stored harp datetime directly (no model load required).
    Slow path: download the attached model bytes and call LinearRegression.predict.

    Args:
        sync_row: A dict from EphysSyncModel.to_dicts() with all column values.
        onix_ts: ONIX hardware timestamp to convert.

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
    model = joblib.load(sync_row["sync_model"])
    harp_seconds = float(model.predict(np.array([[onix_ts]])).flatten()[0])
    return _harp_to_naive(harp_seconds)


@schema
class ProbeType(dj.Lookup):
    definition = """  # Type of probe, with specific electrodes geometry defined
    probe_type: varchar(32)  # e.g. neuropixels_1.0
    """

    class Electrode(dj.Part):
        definition = """  # Electrode site on a probe
        -> master
        electrode: int       # electrode id, starts at 0
        ---
        shank: int           # shank idx, starts at 0, advance left to right
        x_coord: float  # (um) x coordinate of the electrode within the probe
        y_coord: float  # (um) y coordinate of the electrode within the probe
        electrode_name='': varchar(64)  # name of the electrode (e.g. "A1", "B2", etc.)
        """


@schema
class Probe(dj.Lookup):
    definition = """  # An actual physical probe with unique identification
    probe: varchar(32)  # unique identifier for this model of probe (e.g. serial number)
    ---
    -> ProbeType
    probe_comment='' :  varchar(1000)  # comment about this probe (e.g. defective, etc.)
    """


@schema
class ElectrodeConfig(dj.Lookup):
    definition = """  # The electrode configuration on a given probe used for recording
    -> ProbeType
    electrode_config_name: varchar(32)  # e.g. 0-383
    ---
    electrode_config_description: varchar(4000)  # description of the electrode configuration
    electrode_config_hash: uuid  # hash of the electrode configuration
    """

    class Electrode(dj.Part):
        definition = """  # Electrodes used for recording
        -> master
        -> ProbeType.Electrode
        """


@schema
class TargetArea(dj.Lookup):
    definition = """
    target_area: varchar(32)  # e.g. hippocampus, amygdala
    """


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    -> acquisition.Experiment
    -> acquisition.Experiment.Subject
    insertion_number: int  # unique per (experiment, subject)
    ---
    -> Probe
    implantation_date=null: datetime(6)
    """


@schema
class InsertionTargetArea(dj.Manual):
    definition = """
    # Links a probe insertion to one or more target brain areas
    -> ProbeInsertion
    -> TargetArea
    """


@schema
class EphysEpoch(dj.Imported):
    definition = """
    # Per-epoch probe registry. Discovers probes, reads subject-probe mapping, records active insertions.
    -> acquisition.Epoch
    ---
    has_ephys: bool  # whether ephys data was found in this epoch
    n_probes: int    # number of probes discovered (0 if no ephys)
    """

    class Insertion(dj.Part):
        definition = """
        # Which ProbeInsertions are active in this epoch
        -> master
        -> ProbeInsertion
        ---
        probe_label: varchar(32)  # file label discovered from this epoch's files (e.g., "ProbeA")
        """

    def make(self, key: dict[str, Any]) -> None:
        """Discover ephys data in an epoch and register active probe insertions.

        For each epoch:
        1. Check for ephys device subdirectory (NeuropixelsV2Beta/ or NeuropixelsV2/)
        2. If found, discover probes from binary files and parse Metadata.yml
        3. Auto-create Probe entries (probe_type derived from device directory name)
        4. Read subject-probe mapping from probe_assignments.json or carry forward
        5. Create/validate ProbeInsertion entries (with subject)
        6. Insert EphysEpoch.Insertion Part rows

        Subject-probe mapping sources (in priority order):
        - Per-epoch file: probe_assignments.json in the epoch directory
        - Carry-forward: reuse mapping from the most recent EphysEpoch with has_ephys=True
        - Pre-existing: manually inserted ProbeInsertion matched by probe serial

        Args:
            key: Dictionary containing experiment_name and epoch_start
        """
        # Get epoch directory
        dir_type, epoch_dir = (acquisition.Epoch & key).fetch1("directory_type", "epoch_dir")
        if not epoch_dir:
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        data_dir = acquisition.Experiment.get_data_directory(key, dir_type)
        if data_dir is None:
            logger.warning(f"Data directory not found for {key}")
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        epoch_path = data_dir / epoch_dir

        # Discover probes
        device_name, _device_dir, probe_labels = discover_epoch_probes(epoch_path)
        if not probe_labels:
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return
        if device_name is None:  # shouldn't happen when probe_labels is non-empty
            raise RuntimeError(
                f"discover_epoch_probes returned probe_labels without device_name: {epoch_path}"
            )

        # Parse metadata for probe identity (serial numbers)
        metadata = parse_epoch_metadata(epoch_path)

        # Build probe info: {label: probe_id}
        probe_info = {label: get_probe_id(metadata, device_name, label) for label in probe_labels}

        # Auto-create Probe entries (probe_type derived from device directory name)
        probe_type = DEVICE_PROBE_TYPE_MAP.get(device_name)
        if probe_type is None:
            raise ValueError(
                f"Unknown device '{device_name}'. Cannot determine probe_type. "
                f"Known devices: {list(DEVICE_PROBE_TYPE_MAP.keys())}"
            )
        for label in probe_labels:
            probe_id = probe_info[label]
            if not (Probe & {"probe": probe_id}):
                if not (ProbeType & {"probe_type": probe_type}):
                    raise ValueError(
                        f"ProbeType '{probe_type}' not found. Create it first using "
                        f"create_probe_type('{probe_type}', ...)."
                    )
                Probe.insert1(
                    {"probe": probe_id, "probe_type": probe_type},
                    skip_duplicates=True,
                )
                logger.info(f"Auto-created Probe entry: {probe_id} (type={probe_type})")

        # Read subject-probe mapping
        probe_assignments = read_probe_assignments(
            key,
            epoch_path,
            probe_labels,
            self.Insertion,
        )

        # Create/validate ProbeInsertion entries and build Insertion Part rows
        insertion_entries = []
        for label in probe_labels:
            probe_id = probe_info[label]
            subject = probe_assignments[label]["subject"]

            pi_key = find_or_create_probe_insertion(
                key["experiment_name"],
                subject,
                probe_id,
                ProbeInsertion,
                Probe,
            )
            insertion_entries.append(
                {
                    **key,
                    **pi_key,
                    "probe_label": label,
                }
            )

        # Insert master + Part rows
        self.insert1({**key, "has_ephys": True, "n_probes": len(probe_labels)})
        self.Insertion.insert(insertion_entries)


@schema
class EphysSyncModel(dj.Manual):
    """Per-chunk HARP↔ONIX sync regression for an ephys epoch.

    One row per ``HarpSync_*.csv`` (one per ONIX chunk window). Both HARP and
    ONIX bounds are observed values from the CSV (not predicted) — stable
    across re-ingestion.
    """

    definition = """
    -> EphysEpoch
    sync_start: datetime(6)            # PK — observed harp_time[0] from CSV (master clock)
    ---
    sync_end: datetime(6)              # observed harp_time[-1] from CSV
    onix_ts_start: bigint              # observed clock[0] from CSV
    onix_ts_end: bigint                # observed clock[-1] from CSV
    sync_model: <attach>               # joblib-serialized LinearRegression (onix→harp)
    r2: float                          # regression fit quality
    n_samples: int                     # rows in CSV after dropna()
    unique index (experiment_name, epoch_start, onix_ts_start)
    """

    @classmethod
    def ingest(cls, experiment_name: str) -> None:
        """Discover new HarpSync CSVs across all epochs of the experiment and insert sync model rows.

        Idempotent: skips CSVs whose ``(experiment_name, epoch_start, sync_start)``
        is already present.

        Args:
            experiment_name: Name of the experiment to process
        """
        import tempfile

        import joblib

        from aeon.schema.ephys import social_ephys

        resolved = _resolve_raw_dir_and_epochs(experiment_name)
        if resolved is None:
            return
        raw_dir, epoch_dir_to_start = resolved

        csvs = sorted(raw_dir.rglob("*_HarpSync_*.csv"))
        for csv_path in csvs:
            rel_parts = csv_path.relative_to(raw_dir).parts
            # Expect: epoch_dir / device_name / file → 3 parts minimum
            if len(rel_parts) < 3:
                continue
            epoch_dir_name = rel_parts[0]
            device_name = rel_parts[1]

            epoch_start = epoch_dir_to_start.get(epoch_dir_name)
            if epoch_start is None:
                logger.warning(
                    f"Cannot resolve epoch for {csv_path} "
                    f"(epoch_dir={epoch_dir_name} not in Epoch table). Skipping."
                )
                continue

            if device_name not in social_ephys:
                logger.debug(f"Device '{device_name}' not in social_ephys. Skipping {csv_path}.")
                continue
            device_streams = social_ephys[device_name]
            if "HarpSyncModel" not in device_streams:
                logger.debug(f"Device '{device_name}' has no HarpSyncModel stream. Skipping {csv_path}.")
                continue
            reader = device_streams["HarpSyncModel"]

            df_row = reader.read(csv_path).iloc[0]
            sync_start_dt = _harp_to_naive(df_row["harp_start"])

            existing = cls & {
                "experiment_name": experiment_name,
                "epoch_start": epoch_start,
                "sync_start": sync_start_dt,
            }
            if existing:
                continue

            sync_end_dt = _harp_to_naive(df_row["harp_end"])

            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / f"{csv_path.stem}.joblib"
                joblib.dump(df_row["model"], model_path)
                cls.insert1(
                    {
                        "experiment_name": experiment_name,
                        "epoch_start": epoch_start,
                        "sync_start": sync_start_dt,
                        "sync_end": sync_end_dt,
                        "onix_ts_start": int(df_row["clock_start"]),
                        "onix_ts_end": int(df_row["clock_end"]),
                        "sync_model": str(model_path),
                        "r2": float(df_row["r2"]),
                        "n_samples": int(df_row["n_samples"]),
                    }
                )
                logger.info(
                    f"Inserted EphysSyncModel: {experiment_name} "
                    f"epoch={epoch_start} sync_start={sync_start_dt}"
                )


@schema
class EphysChunk(dj.Manual):
    definition = """  # A recording period corresponds to a 1-hour ephys data acquisition
    -> ProbeInsertion                      # (experiment_name, subject, insertion_number)
    chunk_start: datetime(6)               # start of an ephys chunk (in HARP clock)
    ---
    -> EphysEpoch                          # adds epoch_start — "this chunk came from this epoch"
    chunk_end: datetime(6)                 # end of an ephys chunk (in HARP clock)
    -> ElectrodeConfig                     # the electrode configuration used for this ephys recording
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(128)
        ---
        -> acquisition.Experiment.Directory
        file_path: varchar(255)  # path of the file, relative to the data repository
        """

    class SyncModel(dj.Part):
        """Link-only: each EphysChunk references 1+ EphysSyncModel rows.

        The actual model bytes and ONIX bounds live on EphysSyncModel.
        Multiple link rows are inserted when an AmplifierData_N.bin straddles a
        HarpSync chunk boundary.
        """

        definition = """
        -> master
        -> EphysSyncModel
        """

    @classmethod
    def ingest_chunks(cls, experiment_name: str) -> None:
        """Ingest ephys recording chunks with clock synchronization.

        Discovers ephys binary files across all epochs, resolves each file's
        ProbeInsertion (with subject) via EphysEpoch.Insertion, and creates
        chunk entries with sync models.

        Files without a subject-probe mapping (no EphysEpoch.Insertion) are
        skipped with a warning.

        Args:
            experiment_name: Name of the experiment to process
        """
        import numpy as np

        resolved = _resolve_raw_dir_and_epochs(experiment_name)
        if resolved is None:
            return
        raw_dir, epoch_dir_to_start = resolved

        exp_key = {"experiment_name": experiment_name}

        # Build insertion lookup from EphysEpoch.Insertion:
        # {(epoch_start, probe_label): insertion_key}
        insertion_lookup: dict[tuple[datetime, str], dict] = {}
        for entry in (EphysEpoch.Insertion & exp_key).to_dicts():
            insertion_lookup[(entry["epoch_start"], entry["probe_label"])] = {
                "experiment_name": entry["experiment_name"],
                "subject": entry["subject"],
                "insertion_number": entry["insertion_number"],
            }

        if not insertion_lookup:
            logger.warning(
                f"No EphysEpoch.Insertion entries found for {experiment_name}. "
                "Run EphysEpoch.populate() first."
            )
            return

        # Discover ALL ephys binary files across epochs
        all_ephys_files = sorted(
            raw_dir.rglob("*_AmplifierData*.bin"),
            key=lambda x: x.as_posix(),
        )

        if not all_ephys_files:
            logger.info(f"No ephys amplifier files found in {raw_dir}")
            return

        for ephys_file in all_ephys_files:
            rel_path = ephys_file.relative_to(raw_dir).as_posix()

            # Skip already-ingested files
            if cls.File & exp_key & {"file_path": rel_path}:
                continue

            # Parse probe_label from filename
            name_match = re.search(r"_(Probe[A-Z])_AmplifierData", ephys_file.name)
            if not name_match:
                logger.warning(f"Cannot parse probe label from {ephys_file.name}. Skipping.")
                continue
            probe_label = name_match.group(1)

            # Determine epoch from directory path
            # File path structure: raw_dir / epoch_dir / device_name / files
            rel_parts = ephys_file.relative_to(raw_dir).parts
            # Expect: epoch_dir / device_name / file → 3 parts minimum
            if len(rel_parts) < 3:
                logger.warning(f"Unexpected file path structure: {ephys_file}. Skipping.")
                continue
            epoch_dir_name = rel_parts[0]

            epoch_start = epoch_dir_to_start.get(epoch_dir_name)
            if epoch_start is None:
                logger.warning(
                    f"Cannot resolve epoch for {ephys_file} "
                    f"(epoch_dir={epoch_dir_name} not found in Epoch table). Skipping."
                )
                continue

            # Look up ProbeInsertion via EphysEpoch.Insertion
            insertion_key = insertion_lookup.get((epoch_start, probe_label))
            if insertion_key is None:
                logger.warning(
                    f"Skipping {rel_path}: no subject-probe mapping for {probe_label} "
                    f"in epoch {epoch_start}. Register via probe_assignments.json or "
                    f"manual ProbeInsertion insert, then run EphysEpoch.populate()."
                )
                continue

            # Read ONIX timestamps from the companion Clock binary
            clock_file = ephys_file.with_name(ephys_file.name.replace("AmplifierData", "Clock"))
            if not clock_file.exists():
                logger.warning(f"Clock file not found for {ephys_file.name}. Skipping.")
                continue
            onix_ts = np.memmap(clock_file, mode="r", dtype=np.uint64)
            if len(onix_ts) == 0:
                logger.warning(f"Empty Clock file for {ephys_file.name}. Skipping.")
                continue
            first_ts, last_ts = int(onix_ts[0]), int(onix_ts[-1])

            # Query EphysSyncModel rows that cover the first OR last ONIX timestamp
            matched = (
                EphysSyncModel
                & {"experiment_name": experiment_name, "epoch_start": epoch_start}
                & (
                    f"({first_ts} BETWEEN onix_ts_start AND onix_ts_end) "
                    f"OR ({last_ts} BETWEEN onix_ts_start AND onix_ts_end)"
                )
            ).to_dicts(order_by="sync_start")

            if not matched:
                logger.warning(
                    f"No EphysSyncModel row covers ONIX range [{first_ts}, {last_ts}] "
                    f"for {ephys_file.name}. Run EphysSyncModel.ingest() first. Skipping."
                )
                continue

            # Resolve HARP chunk_start / chunk_end via DB-backed sync model
            try:
                chunk_start = _resolve_harp(matched[0], first_ts)
                chunk_end = _resolve_harp(matched[-1], last_ts)
            except Exception as e:
                logger.error(f"Failed to resolve HARP times for {ephys_file}: {e}")
                continue

            # Resolve electrode config (single config per probe_type required)
            probe_name = (ProbeInsertion & insertion_key).fetch1("probe")
            probe_type = (Probe & {"probe": probe_name}).fetch1("probe_type")
            configs = (ElectrodeConfig & {"probe_type": probe_type}).to_arrays("electrode_config_name")
            if len(configs) == 0:
                raise ValueError(f"No electrode configs found for probe_type={probe_type}")
            elif len(configs) > 1:
                raise ValueError(
                    f"Multiple electrode configs for {probe_type}: {configs}. "
                    "Please specify electrode_config_name."
                )
            electrode_config_name = configs[0]

            chunk_entry = {
                **insertion_key,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "epoch_start": epoch_start,
                "probe_type": probe_type,
                "electrode_config_name": electrode_config_name,
            }
            try:
                cls.insert1(chunk_entry)
                cls.File.insert(
                    [
                        {
                            **chunk_entry,
                            "directory_type": "raw",
                            "file_name": f.name,
                            "file_path": f.relative_to(raw_dir).as_posix(),
                        }
                        for f in (ephys_file, clock_file)
                    ],
                    ignore_extra_fields=True,
                )
                cls.SyncModel.insert(
                    [{**chunk_entry, "sync_start": m["sync_start"]} for m in matched],
                    ignore_extra_fields=True,
                )
                logger.info(
                    f"Inserted EphysChunk: {experiment_name} "
                    f"subject={insertion_key['subject']} "
                    f"chunk_start={chunk_start}"
                )
            except Exception as e:
                logger.error(f"Failed to insert EphysChunk for {ephys_file}: {e}")
                continue


@schema
class EphysBlock(dj.Manual):
    """User-defined period of time of ephys data (in HARP clock)."""

    definition = """  # An arbitrary period of time of ephys data
    -> ProbeInsertion
    block_start: datetime(6)  # start of an ephys block (in synced clock - i.e. HARP clock)
    block_end: datetime(6)    # end of an ephys block (in synced clock - i.e. HARP clock)
    """


@schema
class EphysBlockInfo(dj.Imported):
    definition = """
    -> EphysBlock
    ---
    block_duration: float  # (hour)
    -> ElectrodeConfig
    """

    class Chunk(dj.Part):
        definition = """ # the chunk(s) associated with this EphysBlock
        -> master
        -> EphysChunk
        """

    class Channel(dj.Part):
        definition = """  # Electrode-channel mapping
        -> master
        channel_idx: int  # channel idx (idx of the raw data)
        ---
        -> ElectrodeConfig.Electrode
        channel_name="": varchar(64)  # alias of the channel
        """

    def make(self, key: dict[str, Any]) -> None:
        """Compute ephys block metadata and channel mappings.

        Finds relevant ephys chunks for the given block and extracts:
        - Chunk associations (may span more chunks due to clock sync)
        - Electrode configuration validation
        - Channel-to-electrode mappings
        - Block duration calculations

        Args:
            key: Dictionary containing experiment_name, subject, insertion_number, block_start, block_end
        """

        def create_ephys_chunk_restriction(start_time: datetime, end_time: datetime) -> str:
            """Create SQL restriction for chunks overlapping the time window.

            Args:
                start_time: Block start time
                end_time: Block end time

            Returns:
                SQL WHERE clause string for chunk filtering
            """
            start_restriction = f'"{start_time}" BETWEEN chunk_start AND chunk_end'
            end_restriction = f'"{end_time}" BETWEEN chunk_start AND chunk_end'
            start_query = EphysChunk & key & start_restriction
            end_query = EphysChunk & key & end_restriction
            if not start_query:
                # No chunk contains the start time; find first chunk ending after start.
                start_query = EphysChunk & key & f'chunk_start BETWEEN "{start_time}" AND "{end_time}"'
            if not end_query:
                # No chunk contains the end time; find last chunk starting before end.
                end_query = EphysChunk & key & f'chunk_end BETWEEN "{start_time}" AND "{end_time}"'
            if not (start_query and end_query):
                raise ValueError(f"No Chunk found between {start_time} and {end_time}")
            time_restriction = (
                f'chunk_start >= "{min(start_query.to_arrays("chunk_start"))}"'
                f' AND chunk_start < "{max(end_query.to_arrays("chunk_end"))}"'
            )
            return time_restriction

        chunk_restriction = create_ephys_chunk_restriction(key["block_start"], key["block_end"])
        chunk_query = EphysChunk & key & chunk_restriction

        # compute total chunk duration (reserved for future validation)
        _chunk_total_duration = float(
            sum(
                chunk_query.proj(dur="TIMESTAMPDIFF(SECOND, chunk_start, chunk_end) / 3600").to_arrays(
                    "dur"
                )
            )
        )

        block_duration = (key["block_end"] - key["block_start"]).total_seconds() / 3600.0  # in hours

        # Read electrode config from the chunks in this block (set during ingest_chunks)
        probe_type, electrode_config_name = (chunk_query & dj.Top(limit=1)).fetch1(
            "probe_type", "electrode_config_name"
        )
        econfig = {
            "probe_type": probe_type,
            "electrode_config_name": electrode_config_name,
        }

        self.insert1(
            {**key, "block_duration": block_duration, **econfig},
        )
        # EphysChunk
        self.Chunk.insert(
            chunk_query.proj(block_start=f"'{key['block_start']}'", block_end=f"'{key['block_end']}'")
        )

        # Channel
        electrode_df = (ElectrodeConfig.Electrode & econfig).keys(order_by="electrode")
        self.Channel.insert(
            (
                {**key, "channel_idx": ch_idx, "channel_name": ch_idx, **ch_key}
                for ch_idx, ch_key in enumerate(electrode_df)
            ),
        )


def create_probe_type(probe_type: str, manufacturer: str, probe_name: str) -> None:
    """Create a new probe type with electrode geometry from probeinterface.

    Args:
        probe_type: Unique identifier for the probe type
        manufacturer: Probe manufacturer (e.g., "neuropixels")
        probe_name: Specific probe model name (e.g., "NP2004")
    """
    _create_probe_type(probe_type, manufacturer, probe_name, ProbeType)
