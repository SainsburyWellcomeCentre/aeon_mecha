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


def _resolve_harp(sync_row: dict, onix_ts: int, _model_cache: "dict | None" = None) -> datetime:
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

        # Precompute probe→config cache: one set of 3 queries per unique insertion_key
        # rather than 3 queries per AmplifierData file.
        probe_config_cache: dict[tuple, tuple[str, str]] = {}
        for ins in insertion_lookup.values():
            cache_key = (ins["experiment_name"], ins["subject"], ins["insertion_number"])
            if cache_key in probe_config_cache:
                continue
            probe_name = (ProbeInsertion & ins).fetch1("probe")
            probe_type = (Probe & {"probe": probe_name}).fetch1("probe_type")
            configs = (ElectrodeConfig & {"probe_type": probe_type}).to_arrays("electrode_config_name")
            if len(configs) == 0:
                raise ValueError(f"No electrode configs found for probe_type={probe_type}")
            if len(configs) > 1:
                raise ValueError(
                    f"Multiple electrode configs for {probe_type}: {configs}. "
                    "Please specify electrode_config_name."
                )
            probe_config_cache[cache_key] = (probe_type, configs[0])

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

            # Resolve HARP chunk_start / chunk_end via DB-backed sync model.
            # Use a per-file model cache so both calls share one joblib.load when
            # matched[0] and matched[-1] are the same SyncModel row.
            model_cache: dict = {}
            try:
                chunk_start = _resolve_harp(matched[0], first_ts, _model_cache=model_cache)
                chunk_end = _resolve_harp(matched[-1], last_ts, _model_cache=model_cache)
            except Exception as e:
                logger.error(f"Failed to resolve HARP times for {ephys_file}: {e}")
                continue

            # Resolve electrode config from precomputed cache (no DB queries per file)
            cache_key = (
                insertion_key["experiment_name"],
                insertion_key["subject"],
                insertion_key["insertion_number"],
            )
            probe_type, electrode_config_name = probe_config_cache[cache_key]

            chunk_entry = {
                **insertion_key,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "epoch_start": epoch_start,
                "probe_type": probe_type,
                "electrode_config_name": electrode_config_name,
            }
            try:
                with cls.connection.transaction:
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


@schema
class OnixImuChunk(dj.Imported):
    """One row per ONIX sync window with all four Bno055 streams merged on sample index.

    The codec column ``stream_df`` returns an ONIX-clock-indexed DataFrame.
    For HARP-indexed data, use :meth:`OnixImuChunk.synced_df`.
    """

    definition = """
    -> EphysSyncModel
    ---
    sample_count: int
    timestamps: json
    euler_x: json
    euler_y: json
    euler_z: json
    gravity_vector_x: json
    gravity_vector_y: json
    gravity_vector_z: json
    linear_acceleration_x: json
    linear_acceleration_y: json
    linear_acceleration_z: json
    quaternion_w: json
    quaternion_x: json
    quaternion_y: json
    quaternion_z: json
    stream_df: <aeon_onix_stream>
    """

    @classmethod
    def synced_df(cls, key):
        """Fetch stream_df for a single chunk and apply its HARP sync regression.

        Args:
            key: A complete OnixImuChunk primary key dict — must resolve to
                exactly one row. PK fields: ``experiment_name``, ``epoch_start``,
                ``sync_start``. ``fetch1()`` raises if the key resolves to zero
                or multiple rows.

        Returns:
            HARP-time-indexed DataFrame with the columns in
            :data:`aeon.dj_pipeline.utils.onix_imu.IMU_COLUMNS`.

        For raw ONIX-clock-indexed data, fetch ``stream_df`` directly instead
        of using this helper.
        """
        import joblib
        import pandas as pd

        df = (cls & key).fetch1("stream_df")
        sync_attach = (EphysSyncModel & key).fetch1("sync_model")
        model = joblib.load(sync_attach)
        harp_seconds = model.predict(df.index.values.reshape(-1, 1)).flatten()
        df.index = pd.to_datetime([_harp_to_naive(s) for s in harp_seconds])
        return df

    def make(self, key):
        """Populate one OnixImuChunk row per EphysSyncModel key.

        No-IMU rigs (Bno055 files absent) get a row with ``sample_count=0`` and
        empty stat dicts. The ``stream_df`` reference is still valid; the codec
        returns an empty DataFrame on fetch.
        """
        import joblib
        import pandas as pd

        from aeon.dj_pipeline.utils.onix_imu import (
            IMU_COLUMNS,
            load_and_merge_bno055,
            locate_bno055_chunk_index,
        )
        from aeon.dj_pipeline.utils.stats import column_stats, timestamp_stats

        sm = (EphysSyncModel & key).proj("onix_ts_start", "sync_model").fetch1()
        epoch_dir = (acquisition.Epoch & key).fetch1("epoch_dir")
        raw_dir_result = acquisition.Experiment.get_data_directory(
            {"experiment_name": key["experiment_name"]}, "raw"
        )
        if raw_dir_result is None:
            raise FileNotFoundError(
                f"No raw data directory registered for experiment {key['experiment_name']!r}"
            )
        raw_dir = Path(raw_dir_result)
        epoch_path = raw_dir / epoch_dir

        # Discover device directory
        device_name = None
        for candidate in ("NeuropixelsV2Beta", "NeuropixelsV2"):
            if (epoch_path / candidate).is_dir():
                device_name = candidate
                break

        stream_df_ref = {
            "experiment_name": key["experiment_name"],
            "epoch_start": str(key["epoch_start"]),
            "sync_start": str(key["sync_start"]),
            "device_name": device_name or "NeuropixelsV2Beta",
            "stream_group": "Bno055",
        }

        if device_name is None:
            self.insert1(
                {
                    **key,
                    "sample_count": 0,
                    "timestamps": {},
                    **{col: {} for col in IMU_COLUMNS},
                    "stream_df": stream_df_ref,
                }
            )
            return

        device_dir = epoch_path / device_name
        chunk_index = locate_bno055_chunk_index(device_dir, device_name, int(sm["onix_ts_start"]))

        if chunk_index is None:
            self.insert1(
                {
                    **key,
                    "sample_count": 0,
                    "timestamps": {},
                    **{col: {} for col in IMU_COLUMNS},
                    "stream_df": stream_df_ref,
                }
            )
            return

        df = load_and_merge_bno055(device_dir, device_name, chunk_index)

        # Apply regression to compute HARP timestamps for the timestamps summary field.
        # The stored stream_df ref points to the ONIX-indexed DataFrame — sync is
        # applied only when users call OnixImuChunk.synced_df().
        model = joblib.load(sm["sync_model"])
        onix_clock = df.index.values
        harp_seconds = model.predict(onix_clock.reshape(-1, 1)).flatten()
        harp_index = pd.to_datetime([_harp_to_naive(s) for s in harp_seconds])

        self.insert1(
            {
                **key,
                "sample_count": len(df),
                "timestamps": timestamp_stats(harp_index),
                **{col: column_stats(df[col].values) for col in IMU_COLUMNS},
                "stream_df": stream_df_ref,
            }
        )
