import datajoint as dj
import json
import re
import numpy as np
from pathlib import Path
import joblib
import tempfile
from typing import Dict, Any
from datetime import datetime

from swc.aeon.io import api as io_api
from aeon.schema.ephys import social_ephys

from aeon.dj_pipeline import acquisition, get_schema_name

schema = dj.schema(get_schema_name("ephys"))
logger = dj.logger


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
    target_area: varchar(32)  # e.g. "hippocampus", "amygdala"
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

    # Mapping from device directory name to probe_type
    _device_probe_type_map = {
        "NeuropixelsV2Beta": "neuropixels2.0_beta",
        "NeuropixelsV2": "neuropixels2.0",
    }

    def make(self, key: Dict[str, Any]) -> None:
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
        dir_type, epoch_dir = (acquisition.Epoch & key).fetch1(
            "directory_type", "epoch_dir"
        )
        if not epoch_dir:
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        data_dir = acquisition.Experiment.get_data_directory(key, dir_type)
        if data_dir is None:
            logger.warning(f"Data directory not found for {key}")
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        epoch_path = data_dir / epoch_dir

        # Look for ephys device subdirectory
        device_name = None
        device_dir = None
        for subdir_name in ("NeuropixelsV2Beta", "NeuropixelsV2"):
            candidate = epoch_path / subdir_name
            if candidate.exists() and candidate.is_dir():
                device_name = subdir_name
                device_dir = candidate
                break

        if device_name is None:
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        # Discover probes from binary files
        amplifier_files = sorted(device_dir.glob(f"{device_name}_Probe*_AmplifierData_*.bin"))
        if not amplifier_files:
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        # Extract unique probe labels from filenames
        # e.g., "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin" -> "ProbeA"
        probe_labels = set()
        for f in amplifier_files:
            match = re.search(rf"{device_name}_(Probe[A-Z])_AmplifierData", f.name)
            if match:
                probe_labels.add(match.group(1))

        if not probe_labels:
            logger.warning(f"Found amplifier files but couldn't parse probe labels in {device_dir}")
            self.insert1({**key, "has_ephys": False, "n_probes": 0})
            return

        probe_labels = sorted(probe_labels)

        # Parse Metadata.yml for probe identity (serial numbers)
        metadata_path = epoch_path / "Metadata.yml"
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)  # JSON despite .yml extension
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to parse Metadata.yml at {metadata_path}: {e}")

        # Build probe info: {label: probe_id}
        probe_info = {}
        for label in probe_labels:
            probe_info[label] = self._get_probe_id(metadata, device_name, label)

        # Auto-create Probe entries (probe_type derived from device directory name)
        probe_type = self._device_probe_type_map.get(device_name)
        if probe_type is None:
            raise ValueError(
                f"Unknown device '{device_name}'. Cannot determine probe_type. "
                f"Known devices: {list(self._device_probe_type_map.keys())}"
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
        probe_assignments = self._read_probe_assignments(key, epoch_path, probe_labels)

        # Create/validate ProbeInsertion entries and build Insertion Part rows
        insertion_entries = []
        for label in probe_labels:
            probe_id = probe_info[label]
            subject = probe_assignments[label]["subject"]

            # Find or create ProbeInsertion
            pi_key = self._find_or_create_probe_insertion(
                key["experiment_name"], subject, probe_id
            )
            insertion_entries.append({
                **key,
                **pi_key,
                "probe_label": label,
            })

        # Insert master + Part rows
        self.insert1({**key, "has_ephys": True, "n_probes": len(probe_labels)})
        self.Insertion.insert(insertion_entries)

    def _read_probe_assignments(
        self, key: Dict[str, Any], epoch_path: Path, probe_labels: list[str]
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

        Returns:
            Dict mapping probe_label -> {"subject": str, ...}
        """
        raise NotImplementedError(
            "Probe-subject assignment resolution is not yet implemented. "
            "The exact file format and carry-forward logic will be determined "
            "once the experimental data conventions are finalized."
        )

    @staticmethod
    def _find_or_create_probe_insertion(
        experiment_name: str, subject: str, probe_id: str
    ) -> dict:
        """Find existing or create new ProbeInsertion for this (experiment, subject, probe).

        Returns the ProbeInsertion key: {experiment_name, subject, insertion_number}.

        Args:
            experiment_name: Experiment name
            subject: Subject name
            probe_id: Probe identifier (serial number)

        Returns:
            Dict with ProbeInsertion PK fields
        """
        # Check if a ProbeInsertion already exists for this probe + subject + experiment
        existing = (
            ProbeInsertion
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
            ProbeInsertion
            & {"experiment_name": experiment_name, "subject": subject}
        ).fetch("insertion_number")
        new_num = int(existing_nums.max()) + 1 if len(existing_nums) > 0 else 1

        ProbeInsertion.insert1({
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

    @staticmethod
    def _get_probe_id(
        metadata: dict | None, device_name: str, probe_label: str
    ) -> str:
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
        definition = """
        -> master
        onix_ts_start: bigint  # ONIX timestamp at the start of the sync
        ---
        onix_ts_end: bigint  # ONIX timestamp at the end of the sync
        sync_model: attach  # serialized file containing the sync model
        harp_start: datetime(6)  # HARP start time of the sync
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
        exp_key = {"experiment_name": experiment_name}
        raw_dir = acquisition.Experiment.get_data_directory(exp_key, directory_type="raw")
        if raw_dir is None:
            logger.error(f"Raw data directory not found for {experiment_name}")
            return

        # Build epoch lookup: {epoch_dir_name: epoch_start} from Epoch table
        epochs = (acquisition.Epoch & exp_key).fetch(
            "epoch_start", "epoch_dir", as_dict=True
        )
        epoch_dir_to_start = {}
        for ep in epochs:
            if ep["epoch_dir"]:
                # epoch_dir is relative path like "2024-01-15T10-00-00"
                # Extract the top-level directory name
                top_dir = Path(ep["epoch_dir"]).parts[0]
                epoch_dir_to_start[top_dir] = ep["epoch_start"]

        # Build insertion lookup from EphysEpoch.Insertion:
        # {(epoch_start, probe_label): insertion_key}
        insertion_lookup = {}
        insertion_entries = (EphysEpoch.Insertion & exp_key).fetch(as_dict=True)
        for entry in insertion_entries:
            lookup_key = (entry["epoch_start"], entry["probe_label"])
            insertion_lookup[lookup_key] = {
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
            list(raw_dir.rglob("*_AmplifierData*.bin")),
            key=lambda x: x.as_posix(),
        )

        if not all_ephys_files:
            logger.info(f"No ephys amplifier files found in {raw_dir}")
            return

        # Sync model cache (per parent directory)
        sync_models = {}

        for ephys_file in all_ephys_files:
            rel_path = ephys_file.relative_to(raw_dir).as_posix()

            # Skip already-ingested files
            if cls.File & exp_key & {"file_path": rel_path}:
                continue

            # Parse probe_label from filename
            # e.g., "NeuropixelsV2Beta_ProbeA_AmplifierData_0.bin" -> "ProbeA"
            name_match = re.search(r"_(Probe[A-Z])_AmplifierData", ephys_file.name)
            if not name_match:
                logger.warning(f"Cannot parse probe label from {ephys_file.name}. Skipping.")
                continue
            probe_label = name_match.group(1)

            # Determine epoch from directory path
            # File path structure: raw_dir / epoch_dir / device_name / files
            rel_parts = ephys_file.relative_to(raw_dir).parts
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
            lookup_key = (epoch_start, probe_label)
            insertion_key = insertion_lookup.get(lookup_key)
            if insertion_key is None:
                logger.warning(
                    f"Skipping {rel_path}: no subject-probe mapping for {probe_label} "
                    f"in epoch {epoch_start}. Register via probe_assignments.json or "
                    f"manual ProbeInsertion insert, then run EphysEpoch.populate()."
                )
                continue

            # Resolve electrode config
            probe_name = (ProbeInsertion & insertion_key).fetch1("probe")
            probe_type = (Probe & {"probe": probe_name}).fetch1("probe_type")
            configs = (ElectrodeConfig & {"probe_type": probe_type}).fetch(
                "electrode_config_name"
            )
            if len(configs) == 0:
                raise ValueError(f"No electrode configs found for probe_type={probe_type}")
            elif len(configs) > 1:
                raise ValueError(
                    f"Multiple electrode configs for {probe_type}: {configs}. "
                    "Please specify electrode_config_name."
                )
            electrode_config_name = configs[0]

            # Process the file into a chunk entry
            try:
                cls._process_ephys_file(
                    ephys_file=ephys_file,
                    raw_dir=raw_dir,
                    insertion_key=insertion_key,
                    epoch_start=epoch_start,
                    probe_label=probe_label,
                    probe_type=probe_type,
                    electrode_config_name=electrode_config_name,
                    sync_models=sync_models,
                )
            except Exception as e:
                logger.error(f"Failed to process {ephys_file}: {e}")
                continue

    @classmethod
    def _process_ephys_file(
        cls,
        ephys_file: Path,
        raw_dir: Path,
        insertion_key: dict,
        epoch_start: datetime,
        probe_label: str,
        probe_type: str,
        electrode_config_name: str,
        sync_models: dict,
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
                if idx == len(matched_sync) - 1:
                    chunk_end = r.model.predict(
                        np.array(onix_ts[-1]).reshape(-1, 1)
                    ).flatten()[0]
                    chunk_end = io_api.to_datetime(chunk_end)

                model_path = Path(tmpdir) / (
                    ephys_file.stem + f"_{r.clock_start}.joblib"
                )
                joblib.dump(r.model, model_path)

                sync_entries.append(
                    {
                        "onix_ts_start": r.clock_start,
                        "onix_ts_end": r.clock_end,
                        "sync_model": model_path,
                        "harp_start": r.name,
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
            cls.insert1(chunk_entry)
            cls.File.insert(
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
            cls.SyncModel.insert(
                [{**chunk_entry, **sync_entry} for sync_entry in sync_entries],
                ignore_extra_fields=True,
            )


@schema
class EphysBlock(dj.Manual):
    """
    User-defined period of time of ephys data (in HARP clock)
    """

    definition = """  # A an arbitrary period of time of ephys data
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

    def make(self, key: Dict[str, Any]) -> None:
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
                # No chunk contains the start time, need to find the first chunk that ends after the start time
                start_query = (
                    EphysChunk
                    & key
                    & f'chunk_start BETWEEN "{start_time}" AND "{end_time}"'
                )
            if not end_query:
                # No chunk contains the end time, need to find the last chunk that starts before the end time
                end_query = (
                    EphysChunk
                    & key
                    & f'chunk_end BETWEEN "{start_time}" AND "{end_time}"'
                )
            if not (start_query and end_query):
                raise ValueError(f"No Chunk found between {start_time} and {end_time}")
            time_restriction = (
                f'chunk_start >= "{min(start_query.fetch("chunk_start"))}"'
                f' AND chunk_start < "{max(end_query.fetch("chunk_end"))}"'
            )
            return time_restriction

        chunk_restriction = create_ephys_chunk_restriction(
            key["block_start"], key["block_end"]
        )
        chunk_query = EphysChunk & key & chunk_restriction

        # validate durations
        chunk_total_duration = float(
            sum(
                chunk_query.proj(
                    dur="TIMESTAMPDIFF(SECOND, chunk_start, chunk_end) / 3600"
                ).fetch("dur")
            )
        )

        block_duration = (
            key["block_end"] - key["block_start"]
        ).total_seconds() / 3600.0  # in hours

        # Read electrode config from the chunks in this block (set during ingest_chunks)
        first_chunk = chunk_query.fetch(
            "probe_type", "electrode_config_name", limit=1, as_dict=True
        )[0]
        econfig = {
            "probe_type": first_chunk["probe_type"],
            "electrode_config_name": first_chunk["electrode_config_name"],
        }

        self.insert1(
            {**key, "block_duration": block_duration, **econfig},
        )
        # EphysChunk
        self.Chunk.insert(
            chunk_query.proj(
                block_start=f"'{key['block_start']}'", block_end=f"'{key['block_end']}'"
            )
        )

        # Channel
        electrode_df = (ElectrodeConfig.Electrode & econfig).fetch(
            "KEY", order_by="electrode"
        )
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

    with ProbeType.connection.transaction:
        ProbeType.insert1(dict(probe_type=probe_type))
        ProbeType.Electrode.insert(electrode_df, ignore_extra_fields=True)
