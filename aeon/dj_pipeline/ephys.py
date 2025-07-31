import datajoint as dj
import numpy as np
from pathlib import Path
import joblib
import tempfile

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
class EphysChunk(dj.Manual):
    definition = """  # A recording period corresponds to a 1-hour ephys data acquisition
    -> acquisition.Experiment
    -> Probe  # the probe used for this ephys recording
    chunk_start: datetime(6)  # start of an ephys chunk (in HARP clock)
    ---
    chunk_end: datetime(6)    # end of an ephys chunk (in HARP clock)
    -> ElectrodeConfig  # the electrode configuration used for this ephys recording
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
    def ingest_chunks(cls, experiment_name):
        """
        For each ephys file:
        1. look for start/end in ONIX
        2. map to start/end in HARP
        3. store the HARP start/end
        4. infer the probe type and electrode config
        """
        key = {"experiment_name": experiment_name}
        raw_dir = acquisition.Experiment.get_data_directory(key, directory_type="raw")
        ephys_files = sorted(
            list(raw_dir.rglob("*ProbeA_AmplifierData*.bin")),
            key=lambda x: int(x.stem.split("_")[-1]),
        )
        sync_models = {}

        def ephys_chunk_from_file(ephys_file):
            # hardcoded probe/electrode info here, should be retrieved from the file metadata
            probe_name = "NP2004-001"
            probe_type = "neuropixels - NP2004"
            electrode_config_name = "0-383"

            clock_file = ephys_file.with_name(
                ephys_file.name.replace("AmplifierData", "Clock")
            )
            onix_ts = np.memmap(clock_file, mode="r", dtype=np.uint64)

            model_parent_dir = raw_dir / ephys_file.relative_to(raw_dir).parents[-2]
            if model_parent_dir.as_posix() not in sync_models:
                # Load the sync model only once per parent directory
                sync_models[model_parent_dir.as_posix()] = io_api.load(
                    model_parent_dir,
                    social_ephys.NeuropixelsV2Beta.HarpSyncModel,
                )

            sync_model = sync_models[model_parent_dir.as_posix()]
            matched_sync = sync_model.query(
                f"(clock_start <= {onix_ts[0]} <= clock_end)"
                f" | "
                f"(clock_start <= {onix_ts[-1]} <= clock_end)"
            )

            sync_entries = []
            tmpdir = tempfile.TemporaryDirectory()
            for idx, (_, r) in enumerate(matched_sync.iterrows()):
                if idx == 0:
                    chunk_start = r.model.predict(
                        np.array(onix_ts[0]).reshape(-1, 1)
                    ).flatten()[0]
                if idx == len(matched_sync) - 1:
                    chunk_end = r.model.predict(
                        np.array(onix_ts[-1]).reshape(-1, 1)
                    ).flatten()[0]

                model_path = Path(tmpdir.name) / (
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
                **key,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "probe": probe_name,
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

        for ephys_file in ephys_files:
            rel_path = ephys_file.relative_to(raw_dir).as_posix()
            if cls.File & key & {"file_path": rel_path}:
                continue

            try:
                ephys_chunk_from_file(ephys_file)
            except Exception as e:
                logger.error(f"Failed to process {ephys_file}: {e}")
                continue


@schema
class EphysBlock(dj.Manual):
    """
    User-defined period of time of ephys data (in HARP clock)
    """

    definition = """  # A an arbitrary period of time of ephys data
    -> acquisition.Experiment
    -> Probe  # the probe used for this ephys recording
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

    def make(self, key):
        """
        - Find relevant ephys chunks for the given ephys block.
        - For each chunk, extract the start and end times in the native clock (i.e. ONIX clock).
            - For example: ephys block spans 3 chunks (3 hours)
            - Start/end of block is 2025-01-01 07:00:00, 2025-01-01 10:00:00
            - But due to the clock synchronization, there are actually 5 ephys chunks involved
            - So the start/end of the ephys block in the native clock is:
              2025-01-01 06:59:11 (start of first chunk), 2025-01-01 10:02:05 (end of last chunk)
        - Retrieve & confirm the electrode configuration for the given ephys block.
            - Ensure all associated ephys chunks have the same electrode configuration.
        - Extract electrode-channel mapping
        - Extract other metadata for this ephys block.
        """

        def create_ephys_chunk_restriction(start_time, end_time):
            """Create a time restriction string for the chunks between the specified "start" and "end" times."""
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

        # ElectrodeConfig & Channel - hardcode
        econfig = {
            "probe_type": "neuropixels - NP2004",
            "electrode_config_name": "0-383",
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


def create_probe_type(probe_type: str, manufacturer: str, probe_name: str):
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
