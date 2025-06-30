import datajoint as dj

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
        electrode: int       # electrode idx, starts at 0
        ---
        shank: int           # shank idx, starts at 0, advance left to right
        shank_col: int       # column idx, starts at 0, advance left to right
        shank_row: int       # row idx, starts at 0, advance bottom to top
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe
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

    def generate(self):
        """
        For each ephys file:
        1. look for start/end in ONIX
        2. map to start/end in HARP
        3. store the HARP start/end
        4. infer the probe type and electrode config
        """
        pass
    

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
        pass
