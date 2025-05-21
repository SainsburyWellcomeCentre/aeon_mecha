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
        electrode: int       # electrode index, starts at 0
        ---
        shank: int           # shank index, starts at 0, advance left to right
        shank_col: int       # column index, starts at 0, advance left to right
        shank_row: int       # row index, starts at 0, advance bottom to top
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe
        """


@schema
class Probe(dj.Lookup):
    definition = """  # Represent a physical probe with unique identification
    probe: varchar(32)  # unique identifier for this model of probe (e.g. serial number)
    ---
    -> ProbeType
    probe_comment='' :  varchar(1000)  # comment about this probe (e.g. defective, etc.)
    """


@schema
class ElectrodeConfig(dj.Lookup):
    definition = """  # The electrode configuration setting on a given probe
    -> ProbeType
    electrode_config_name: varchar(32)  # e.g. "0-383"
    ---
    electrode_config_description: varchar(4000)  # description of the electrode configuration
    electrode_config_hash: uuid  # hash of the electrode configuration
    """

    class Electrode(dj.Part):
        definition = """  # Electrodes selected for recording
        -> master
        -> ProbeType.Electrode
        """


@schema
class EphysChunk(dj.Manual):
    definition = """  # A recording period corresponds to a 1-hour ephys data acquisition
    -> acquisition.Experiment
    -> Probe  # the probe used for this ephys recording
    chunk_start: datetime(6)  # start of an ephys chunk (in ONIX clock)
    ---
    chunk_end: datetime(6)    # end of an ephys chunk (in ONIX clock)
    -> ElectrodeConfig  # the electrode configuration used for this ephys recording
    """
    

@schema
class EphysBlock(dj.Manual):
    definition = """  # A an arbitrary period of time of ephys data
    -> acquisition.Experiment
    -> Probe  # the probe used for this ephys recording
    block_start: datetime(6)  # start of an ephys block (in native clock - e.g. ONIX clock)
    ---
    block_end: datetime(6)    # end of an ephys block (in native clock - e.g. ONIX clock)
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
        channel_idx: int  # channel index (index of the raw data)
        ---
        -> probe.ElectrodeConfig.Electrode
        channel_name="": varchar(64)  # alias of the channel
        """

    def make(self, key):
        """
        - Find relevant ephys chunks for the given ephys block.
        - Retrieve & confirm the electrode configuration for the given ephys block.
        - Extract electrode-channel mapping
        - Extract other metadata for this ephys block.
        """
        pass
