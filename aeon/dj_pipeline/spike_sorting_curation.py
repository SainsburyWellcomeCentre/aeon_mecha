import os
import datajoint as dj
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

from aeon.dj_pipeline import ephys, spike_sorting, get_schema_name
from aeon.dj_pipeline.utils.paths import get_sorting_root_dir

# schema = dj.schema(get_schema_name("spike_sorting_curation"))
schema = dj.Schema()
logger = dj.logger


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # method/package used to perform manual curation (e.g. SpikeInterface, Phy, FigURL, etc.)
    """
    contents = [
        ("Phy", "SpikeInterface"),
    ]


@schema
class ManualCuration(dj.Manual):
    definition = """
    # Manual curation from a SortedSpikes
    -> spike_sorting.SpikeSorting
    curation_id: int
    ---
    curation_datetime: datetime    # UTC time when the curation was performed
    parent_curation_id=-1: int     # if -1, this curation is based on the raw spike sorting results
    -> CurationMethod              # which method/package used for manual curation (inform how to ingest the results)
    description="": varchar(1000)  # user-defined description/note of the curation
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@dj_store
        """


@schema
class OfficialCuration(dj.Manual):
    definition = """  # One final/official curation for a SortedSpikes
    -> spike_sorting.SortedSpikes
    ---
    -> ManualCuration
    """


@schema
class ApplyOfficialCuration(dj.Imported):
    definition = """
    -> OfficialCuration
    ---
    execution_time: datetime        # datetime of the start of this step
    new_unit_count: int             # number of new units added
    removed_unit_count: int         # number of units removed
    """

    def make(self, key):
        """
        Update/overwrite the SortedSpikes (and downstream) with the official curation results
        """
        pass
