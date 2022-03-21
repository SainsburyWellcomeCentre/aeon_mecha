import datajoint as dj
import pandas as pd
import numpy as np
import datetime

from aeon.preprocess import api as aeon_api
from aeon.util import utils as aeon_utils

from .. import lab, acquisition, tracking, qc
from .. import get_schema_name

# schema = dj.schema(get_schema_name('analysis'))
schema = dj.schema()


@schema
class Place(dj.Lookup):
    definition = """
    place: varchar(32)  # User-friendly name of a "place" animals can visit - e.g. nest, arena, foodpatch
    ---
    place_description: varchar(1000)
    """


@schema
class Visit(dj.Manual):
    definition = """
    -> acquisition.Experiment.Subject
    -> Place
    visit_start: datetime(6)
    """


@schema
class VisitEnd(dj.Manual):
    definition = """
    -> Visit
    ---
    visit_end: datetime(6)
    visit_duration: float
    """
