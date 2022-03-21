import datajoint as dj
import pandas as pd
import numpy as np
import datetime

from aeon.preprocess import api as aeon_api
from aeon.util import utils as aeon_utils

from .. import lab, acquisition, tracking, qc
from .. import get_schema_name

schema = dj.schema(get_schema_name('analysis'))


@schema
class Place(dj.Lookup):
    definition = """
    place: varchar(32)  # User-friendly name of a 'place' animals can visit - e.g. nest, arena, foodpatch
    ---
    place_description: varchar(1000)
    """

    contents = [
        ('environment', 'the entire environment where the animals undergo experiment (previously known as arena)')
    ]


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
    visit_duration: float  # (hour)
    """


# @schema
# class OverlapVisit(dj.Computed):
#     definition = """
#     -> Experiment
#     -> Place
#     overlap_start: datetime(6)
#     ---
#     overlap_end: datetime(6)
#     overlap_duration: float  # (hour)
#     subject_count: int
#     """
#
#     class Visit(dj.Part):
#         definition = """
#         -> master
#         -> Visit
#         """


# ---- HELPERS ----

def ingest_environment_visits(experiment_names=['exp0.2-r0']):
    """
    Function to ingest into `Visit` and `VisitEnd` for specified experiments (default: 'exp0.2-r0')
    Using api method: `aeon_api.subjectvisit`
    This ingestion routine handles only those "complete" visits, not ingesting any "on-going" visits
    """
    place_key = {'place': 'environment'}
    for experiment_name in experiment_names:
        exp_key = {'experiment_name': experiment_name}
        raw_data_dir = acquisition.Experiment.get_data_directory(exp_key)

        subjects_last_visits = (acquisition.Experiment.Subject & exp_key).aggr(
            VisitEnd & place_key, last_visit='MAX(visit_end)').fetch('last_visit')
        start = pd.Timestamp(min(subjects_last_visits)) if len(subjects_last_visits) else None
        end = pd.Timestamp.now() if start else None

        if experiment_name in ('exp0.1-r0', 'social0-r1'):
            subject_data = aeon_api.load(
                raw_data_dir.as_posix(),
                aeon_api.subjectreader,
                device='SessionData',
                prefix='SessionData_2',
                extension="*.csv",
                start=start,
                end=end
            )
            subject_data.replace('Start', 'Enter', inplace=True)
            subject_data.replace('End', 'Exit', inplace=True)
        else:
            subject_data = aeon_api.subjectdata(
                raw_data_dir.as_posix(),
                start=start,
                end=end
            )

        subject_visits = aeon_api.subjectvisit(subject_data)

        subject_list = (acquisition.Experiment.Subject & exp_key).fetch("subject")

        for _, r in subject_visits.iterrows():
            if r.id in subject_list:
                with Visit.connection.transaction:
                    Visit.insert1({
                        **exp_key, **place_key,
                        "subject": r.id,
                        "visit_start": r.enter},
                        skip_duplicates=True)
                    VisitEnd.insert1({
                        **exp_key, **place_key,
                        "subject": r.id,
                        "visit_start": r.enter,
                        "visit_end": r.exit,
                        "visit_duration": r.duration.total_seconds() / 3600},
                        skip_duplicates=True)
