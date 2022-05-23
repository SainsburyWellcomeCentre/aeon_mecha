import datajoint as dj
import pandas as pd
import numpy as np
import datetime

from aeon.analyze import utils as analyze_utils

from .. import lab, acquisition, tracking, qc
from .. import get_schema_name

schema = dj.schema(get_schema_name("analysis"))


@schema
class Place(dj.Lookup):
    definition = """
    place: varchar(32)  # User-friendly name of a 'place' animals can visit - e.g. nest, arena, foodpatch
    ---
    place_description: varchar(1000)
    """

    contents = [
        (
            "environment",
            "the entire environment where the animals undergo experiment (previously known as arena)",
        )
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


@schema
class OverlapVisit(dj.Computed):
    definition = """
    -> acquisition.Experiment
    -> Place
    overlap_start: datetime(6)
    ---
    overlap_end: datetime(6)
    overlap_duration: float  # (hour)
    subject_count: int
    """

    class Visit(dj.Part):
        definition = """
        -> master
        -> Visit
        """

    @property
    def key_source(self):
        return dj.U("experiment_name", "place", "overlap_start") & (
            Visit & VisitEnd
        ).proj(overlap_start="visit_start")

    def make(self, key):
        visit_starts, visit_ends = (
            Visit * VisitEnd & key & {"visit_start": key["overlap_start"]}
        ).fetch("visit_start", "visit_end")
        visit_start = min(visit_starts)
        visit_end = max(visit_ends)

        overlap_visits = []
        for _ in range(100):
            overlap_query = (
                Visit * VisitEnd
                & {"experiment_name": key["experiment_name"], "place": key["place"]}
                & f'visit_start BETWEEN "{visit_start}" AND "{visit_end}"'
            )
            if len(overlap_query) <= 1:
                break
            overlap_visits.extend(
                overlap_query.proj(overlap_start=f'"{key["overlap_start"]}"').fetch(
                    as_dict=True
                )
            )
            visit_starts, visit_ends = overlap_query.fetch("visit_start", "visit_end")
            if visit_start == max(visit_starts) and visit_end == max(visit_ends):
                break
            else:
                visit_start = max(visit_starts)
                visit_end = max(visit_ends)

        if overlap_visits:
            self.insert1(
                {
                    **key,
                    "overlap_end": visit_end,
                    "overlap_duration": (
                        visit_end - key["overlap_start"]
                    ).total_seconds()
                    / 3600,
                    "subject_count": len(set(v["subject"] for v in overlap_visits)),
                }
            )
            self.Visit.insert(overlap_visits, skip_duplicates=True)


# ---- HELPERS ----


def get_subject_environment_visits(experiment_name, subject, start=None, end=None):
    """
    Function to retrieve the enter/exit times and compute the "visits" to the "environment"
        for a given subject of an experiment
    Using "analyze" method: `aeon.analyze.utils.visits()`

    :param str experiment_name: name of the experiment
    :param str subject: name of the subject to retrieve the visits
    :param datetime start: start time to search for the visits
    :param datetime end: end time to search for the visits
    :return: The dataframe of the subject visits
    """
    start = start or "1900-01-01"
    end = end or "2200-01-01"

    enter_exit_query = (
        acquisition.SubjectEnterExit.Time * acquisition.EventType
        & {"experiment_name": experiment_name, "subject": subject}
        & "event_type in ('SubjectEnteredArena', 'SubjectExitedArena')"
        & f'enter_exit_time BETWEEN "{start}" AND "{end}"'
    )

    if not enter_exit_query:
        return pd.DataFrame()

    enter_exit_df = pd.DataFrame(
        zip(
            *enter_exit_query.fetch(
                "subject", "enter_exit_time", "event_type", order_by="enter_exit_time"
            )
        )
    )
    enter_exit_df.columns = ["id", "time", "event"]
    enter_exit_df.set_index("time", inplace=True)

    enter_exit_df.replace("SubjectEnteredArena", "Enter", inplace=True)
    enter_exit_df.replace("SubjectExitedArena", "Exit", inplace=True)

    subject_visits = analyze_utils.visits(enter_exit_df)

    return subject_visits


def ingest_environment_visits(experiment_names=["exp0.2-r0"]):
    """
    Function to ingest into `Visit` and `VisitEnd` for specified experiments (default: 'exp0.2-r0')
    This ingestion routine handles only those "complete" visits, not ingesting any "on-going" visits

    :param list experiment_names: list of names of the experiment to ingest into the Visit table
    """
    place_key = {"place": "environment"}
    for experiment_name in experiment_names:
        exp_key = {"experiment_name": experiment_name}

        subjects_last_visits = (
            (acquisition.Experiment.Subject & exp_key)
            .aggr(VisitEnd & place_key, last_visit="MAX(visit_end)")
            .fetch("last_visit")
        )
        start = min(subjects_last_visits) if len(subjects_last_visits) else None
        end = datetime.datetime.now() if start else None

        for subject in (acquisition.Experiment.Subject & exp_key).fetch("subject"):
            subject_visits = get_subject_environment_visits(
                experiment_name, subject, start=start, end=end
            )
            for _, r in subject_visits.iterrows():
                with Visit.connection.transaction:
                    Visit.insert1(
                        {
                            **exp_key,
                            **place_key,
                            "subject": r.id,
                            "visit_start": r.enter,
                        },
                        skip_duplicates=True,
                    )
                    VisitEnd.insert1(
                        {
                            **exp_key,
                            **place_key,
                            "subject": r.id,
                            "visit_start": r.enter,
                            "visit_end": r.exit,
                            "visit_duration": r.duration.total_seconds() / 3600,
                        },
                        skip_duplicates=True,
                    )
