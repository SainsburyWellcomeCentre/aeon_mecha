"""Module for visit-related tables in the analysis schema."""

import datetime
import datajoint as dj
import pandas as pd
import numpy as np
from collections import deque

from aeon.analysis import utils as analysis_utils

from aeon.dj_pipeline import get_schema_name, fetch_stream
from aeon.dj_pipeline import acquisition

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
        """Key source for OverlapVisit."""
        return dj.U("experiment_name", "place", "overlap_start") & (
            Visit & VisitEnd
        ).proj(overlap_start="visit_start")

    def make(self, key):
        """Populate OverlapVisit table with overlapping visits."""
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
                    "subject_count": len({v["subject"] for v in overlap_visits}),
                }
            )
            self.Visit.insert(overlap_visits, skip_duplicates=True)


# ---- HELPERS ----


def ingest_environment_visits(experiment_names: list | None = None):
    """Function to populate into `Visit` and `VisitEnd` for specified
    experiments (default: 'exp0.2-r0'). This ingestion routine handles
    only those "complete" visits, not ingesting any "on-going" visits
    using "analyze" method: `aeon.analyze.utils.visits()`.

    Args:
        experiment_names (list, optional): list of names of the experiment
        to populate into the Visit table. Defaults to None.
    """

    if experiment_names is None:
        experiment_names = ["exp0.2-r0"]
    place_key = {"place": "environment"}
    for experiment_name in experiment_names:
        exp_key = {"experiment_name": experiment_name}

        subjects_last_visits = (
            (acquisition.Experiment.Subject & exp_key)
            .aggr(VisitEnd & place_key, last_visit="MAX(visit_end)")
            .fetch("last_visit")
        )
        start = min(subjects_last_visits) if len(subjects_last_visits) else "1900-01-01"
        end = datetime.datetime.now() if start else "2200-01-01"

        enter_exit_query = (
            acquisition.SubjectEnterExit.Time * acquisition.EventType
            & {"experiment_name": experiment_name}
            & "event_type in ('SubjectEnteredArena', 'SubjectExitedArena')"
            & f'enter_exit_time BETWEEN "{start}" AND "{end}"'
        )

        if not enter_exit_query:
            continue

        enter_exit_df = pd.DataFrame(
            zip(
                *enter_exit_query.fetch(
                    "subject",
                    "enter_exit_time",
                    "event_type",
                    order_by="enter_exit_time",
                ), strict=False
            )
        )
        enter_exit_df.columns = ["id", "time", "event"]
        enter_exit_df.set_index("time", inplace=True)

        subject_visits = analysis_utils.visits(
            enter_exit_df, onset="SubjectEnteredArena", offset="SubjectExitedArena"
        )

        for _, r in subject_visits.iterrows():
            if pd.isna(r.subjectexitedarena):
                continue
            with Visit.connection.transaction:
                Visit.insert1(
                    {
                        **exp_key,
                        **place_key,
                        "subject": r.id,
                        "visit_start": r.subjectenteredarena,
                    },
                    skip_duplicates=True,
                )
                VisitEnd.insert1(
                    {
                        **exp_key,
                        **place_key,
                        "subject": r.id,
                        "visit_start": r.subjectenteredarena,
                        "visit_end": r.subjectexitedarena,
                        "visit_duration": r.duration.total_seconds() / 3600,
                    },
                    skip_duplicates=True,
                )


def get_maintenance_periods(experiment_name, start, end):
    """Get maintenance periods for the specified experiment and time range."""
    # get states from acquisition.Environment.EnvironmentState
    chunk_restriction = acquisition.create_chunk_restriction(
        experiment_name, start, end
    )
    state_query = (
        acquisition.Environment.EnvironmentState
        & {"experiment_name": experiment_name}
        & chunk_restriction
    )
    env_state_df = fetch_stream(state_query)[start:end]
    if env_state_df.empty:
        return deque([])

    env_state_df.reset_index(inplace=True)
    env_state_df = env_state_df[
        env_state_df["state"].shift() != env_state_df["state"]
    ].reset_index(
        drop=True
    )  # remove duplicates and keep the first one
    # An experiment starts with visit start (anything before the first maintenance is experiment)
    # Delete the row if it starts with "Experiment"
    if env_state_df.iloc[0]["state"] == "Experiment":
        env_state_df.drop(index=0, inplace=True)  # look for the first maintenance
        if env_state_df.empty:
            return deque([])

    # Last entry is the visit end
    if env_state_df.iloc[-1]["state"] == "Maintenance":
        log_df_end = pd.DataFrame({"time": [pd.Timestamp(end)], "state": ["VisitEnd"]})
        env_state_df = pd.concat([env_state_df, log_df_end])
        env_state_df.reset_index(drop=True, inplace=True)

    maintenance_starts = env_state_df.loc[
        env_state_df["state"] == "Maintenance", "time"
    ].values
    maintenance_ends = env_state_df.loc[
        env_state_df["state"] != "Maintenance", "time"
    ].values

    return deque(
        [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in zip(maintenance_starts, maintenance_ends, strict=False)
        ]
    )  # queue object. pop out from left after use


def filter_out_maintenance_periods(data_df, maintenance_period, end_time, dropna=False):
    """Filter out maintenance periods from the data_df."""
    maint_period = maintenance_period.copy()
    while maint_period:
        (maintenance_start, maintenance_end) = maint_period[0]
        if end_time < maintenance_start:  # no more maintenance for this date
            break
        maintenance_filter = (data_df.index >= maintenance_start) & (
            data_df.index <= maintenance_end
        )
        data_df[maintenance_filter] = np.nan
        if end_time >= maintenance_end:  # remove this range
            maint_period.popleft()
        else:
            break
    if dropna:
        data_df.dropna(inplace=True)
    return data_df
