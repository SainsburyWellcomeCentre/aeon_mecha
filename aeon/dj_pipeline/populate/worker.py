"""
This module defines the workers for the AEON pipeline.
"""

import datajoint as dj
from datajoint_utilities.dj_worker import (
    DataJointWorker,
    ErrorLog,
    WorkerLog,
    RegisteredWorker,
)
from datajoint_utilities.dj_worker.worker_schema import is_djtable

from aeon.dj_pipeline import db_prefix
from aeon.dj_pipeline import subject, acquisition, tracking, qc
from aeon.dj_pipeline.analysis import block_analysis
from aeon.dj_pipeline.utils import streams_maker

streams = streams_maker.main()

__all__ = [
    "acquisition_worker",
    "analysis_worker",
    "pyrat_worker",
    "streams_worker",
    "WorkerLog",
    "ErrorLog",
    "logger",
    "AutomatedExperimentIngestion",
]

# ---- Some constants ----
logger = dj.logger
worker_schema_name = db_prefix + "worker"

# ---- Manage experiments for automated ingestion ----

schema = dj.Schema(worker_schema_name)


@schema
class AutomatedExperimentIngestion(dj.Manual):
    definition = """  # experiments to undergo automated ingestion
    -> acquisition.Experiment
    """


def ingest_epochs_chunks():
    """Ingest epochs and chunks for experiments specified in AutomatedExperimentIngestion."""
    experiment_names = AutomatedExperimentIngestion.fetch("experiment_name")
    for experiment_name in experiment_names:
        acquisition.Epoch.ingest_epochs(experiment_name)
        acquisition.Chunk.ingest_chunks(experiment_name)


# ---- Define worker(s) ----
# configure a worker to process `acquisition`-related tasks
acquisition_worker = DataJointWorker(
    "acquisition_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    max_idled_cycle=6,
    sleep_duration=1200,
)
acquisition_worker(ingest_epochs_chunks)
acquisition_worker(acquisition.EpochConfig)
acquisition_worker(acquisition.Environment)
acquisition_worker(block_analysis.BlockDetection)

# configure a worker to handle pyrat sync
pyrat_worker = DataJointWorker(
    "pyrat_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    max_idled_cycle=400,
    sleep_duration=30,
)

pyrat_worker(subject.CreatePyratIngestionTask)
pyrat_worker(subject.PyratIngestion)
pyrat_worker(subject.SubjectDetail)
pyrat_worker(subject.PyratCommentWeightProcedure)

# configure a worker to ingest all data streams
streams_worker = DataJointWorker(
    "streams_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    max_idled_cycle=50,
    sleep_duration=60,
    autoclear_error_patterns=["%BlockAnalysis Not Ready%"],
)

for attr in vars(streams).values():
    if is_djtable(attr, dj.user_tables.AutoPopulate):
        streams_worker(attr, max_calls=10)

streams_worker(qc.CameraQC, max_calls=10)
streams_worker(tracking.SLEAPTracking, max_calls=10)

# configure a worker to run the analysis tables
analysis_worker = DataJointWorker(
    "analysis_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    max_idled_cycle=20,
    sleep_duration=60,
)

analysis_worker(block_analysis.BlockAnalysis, max_calls=6)
analysis_worker(block_analysis.BlockPlots, max_calls=6)
analysis_worker(block_analysis.BlockSubjectAnalysis, max_calls=6)
analysis_worker(block_analysis.BlockSubjectPlots, max_calls=6)


def get_workflow_operation_overview():
    """Get the workflow operation overview for the worker schema."""
    from datajoint_utilities.dj_worker.utils import get_workflow_operation_overview

    return get_workflow_operation_overview(
        worker_schema_name=worker_schema_name, db_prefixes=[db_prefix]
    )
