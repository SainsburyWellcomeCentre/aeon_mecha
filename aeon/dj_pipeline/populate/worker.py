"""This module defines the workers for the AEON pipeline."""

import datajoint as dj
from datajoint_utilities.dj_worker import DataJointWorker, ErrorLog, WorkerLog
from datajoint_utilities.dj_worker.worker_schema import is_djtable

from aeon.dj_pipeline import acquisition, db_prefix, qc, subject, tracking
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
    max_idled_cycle=1,
    sleep_duration=5,
)
acquisition_worker(ingest_epochs_chunks)
acquisition_worker(acquisition.EpochConfig)
acquisition_worker(acquisition.Environment)
#acquisition_worker(block_analysis.BlockDetection)

# configure a worker to handle pyrat sync
pyrat_worker = DataJointWorker(
    "pyrat_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    max_idled_cycle=1,
    sleep_duration=5,
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
    max_idled_cycle=1,
    sleep_duration=5,
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
    max_idled_cycle=1,
    sleep_duration=5,
)

analysis_worker(block_analysis.BlockAnalysis, max_calls=6)
analysis_worker(block_analysis.BlockSubjectAnalysis, max_calls=6)
analysis_worker(block_analysis.BlockForaging, max_calls=6)
analysis_worker(block_analysis.BlockPatchPlots, max_calls=6)
analysis_worker(block_analysis.BlockSubjectPositionPlots, max_calls=6)


def get_workflow_operation_overview():
    """Get the workflow operation overview for the worker schema."""
    from datajoint_utilities.dj_worker.utils import get_workflow_operation_overview

    return get_workflow_operation_overview(worker_schema_name=worker_schema_name, db_prefixes=[db_prefix])


def retrieve_schemas_sizes(schema_only=False, all_schemas=False):
    schema_names = [n for n in dj.list_schemas() if n != "mysql"]
    if not all_schemas:
        schema_names = [n for n in schema_names
                        if n.startswith(db_prefix) and not n.startswith(f"{db_prefix}archived")]

    if schema_only:
        return {n: dj.Schema(n).size_on_disk / 1e9 for n in schema_names}

    schema_sizes = {n: {} for n in schema_names}
    for n in schema_names:
        vm = dj.VirtualModule(n, n)
        schema_sizes[n]["schema_gb"] = vm.schema.size_on_disk / 1e9
        schema_sizes[n]["tables_gb"] = {n: t().size_on_disk / 1e9
                                        for n, t in vm.__dict__.items()
                                        if isinstance(t, dj.user_tables.TableMeta)}
    return schema_sizes
