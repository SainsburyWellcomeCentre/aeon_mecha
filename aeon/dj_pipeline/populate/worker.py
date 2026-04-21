"""This module defines the workers for the AEON pipeline."""

import datajoint as dj
from datajoint_utilities.dj_worker import DataJointWorker, ErrorLog, WorkerLog
from datajoint_utilities.dj_worker.worker_schema import is_djtable

from aeon.dj_pipeline import acquisition, db_prefix, qc, subject, tracking
from aeon.dj_pipeline.utils import streams_maker
from aeon.dj_pipeline.utils.load_metadata import (
    get_experiment_pydantic,
    populate_catalog_from_pydantic,
)

# STEP 1: Populate catalog from all registered Experiment classes
# This must happen BEFORE streams_maker.main() creates tables
for exp in acquisition.Experiment.DevicesSchema.fetch(as_dict=True):
    devices_schema_name = exp["devices_schema_name"]
    try:
        experiment_class = get_experiment_pydantic(devices_schema_name)
        populate_catalog_from_pydantic(experiment_class)
    except (ImportError, ModuleNotFoundError) as e:
        dj.logger.error(
            f"Failed to import Experiment class '{devices_schema_name}': {e}. "
            "Check if the package is installed."
        )
        raise  # Re-raise to fail fast on missing dependencies
    except Exception as e:
        dj.logger.warning(f"Could not populate catalog for {devices_schema_name}: {e}")

# STEP 2: Create tables (MUST be outside transaction)
streams = streams_maker.main()

__all__ = [
    "acquisition_worker",
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
)

for attr in vars(streams).values():
    if is_djtable(attr, dj.user_tables.AutoPopulate):
        streams_worker(attr, max_calls=10)

streams_worker(qc.CameraQC, max_calls=10)
streams_worker(tracking.SLEAPTracking, max_calls=10)


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
