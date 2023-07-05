import datajoint as dj
from datajoint_utilities.dj_worker import (
    DataJointWorker,
    ErrorLog,
    WorkerLog,
    is_djtable,
)

from aeon.dj_pipeline import (
    acquisition,
    analysis,
    db_prefix,
    qc,
    report,
    streams_maker,
    tracking,
)
from aeon.dj_pipeline.utils import load_metadata

streams = streams_maker.main()

__all__ = [
    "acquisition_worker",
    "mid_priority",
    "streams_worker",
    "WorkerLog",
    "ErrorLog",
    "logger",
]

# ---- Some constants ----
logger = dj.logger
worker_schema_name = db_prefix + "worker"
load_metadata.insert_stream_types()


# ---- Manage experiments for automated ingestion ----

schema = dj.Schema(worker_schema_name)


@schema
class AutomatedExperimentIngestion(dj.Manual):
    definition = """  # experiments to undergo automated ingestion
    -> acquisition.Experiment
    """


def ingest_colony_epochs_chunks():
    """
    Load and insert subjects from colony.csv
    Ingest epochs and chunks
     for experiments specified in AutomatedExperimentIngestion
    """
    load_metadata.ingest_subject()
    experiment_names = AutomatedExperimentIngestion.fetch("experiment_name")
    for experiment_name in experiment_names:
        acquisition.Epoch.ingest_epochs(experiment_name)
        acquisition.Chunk.ingest_chunks(experiment_name)


def ingest_environment_visits():
    """
    Extract and insert complete visits
     for experiments specified in AutomatedExperimentIngestion
    """
    experiment_names = AutomatedExperimentIngestion.fetch("experiment_name")
    analysis.ingest_environment_visits(experiment_names)


# ---- Define worker(s) ----
# configure a worker to process `acquisition`-related tasks
acquisition_worker = DataJointWorker(
    "acquisition_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=-1,
    sleep_duration=600,
)
acquisition_worker(ingest_colony_epochs_chunks)
acquisition_worker(acquisition.ExperimentLog)
acquisition_worker(acquisition.SubjectEnterExit)
acquisition_worker(acquisition.SubjectWeight)
acquisition_worker(acquisition.FoodPatchEvent)
acquisition_worker(acquisition.WheelState)

acquisition_worker(ingest_environment_visits)

# configure a worker to process mid-priority tasks
mid_priority = DataJointWorker(
    "mid_priority",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=-1,
    sleep_duration=120,
)

mid_priority(qc.CameraQC)
mid_priority(tracking.CameraTracking)
mid_priority(acquisition.FoodPatchWheel)
mid_priority(acquisition.WeightMeasurement)
mid_priority(acquisition.WeightMeasurementFiltered)

mid_priority(analysis.OverlapVisit)

mid_priority(analysis.VisitSubjectPosition)
mid_priority(analysis.VisitTimeDistribution)
mid_priority(analysis.VisitSummary)
mid_priority(analysis.VisitForagingBout)

# report tables
mid_priority(report.delete_outdated_plot_entries)
mid_priority(report.SubjectRewardRateDifference)
mid_priority(report.SubjectWheelTravelledDistance)
mid_priority(report.ExperimentTimeDistribution)
mid_priority(report.VisitDailySummaryPlot)

# ---- Define worker(s) ----
# configure a worker to ingest all data streams
streams_worker = DataJointWorker(
    "streams_worker",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=1,
    sleep_duration=600,
)

for attr in vars(streams).values():
    if is_djtable(attr, dj.user_tables.AutoPopulate):
        streams_worker(attr)
