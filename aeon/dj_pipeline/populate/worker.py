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
    "high_priority",
    "mid_priority",
    "streams_worker",
    "WorkerLog",
    "ErrorLog",
    "logger",
]

# ---- Some constants ----
logger = dj.logger
_current_experiment = "exp0.2-r0"
worker_schema_name = db_prefix + "workerlog"
load_metadata.insert_stream_types()


# ---- Define worker(s) ----
# configure a worker to process high-priority tasks
high_priority = DataJointWorker(
    "high_priority",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=-1,
    sleep_duration=600,
)
high_priority(load_metadata.ingest_subject)
high_priority(acquisition.Epoch.ingest_epochs, experiment_name=_current_experiment)
high_priority(acquisition.Chunk.ingest_chunks, experiment_name=_current_experiment)
high_priority(acquisition.ExperimentLog)
high_priority(acquisition.SubjectEnterExit)
high_priority(acquisition.SubjectWeight)
high_priority(acquisition.FoodPatchEvent)
high_priority(acquisition.WheelState)
high_priority(acquisition.WeightMeasurement)
high_priority(acquisition.WeightMeasurementFiltered)

high_priority(
    analysis.ingest_environment_visits, experiment_names=[_current_experiment]
)

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

mid_priority(
    analysis.visit.ingest_environment_visits, experiment_names=[_current_experiment]
)
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
