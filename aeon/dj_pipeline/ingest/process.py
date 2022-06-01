"""Start an Aeon ingestion process

This script defines auto-processing routines to operate the DataJoint pipeline for the
Aeon project. Three separate "process" functions are defined to call `populate()` for
different groups of tables, depending on their priority in the ingestion routines (high,
mid, low).

Each process function is run in a while-loop with the total run-duration configurable
via command line argument '--duration' (if not set, runs perpetually)

    - the loop will not begin a new cycle after this period of time (in seconds)
    - the loop will run perpetually if duration<0 or if duration==None
    - the script will not be killed _at_ this limit, it will keep executing,
        and just stop repeating after the time limit is exceeded

Usage as a console entrypoint:

        aeon_ingest high_priority
        aeon_ingest mid_priority -d 600 -s 60
        aeon_ingest --help


Usage as a script:

    python ./aeon/dj_pipeline/ingest/process.py --help


Usage from python:

    `from aeon.dj_pipeline.ingest.process import run; run(worker_name='high_priority', duration=20, sleep=5)`

"""

import logging
import sys

import datajoint as dj
from datajoint_utilities.dj_worker import DataJointWorker, WorkerLog, parse_args  # noqa

from aeon.dj_pipeline import acquisition, analysis, db_prefix, qc, report, tracking

# ---- Some constants ----

_logger = logging.getLogger(__name__)
_current_experiment = "exp0.2-r0"
worker_schema_name = db_prefix + "workerlog"

# ---- Define worker(s) ----
# configure a worker to process high-priority tasks
high_priority = DataJointWorker(
    "high_priority",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=-1,
    sleep_duration=600,
)

high_priority(acquisition.Epoch.ingest_epochs, experiment_name=_current_experiment)
high_priority(acquisition.Chunk.ingest_chunks, experiment_name=_current_experiment)
high_priority(acquisition.ExperimentLog)
high_priority(acquisition.SubjectEnterExit)
high_priority(acquisition.SubjectWeight)
high_priority(acquisition.FoodPatchEvent)
high_priority(acquisition.WheelState)
high_priority(acquisition.WeightMeasurement)

high_priority(analysis.InArena)
high_priority(analysis.InArenaEnd)
high_priority(analysis.InArenaTimeSlice)

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
mid_priority(analysis.InArenaSubjectPosition)
mid_priority(analysis.InArenaTimeDistribution)
mid_priority(analysis.InArenaSummary)
mid_priority(analysis.InArenaRewardRate)
# report tables
mid_priority(report.delete_outdated_plot_entries)
mid_priority(report.SubjectRewardRateDifference)
mid_priority(report.SubjectWheelTravelledDistance)
mid_priority(report.ExperimentTimeDistribution)
mid_priority(report.InArenaSummaryPlot)

# ---- some wrappers to support execution as script or CLI

configured_workers = {"high_priority": high_priority, "mid_priority": mid_priority}


def setup_logging(loglevel):
    """
    Setup basic logging

    :param loglevel: minimum loglevel for emitting messages
    :type loglevel: int
    """

    if loglevel is None:
        loglevel = logging.getLevelName(dj.config.get("loglevel", "INFO"))

    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format="%(asctime)s %(process)d %(processName)s "
        "%(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(**kwargs):
    """
    Run ingestion routine depending on the configured worker

    :param worker_name: Select the worker
    :type worker_name: str
    :param duration: Run duration of the process
    :type duration: int, optional
    :param sleep: Sleep time between subsequent runs
    :type sleep: int, optional
    :param loglevel: Set the logging output level
    :type loglevel: str, optional
    """

    setup_logging(kwargs.get("loglevel"))
    _logger.debug("Starting ingestion process.")
    _logger.info(f"worker_name={kwargs['worker_name']}")

    worker = configured_workers[kwargs["worker_name"]]
    if kwargs.get("duration") is not None:
        worker._run_duration = kwargs["duration"]
    if kwargs.get("sleep") is not None:
        worker._sleep_duration = kwargs["sleep"]

    try:
        worker.run()
    except Exception:
        _logger.exception(
            "action '{}' encountered an exception:".format(kwargs["worker_name"])
        )

    _logger.info("Ingestion process ended.")


def cli():
    """
    Calls :func:`run` passing the CLI arguments extracted from `sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    args = parse_args(sys.argv[1:])
    run(
        worker_name=args.worker_name,
        duration=args.duration,
        sleep=args.sleep,
        loglevel=args.loglevel,
    )


if __name__ == "__main__":
    cli()
