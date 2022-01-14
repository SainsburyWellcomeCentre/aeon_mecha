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

        aeon_ingest high
        aeon_ingest mid -d 600 -s 60
        aeon_ingest --help


Usage as a script:

    python ./aeon/dj_pipeline/ingest/process.py --help


Usage from python:

    `from aeon.dj_pipeline.ingest.process import run; run(priority='high')`

"""

import argparse
import logging
import sys

from aeon.dj_pipeline import acquisition, analysis, db_prefix, qc, report, tracking
from aeon.dj_pipeline.ingest.dj_worker import DataJointWorker, WorkerLog

# ---- Some constants ----

_logger = logging.getLogger(__name__)
_current_experiment = "exp0.1-r0"
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

high_priority(acquisition.Chunk.generate_chunks, experiment_name=_current_experiment)
high_priority(acquisition.SubjectEnterExit)
high_priority(acquisition.SubjectAnnotation)
high_priority(acquisition.SubjectWeight)
high_priority(acquisition.WheelState)
high_priority(acquisition.Session)
high_priority(acquisition.SessionEnd)
high_priority(acquisition.TimeSlice)

# configure a worker to process mid-priority tasks
mid_priority = DataJointWorker(
    "mid_priority",
    worker_schema_name=worker_schema_name,
    db_prefix=db_prefix,
    run_duration=-1,
    sleep_duration=120,
)

mid_priority(qc.CameraQC)
mid_priority(tracking.SubjectPosition)
mid_priority(analysis.SessionTimeDistribution)
mid_priority(analysis.SessionSummary)
mid_priority(analysis.SessionRewardRate)
# report tables
mid_priority(report.delete_outdated_plot_entries)
mid_priority(report.SubjectRewardRateDifference)
mid_priority(report.SubjectWheelTravelledDistance)
mid_priority(report.ExperimentTimeDistribution)
mid_priority(report.SessionSummaryPlot)

# ---- some wrappers to support execution as script or CLI

_ingestion_settings = {"priority": "high", "duration": -1, "sleep": 60}


# combine different formatters
class ArgumentDefaultsRawDescriptionHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def parse_args(args):
    """
    Parse command line parameters

    :param args: command line parameters as list of strings (for example  ``["--help"]``)
    :type args: List[str]
    :return: `argparse.Namespace`: command line parameters namespace
    :rtype: obj
    """

    from aeon import __version__

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsRawDescriptionHelpFormatter
    )

    parser.add_argument(
        "priority",
        help="Select the processing priority level",
        type=str,
        choices=["high", "mid"],
    )

    parser.add_argument(
        "-d",
        "--duration",
        dest="duration",
        help="Run duration of the entire process",
        type=int,
        metavar="INT",
        default=_ingestion_settings["duration"],
    )

    parser.add_argument(
        "-s",
        "--sleep",
        dest="sleep",
        help="Sleep time between subsequent runs",
        type=int,
        metavar="INT",
        default=_ingestion_settings["sleep"],
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="Set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="Set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"aeon {__version__}",
    )

    return parser.parse_args(args)


def setup_logging(loglevel):
    """
    Setup basic logging

    :param loglevel: minimum loglevel for emitting messages
    :type loglevel: int
    """

    if loglevel is None:
        loglevel = logging.INFO

    logging.basicConfig(
        level=loglevel,
        stream=sys.stdout,
        format="%(asctime)s %(process)d %(processName)s "
        "%(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(**kwargs):
    """
    Run ingestion routine depending on the priority value

    :param priority: Select the processing level
    :type priority: str
    :param duration: Run duration of the process
    :type duration: int, optional
    :param sleep: Sleep time between subsequent runs
    :type sleep: int, optional
    :param loglevel: Set the logging output level
    :type loglevel: str, optional
    """
    kwargs = {**_ingestion_settings, **kwargs}

    setup_logging(kwargs.get("loglevel"))
    _logger.debug("Starting ingestion process.")
    _logger.info(f"priority={kwargs['priority']}")

    priority_worker_mapper = {"high": high_priority, "mid": mid_priority}

    worker = priority_worker_mapper[kwargs["priority"]]
    worker._run_duration = kwargs["duration"]
    worker._sleep_duration = kwargs["sleep"]

    try:
        worker.run()
    except Exception:
        _logger.exception(
            "action '{}' encountered an exception:".format(kwargs["priority"])
        )

    _logger.info("Ingestion process ended.")


def cli():
    """
    Calls :func:`run` passing the CLI arguments extracted from `sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    args = parse_args(sys.argv[1:])
    run(
        priority=args.priority,
        duration=args.duration,
        sleep=args.sleep,
        loglevel=args.loglevel,
    )


if __name__ == "__main__":
    cli()
