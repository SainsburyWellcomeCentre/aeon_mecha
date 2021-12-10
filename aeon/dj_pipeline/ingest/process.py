"""Start an Aeon ingestion process

This script defines auto-processing routines to operate the DataJoint pipeline for the Aeon
project. Three separate "process" functions are defined to call `populate()` for different
groups of tables, depending on their priority in the ingestion routines (high, mid, low).

Each process function is run in a while-loop with the total run-duration configurable via
command line argument '--duration' (if not set, runs perpetually)

    - the loop will not begin a new cycle after this period of time (in seconds)
    - the loop will run perpetually if duration<0 or if duration==None
    - the script will not be killed _at_ this limit, it will keep executing,
      and just stop repeating after the time limit is exceeded

Some populate settings (e.g. 'limit', 'max_calls') can be set to process some number of jobs at
a time for every iteration of the loop, instead of all jobs. This allows the processing to
propagate through the pipeline more horizontally

Usage as a script:

    aeon_ingest high
    aeon_ingest low -d 30 -s 1
    aeon_ingest mid -d 600

See script help messages:

    aeon_ingest --help

(or the function `aeon.dj_pipeline.ingest.process:process` can be imported directly)
"""

import argparse
import logging
import sys

from pythonjsonlogger import jsonlogger

from aeon.dj_pipeline import db_prefix
from aeon.dj_pipeline.ingest.dj_worker import DataJointWorker, WorkerLog

from aeon.dj_pipeline import acquisition, qc, tracking, analysis, report


# ---- Some constants ----

_logger = logging.getLogger(__name__)
_current_experiment = "exp0.1-r0"

# ---- Define worker(s) ----

worker_schema_name = db_prefix + 'workerlog'

# configure a worker to process high-priority tasks

high_priority = DataJointWorker('high_priority',
                                worker_schema_name=worker_schema_name,
                                db_prefix=db_prefix,
                                run_duration=-1,
                                sleep_duration=600)

high_priority(acquisition.Chunk.generate_chunks, experiment_name=_current_experiment)
high_priority(acquisition.SubjectEnterExit)
high_priority(acquisition.SubjectAnnotation)
high_priority(acquisition.SubjectWeight)
high_priority(acquisition.WheelState)
high_priority(acquisition.Session)
high_priority(acquisition.SessionEnd)
high_priority(acquisition.TimeSlice)

# configure a worker to process mid-priority tasks

mid_priority = DataJointWorker('mid_priority',
                               worker_schema_name=worker_schema_name,
                               db_prefix=db_prefix,
                               run_duration=-1,
                               sleep_duration=120)

mid_priority(qc.CameraQC)
mid_priority(tracking.SubjectPosition)
mid_priority(analysis.SessionTimeDistribution)
mid_priority(analysis.SessionSummary)
mid_priority(analysis.SessionRewardRate)
mid_priority(report.SubjectRewardRateDifference.delete_outdated_entries)
mid_priority(report.SubjectRewardRateDifference)
mid_priority(report.SubjectWheelTravelledDistance.delete_outdated_entries)
mid_priority(report.SubjectWheelTravelledDistance)
# mid_priority(report.SessionSummaryPlot)


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
        default=_ingestion_settings["priority"],
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
        "--version",
        action="version",
        version=f"aeon {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )

    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
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

    logHandler = logging.StreamHandler()

    logformat = "%(asctime)s %(process)d %(processName)s %(levelname)s %(name)s %(message)s"
    formatter = jsonlogger.JsonFormatter(logformat)
    logHandler.setFormatter(formatter)

    _logger.addHandler(logHandler)
    _logger.setLevel(loglevel)


def main(args):
    """
    Wrapper allowing process functions to be called from CLI

    :param args: command line parameters as list of strings (for example  ``["--help"]``)
    :type args: List[str]
    """

    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting ingestion process.")
    _logger.info(f"priority={args.priority}")

    priority_worker_mapper = {'high': high_priority, 'mid': mid_priority}

    worker = priority_worker_mapper[args.priority]
    worker._run_duration = args.duration
    worker._sleep_duration = args.sleep

    try:
        worker.run()
    except Exception:
        _logger.exception("action '{}' encountered an exception:".format(args.priority))

    _logger.info("Ingestion process ended.")


def run():
    """
    Calls :func:`main` passing the CLI arguments extracted from `sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """

    main(sys.argv[1:])


if __name__ == "__main__":
    run()
