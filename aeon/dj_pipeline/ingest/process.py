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
    aeon_ingest --help
    aeon_ingest low -d 30 -s 1 -m 5
    aeon_ingest djconfig --help

(or the `process` function can be used as a python module normally)
"""


import argparse
import logging
import sys
import time

import datajoint as dj

_logger = logging.getLogger(__name__)


_current_experiment = "exp0.1-r0"

# TODO: other datajoint populate settings here and in command line args
_datajoint_settings = {"reserve_jobs": True, "suppress_errors": True, "display_progress": True}

_ingestion_defaults = {"priority": "high", "duration": -1, "sleep": 5, "max_calls": -1}


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "priority",
        help="Select the processing priority level",
        type=str,
        default=_ingestion_defaults["priority"],
        choices=["high", "mid", "low"],
    )

    parser.add_argument(
        "-d",
        "--duration",
        dest="duration",
        help="Run duration of the entire process",
        type=int,
        default=_ingestion_defaults["duration"],
    )

    parser.add_argument(
        "-s",
        "--sleep",
        dest="sleep",
        help="Sleep time between subsequent runs",
        type=int,
        default=_ingestion_defaults["sleep"],
    )

    parser.add_argument(
        "-m",
        "--maxcalls",
        dest="max_calls",
        help="Max number of jobs to process within each loop iteration",
        type=int,
        default=_ingestion_defaults["max_calls"],
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

    subparsers = parser.add_subparsers()

    parser_djconf = subparsers.add_parser("djconfig", description="DataJoint config options")

    # TODO: Other possible dj config arguments to add here

    parser_djconf.add_argument(
        "-p", "--port", dest="port", type=int, help='entry for "database.port"'
    )

    parser_djconf.add_argument(
        "--dbprefix", dest="db_prefix", type=str, help='entry for "database.prefix"'
    )

    return parser.parse_args(args)


def dj_config_override(args):
    """
    Overwrite any configuration options in `dj.config` with those set in `args`, if any.

    :param args: `argparse.Namespace`: command line parameters namespace
    :type args: obj
    """

    if args.port is not None:
        dj.config["database.port"] = args.port

    if args.db_prefix is not None:
        if "custom" not in dj.config:
            dj.config["custom"] = {}
        dj.config["custom"]["database.prefix"] = args.db_prefix


def setup_logging(loglevel):
    """
    Setup basic logging

    :param loglevel: minimum loglevel for emitting messages
    :type loglevel: int
    """

    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def process(priority, **kwargs):
    """
    Run type of ingestion routine depending on priority value

    :param priority: Select the processing level
    :type priority: str
    :param run_duration: Run duration of the process
    :type run_duration: int
    :param sleep_duration: Sleep time between subsequent runs
    :type sleep_duration: int
    :param max_calls: Max number of jobs to process within each loop iteration
    :type max_calls: int
    """

    # importing here to connect to db after args are parsed
    from aeon.dj_pipeline import analysis, acquisition, tracking
    from aeon.dj_pipeline.ingest.monitor import ProcessJob

    _priority_tables = {
        "high": (
            acquisition.SubjectEnterExit,
            acquisition.SubjectAnnotation,
            acquisition.SubjectWeight,
            acquisition.FoodPatchEvent,
            acquisition.WheelState,
            acquisition.Session,
            acquisition.SessionEnd,
            acquisition.TimeSlice,
        ),
        "mid": (
            tracking.SubjectPosition,
            analysis.SessionTimeDistribution,
            analysis.SessionSummary,
        ),
        "low": (acquisition.FoodPatchWheel, tracking.SubjectDistance),
    }

    run_duration = kwargs.get("run_duration", _ingestion_defaults["duration"])

    sleep_duration = kwargs.get("sleep_duration", _ingestion_defaults["sleep"])

    max_calls = kwargs.get("max_calls", _ingestion_defaults["max_calls"])
    if max_calls is not None and max_calls < 0:
        max_calls = None

    start_time = time.time()

    while (
        (time.time() - start_time < run_duration)
        or (run_duration is None)
        or (run_duration < 0)
    ):

        if priority == "high":
            ProcessJob.log_process_job(acquisition.Chunk)
            acquisition.Chunk.generate_chunks(experiment_name=_current_experiment)

        for table_to_process in _priority_tables[priority]:
            ProcessJob.log_process_job(table_to_process)
            table_to_process.populate(**_datajoint_settings, max_calls=max_calls)

        time.sleep(sleep_duration)


def main(args):
    """
    Wrapper allowing process functions to be called from CLI

    :param args: command line parameters as list of strings (for example  ``["--help"]``)
    :type args: List[str]
    """

    args = parse_args(args)
    dj_config_override(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting ingestion process.")
    print(f"priority={args.priority}")

    try:
        process(
            args.priority,
            run_duration=args.duration,
            sleep_duration=args.sleep,
            max_calls=args.max_calls,
        )
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
