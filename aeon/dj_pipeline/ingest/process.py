import argparse
import logging
import sys
import time


_logger = logging.getLogger(__name__)


"""
Auto-processing routines defined to operate the DataJoint pipeline for the Aeon project
Several "process" functions defined to call the `populate()` for different grouping of tables,
depending on their priority in the ingestion routines.

Each process function is ran in a while-loop with the total run-duration
configurable via command line argument '--duration' (if not set, runs perpetually)
    - the loop will not begin a new cycle after this period of time (in second)
    - the loop will run perpetually if duration<0 or if duration==None
    - the script will not be killed _at_ this limit, it will keep executing,
        and just stop repeating after the time limit is exceeded

Some populate settings (e.g. 'limit', 'max_calls') can be set to process
some number of jobs at a time for every iteration of the loop, instead of all jobs.
This allows the processing to propagate through the pipeline more horizontally

TODO: other datajoint populate settings in command line args

Usage as a script:

    aeon_ingest high
    aeon_ingest --help

(or the `process` function can be used as python module normally)
"""


_current_experiment = "exp0.1-r0"

_datajoint_settings = {"reserve_jobs": True, "suppress_errors": True, "display_progress": True}

_ingestion_defaults = {"priority": "high", "duration": -1, "sleep": 5, "max_calls": None}


def parse_args(args):
    """
    Parse command line parameters

    :param args: command line parameters as list of strings (for example  ``["--help"]``)
    :type args: List[str]
    :return: `argparse.Namespace`: command line parameters namespace
    :rtype: obj
    """

    from aeon import __version__

    parser = argparse.ArgumentParser(description="Settings for running the ingestion routine.")

    parser.add_argument(
        "priority",
        help="Select the processing level",
        type=str,
        default=_ingestion_defaults["priority"],
        choices=["high", "mid", "low"],
    )

    parser.add_argument(
        "-d",
        "--duration",
        dest="duration",
        help="Run duration of the process",
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

    return parser.parse_args(args)


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


def process(priority, *, run_duration, sleep_duration, max_calls):
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
