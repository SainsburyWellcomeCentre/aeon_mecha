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

    python ./aeon/dj_pipeline/populate/process.py --help


Usage from python:

    `from aeon.dj_pipeline.populate.process import run; run(worker_name='high_priority', duration=20, sleep=5)`

"""

import sys
import datajoint as dj
from datajoint_utilities.dj_worker import parse_args

from aeon.dj_pipeline.populate.worker import (
    acquisition_worker,
    mid_priority,
    streams_worker,
    logger,
)


# ---- some wrappers to support execution as script or CLI

configured_workers = {
    "high_priority": acquisition_worker,
    "mid_priority": mid_priority,
    "streams_worker": streams_worker,
}


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
    loglevel = kwargs.get("loglevel") or dj.config.get("loglevel", "INFO")

    logger.setLevel(loglevel)
    logger.debug("Starting ingestion process.")
    logger.info(f"worker_name={kwargs['worker_name']}")

    worker = configured_workers[kwargs["worker_name"]]
    if kwargs.get("duration") is not None:
        worker._run_duration = kwargs["duration"]
    if kwargs.get("sleep") is not None:
        worker._sleep_duration = kwargs["sleep"]

    try:
        worker.run()
    except Exception:
        logger.exception(
            "action '{}' encountered an exception:".format(kwargs["worker_name"])
        )

    logger.info("Ingestion process ended.")


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
