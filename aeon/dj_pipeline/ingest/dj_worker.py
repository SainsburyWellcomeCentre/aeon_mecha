"""
Mechanism to set up and manage "workers" to operate a DataJoint pipeline

Each "worker" is run in a while-loop with the total run-duration configurable via
command line argument '--duration' (if not set, runs perpetually)

    - the loop will not begin a new cycle after this period of time (in seconds)
    - the loop will run perpetually if duration<0 or if duration==None
    - the script will not be killed _at_ this limit, it will keep executing,
      and just stop repeating after the time limit is exceeded

Some populate settings (e.g. 'limit', 'max_calls') can be set to process some number of jobs at
a time for every iteration of the loop, instead of all jobs. This allows for the controll of the processing to
propagate through the pipeline more horizontally or vertically.
"""


import datajoint as dj
import inspect
import time
import os
import platform
from datetime import datetime


_populate_settings = {
    'display_progress': True,
    'reserve_jobs': True,
    'suppress_errors': True}


class WorkerLog(dj.Manual):
    definition = """
    # Registration of processing jobs running .populate() jobs or custom function
    process_timestamp : datetime      # timestamp of the processing job
    process           : varchar(64)
    ---
    worker_name=''    : varchar(255)  # name of the worker
    host              : varchar(255)  # system hostname
    user=''           : varchar(255)  # database user
    pid=0             : int unsigned  # system process id  
    """

    _table_name = '~worker_log'

    @classmethod
    def log_process_job(cls, process, worker_name='', db_prefix=''):
        if isinstance(process, dj.user_tables.TableMeta):
            schema_name, table_name = process.full_table_name.split('.')
            schema_name = schema_name.strip('`').replace(db_prefix, '')
            table_name = dj.utils.to_camel_case(table_name.strip('`'))
            process_name = f'{schema_name}.{table_name}'
            user = process.connection.get_user()
        elif inspect.isfunction(process) or inspect.ismethod(process):
            process_name = process.__name__
            user = ''
        else:
            raise ValueError('Input process must be either a DataJoint table or a function')

        if not worker_name:
            frame = inspect.currentframe()
            function_name = frame.f_back.f_code.co_name
            module_name = inspect.getmodule(frame.f_back).__name__
            worker_name = f'{module_name}.{function_name}'

        cls.insert1({'process': process_name,
                     'process_timestamp': datetime.utcnow(),
                     'worker_name': worker_name,
                     'host': platform.node(),
                     'user': user,
                     'pid': os.getpid()})

    @classmethod
    def get_recent_jobs(cls, backtrack_minutes=60):
        recent = (cls.proj(
            minute_elapsed='TIMESTAMPDIFF(MINUTE, process_timestamp, UTC_TIMESTAMP())')
                  & f'minute_elapsed < {backtrack_minutes}')

        recent_jobs = dj.U('process').aggr(
            cls & recent,
            worker_count='count(DISTINCT pid)',
            minutes_since_oldest='TIMESTAMPDIFF(MINUTE, MIN(process_timestamp), UTC_TIMESTAMP())',
            minutes_since_newest='TIMESTAMPDIFF(MINUTE, MAX(process_timestamp), UTC_TIMESTAMP())')

        return recent_jobs

    @classmethod
    def delete_old_logs(cls, cutoff_days=3):
        old_jobs = (cls.proj(
            elapsed_days=f'TIMESTAMPDIFF(DAY, process_timestamp, "{datetime.utcnow()}")')
                    & f'elapsed_days > {cutoff_days}')
        if old_jobs:
            with dj.config(safemode=False):
                (cls & old_jobs).delete_quick()


class DataJointWorker:
    """
    A decorator class for running and managing the populate jobs
    """

    def __init__(self, worker_name, worker_schema_name, *,
                 run_duration=-1, sleep_duration=60,
                 autoclear_error_patterns=[], db_prefix=''):
        self.name = worker_name
        self._worker_schema = dj.schema(worker_schema_name)
        self._worker_schema(WorkerLog)

        self._autoclear_error_patterns = autoclear_error_patterns
        self._run_duration = run_duration
        self._sleep_duration = sleep_duration
        self._db_prefix = db_prefix

        self._processes_to_run = []
        self._pipeline_modules = {}

    def __call__(self, process, **kwargs):
        if isinstance(process, dj.user_tables.TableMeta):
            self._processes_to_run.append(('dj_table', process, kwargs))
            schema_name = process.full_table_name.split('.')[0].replace('`', '')
            if schema_name not in self._pipeline_modules:
                self._pipeline_modules[schema_name] = dj.create_virtual_module(schema_name, schema_name)
        elif inspect.isfunction(process) or inspect.ismethod(process):
            self._processes_to_run.append(('function', process, kwargs))
        else:
            raise NotImplemented(f'Unable to handle processing step of type {type(process)}')

    def run(self):
        start_time = time.time()
        while (time.time() - start_time < self._run_duration
               or self._run_duration is None
               or self._run_duration < 0):

            for process_type, process, kwargs in self._processes_to_run:
                WorkerLog.log_process_job(process, worker_name=self.name,
                                          db_prefix=self._db_prefix)
                if process_type == 'dj_table':
                    process.populate(**{**_populate_settings, **kwargs})
                elif process_type == 'function':
                    process(**kwargs)

            _clean_up(self._pipeline_modules.values(),
                      additional_error_patterns=self._autoclear_error_patterns)
            WorkerLog.delete_old_logs()

            time.sleep(self._sleep_duration)


def _clean_up(pipeline_modules, additional_error_patterns=[], stale_hours=24):
    """
    Routine to clear entries from the jobs table that are:
    + generic-type error jobs
    + stale "reserved" jobs
    """
    _generic_errors = ["%Deadlock%", "%DuplicateError%", "%Lock wait timeout%",
                       "%MaxRetryError%", "%KeyboardInterrupt%",
                       "InternalError: (1205%", "%SIGTERM%",
                       "LostConnectionError"]

    for pipeline_module in pipeline_modules:
        # clear generic error jobs
        (pipeline_module.schema.jobs & 'status = "error"'
         & [f'error_message LIKE "{e}"'
            for e in _generic_errors + additional_error_patterns]).delete()
        # clear stale "reserved" jobs
        stale_jobs = ((pipeline_module.schema.jobs & 'status = "reserved"').proj(
            elapsed_days='TIMESTAMPDIFF(HOUR, timestamp, NOW())')
                      & f'elapsed_days > {stale_hours}')
        (pipeline_module.schema.jobs & stale_jobs).delete()
