import os
import platform
import datajoint as dj
import pandas as pd
import inspect
from datetime import datetime

from aeon.dj_pipeline import get_schema_name, db_prefix
from aeon.dj_pipeline import lab, subject, acquisition, tracking, analysis


schema = dj.schema(get_schema_name('monitor'))


@schema
class ProcessJob(dj.Manual):
    definition = """
    # Registration of processing jobs running .populate() jobs (e.g. in process.py script)
    table             : varchar(64)
    process_timestamp : datetime      # timestamp of
    ---
    module_name=''    : varchar(255)  # Name of executing script 
    function_name=''  : varchar(255)  # Name of executing function
    host              : varchar(255)  # system hostname
    user=''           : varchar(255)  # database user
    pid=0             : int unsigned  # system process id  
    """

    @classmethod
    def log_process_job(cls, table):
        schema_name, table_name = table.full_table_name.split('.')
        schema_name = schema_name.strip('`').replace(db_prefix, '')
        table_name = dj.utils.to_camel_case(table_name.strip('`'))

        frame = inspect.currentframe()
        function_name = frame.f_back.f_code.co_name
        module_name = inspect.getmodule(frame.f_back).__name__

        process = {
            'table': f'{schema_name}.{table_name}',
            'process_timestamp': datetime.utcnow(),
            'module_name': module_name,
            'function_name': function_name,
            'host': platform.node(),
            'user': table.connection.get_user(),
            'pid': os.getpid()
        }

        cls.insert1(process)


def print_recent_jobs(backtrack_minutes=60):
    recent = (ProcessJob.proj(
        minute_elapsed='TIMESTAMPDIFF(MINUTE, process_timestamp, UTC_TIMESTAMP())')
              & f'minute_elapsed < {backtrack_minutes}')

    recent_jobs = dj.U('table').aggr(
        ProcessJob & recent,
        worker_count='count(DISTINCT pid)',
        minutes_since_oldest='TIMESTAMPDIFF(MINUTE, MIN(process_timestamp), UTC_TIMESTAMP())',
        minutes_since_newest='TIMESTAMPDIFF(MINUTE, MAX(process_timestamp), UTC_TIMESTAMP())')

    return recent_jobs


def print_current_jobs():
    """
    Return a pandas.DataFrame on the status of each table currently being processed

        table | reserve_count | error_count | oldest_job | newest_job
            - table: {schema}.{table} name
            - reserve_count: number of workers currently working on this table
            - error_count: number of jobs errors for this table
            - oldest_job: timestamp of the oldest job currently being worked on
            - newest_job: timestamp of the most recent job currently being worked on

    Provide insights into the current status of the workers

    One caveat in this function is that we don't know how many workers are being deployed,
     and how they're orchestrated. We can infer by taking the sum of the reserved jobs,
     but this won't reflect idling workers because there's no "key_source" to work on for some
     particular tables
    """
    job_status = []
    for pipeline_module in (lab, subject, acquisition, tracking, analysis):
        reserved = dj.U('table_name').aggr(pipeline_module.schema.jobs & 'status = "reserved"',
                                           reserve_count='count(table_name)',
                                           oldest_job='MIN(timestamp)',
                                           newest_job='MAX(timestamp)')
        errored = dj.U('table_name').aggr(pipeline_module.schema.jobs & 'status = "error"',
                                          error_count='count(table_name)')
        jobs_summary = reserved.join(errored, left=True)

        for job in jobs_summary.fetch(as_dict=True):
            job_status.append({
                'table': f'{pipeline_module.__name__.split(".")[-1]}.{dj.utils.to_camel_case(job.pop("table_name"))}',
                **job})

    if not job_status:
        print('No jobs under process (0 reserved jobs)')
        return

    job_status_df = pd.DataFrame(job_status).set_index('table')
    job_status_df.fillna(0, inplace=True)
    job_status_df = job_status_df.astype({"reserve_count": int, "error_count": int})

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', -1):
        print(job_status_df)

    return job_status_df
