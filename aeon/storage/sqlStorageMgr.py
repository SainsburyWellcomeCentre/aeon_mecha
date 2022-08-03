
import time
import numpy as np
import pandas as pd
import MySQLdb
import datajoint as dj

from . import storageMgr

class SQLStorageMgr(storageMgr.StorageMgr):

    def __init__(self, host, port, user, passwd):
        self._conn = MySQLdb.connect(host=host,
                                     port=port, user=user,
                                     passwd=passwd)

    def __del__(self):
        self._conn.close()

    def getSessionEndTime(self, session_start_time_str):
        sql_stmt = "SELECT in_arena_end FROM aeon_analysis.__in_arena_end WHERE in_arena_start=\"{:s}\"".format(session_start_time_str)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        session_end_time = cur.fetchone()[0]
        cur.close()
        return session_end_time

    def getSessionPositions(self, session_start_time_str,
                            start_offset_secs, duration_secs):
        sql_stmt = "SELECT timestamps, position_x, position_y FROM aeon_tracking._subject_position WHERE session_start=\"{:s}\"".format(session_start_time_str)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        time_stamps = x = y = None
        time_stamps = np.array([], dtype=object)
        x = np.array([], dtype=np.double)
        y = np.array([], dtype=np.double)
        nChunks = len(records)
        for i, row in enumerate(records):
            timestamps_blob = row[0]
            position_x_blob = row[1]
            position_y_blob = row[2]
            start = time.time()
            time_stamps = np.append(time_stamps,
                                    dj.blob.Blob().unpack(blob=timestamps_blob))
            end = time.time()
            print(f"Unpacking timestamps blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            x = np.append(x, dj.blob.Blob().unpack(blob=position_x_blob))
            end = time.time()
            print(f"Unpacking position x blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            y = np.append(y, dj.blob.Blob().unpack(blob=position_y_blob))
            end = time.time()
            print(f"Unpacking position y blob took {end - start} seconds ({i+1}/{nChunks})")
        cur.close()
#         time_stamps = positions.index
        # time_stamps0_sec = time_stamps[0].timestamp()
        # time_stamps_secs = np.array([ts.timestamp()-time_stamps0_sec for ts in time_stamps])
        time_stamps0_sec = time_stamps[0]
        time_stamps_secs = np.array([ts-time_stamps0_sec for ts in time_stamps])
        if duration_secs <0:
            max_secs = time_stamps_secs.max()
        else:
            max_secs = start_offset_secs+duration_secs
        indices_keep = np.where(
            np.logical_and(start_offset_secs<=time_stamps_secs,
                           time_stamps_secs<max_secs))[0]
#         time_stamps_secs = time_stamps_secs[indices_keep]
        time_stamps = time_stamps[indices_keep]
        x = x[indices_keep]
        y = y[indices_keep]
        answer = pd.DataFrame(data={"x": x, "y": y}, index=time_stamps)
        return answer

    def getWheelAngles(self, start_time_str, end_time_str, patch_label):
        # mysql> select timestamps, angle from aeon_acquisition._food_patch_wheel where chunk_start between session_start and session_end;
        sql_stmt = "SELECT timestamps, angle FROM aeon_acquisition._food_patch_wheel WHERE chunk_start BETWEEN \"{:s}\" AND \"{:s}\" AND food_patch_serial_number=\"{:s}\"".format(start_time_str, end_time_str, patch_label)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        nChunks = len(records)
        all_angles = None
        for i, row in enumerate(records):
            timestamps_blob = row[0]
            angle_blob = row[1]
            start = time.time()
            timestamps_secs = pd.DatetimeIndex(dj.blob.Blob().unpack(blob=timestamps_blob))
            end = time.time()
            print(f"Unpacking timestamps blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            angle = dj.blob.Blob().unpack(blob=angle_blob)
            end = time.time()
            print(f"Unpacking angles blob took {end - start} seconds ({i+1}/{nChunks})")
            angle_series = pd.Series(angle, index=timestamps_secs)
            if all_angles is None:
                all_angles = angle_series
            else:
                all_angles = pd.concat((all_angles, angle_series))
        cur.close()
        return all_angles

    def getFoodPatchEventTimes(self, start_time_str, end_time_str, event_label, patch_label):
        # SELECT event_time FROM aeon_acquisition._food_patch_event 
        # INNER JOIN aeon_acquisition.`#event_type` ON 
        #  aeon_acquisition._food_patch_event.event_code=aeon_acquisition.`#event_type`.event_code 
        # WHERE aeon_acquisition.`#event_type`.event_type="TriggerPellet" AND 
        #       food_patch_serial_number="COM4" AND 
        #       event_time BETWEEN "2021-10-01 13:03:45.835619" AND "2021-10-01 17:20:20.224289";
        sql_stmt = "SELECT event_time FROM aeon_acquisition._food_patch_event " \
                   "INNER JOIN aeon_acquisition.`#event_type` ON " \
                     "aeon_acquisition._food_patch_event.event_code=aeon_acquisition.`#event_type`.event_code " \
                    "WHERE aeon_acquisition.`#event_type`.event_type=\"{:s}\" AND "\
                         "food_patch_serial_number=\"{:s}\" AND " \
                         "event_time BETWEEN \"{:s}\" AND \"{:s}\"".format(
                             event_label, patch_label, start_time_str,
                             end_time_str)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        event_times = pd.DatetimeIndex([pd.Timestamp(row[0]) for row in records])
        cur.close()
        return event_times


    def getExperimentsNames(self, directory_type="raw"):
        sql_stmt_pattern = 'SELECT DISTINCT experiment_name FROM aeon_acquisition.experiment__directory WHERE directory_type="{:s}";'
        sql_stmt = sql_stmt_pattern.format(directory_type)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        experiments_names = [row[0] for row in records]
        cur.close()
        return experiments_names


    def getSubjectsNames(self, experiment_name):
        sql_stmt = f'SELECT subject FROM aeon_acquisition.experiment__subject WHERE experiment_name="{experiment_name}"'
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        subjects_names = [row[0] for row in records]
        cur.close()
        return subjects_names


    def getSessionDuration(self, experiment_name, subject_name, session_start):
        sql_stmt_pattern = 'SELECT session_duration FROM aeon_acquisition.__session_end WHERE experiment_name="{:s}" AND subject="{:s}" AND session_start="{:s}"'
        sql_stmt = sql_stmt_pattern.format(experiment_name, subject_name,
                                           session_start)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        session_duration_hours = cur.fetchone()[0]
        cur.close()
        session_duration_secs = session_duration_hours * 3600
        return session_duration_secs


    def getSubjectSessionsStartTimes(self, experiment_name, subject_name):
        sql_stmt_pattern = 'SELECT session_start FROM aeon_acquisition.__session WHERE experiment_name="{:s}" AND subject="{:s}";'
        sql_stmt = sql_stmt_pattern.format(experiment_name, subject_name)
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        sessions_start_times = [row[0] for row in records]
        cur.close()
        return sessions_start_times



