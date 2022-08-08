
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

    def getSessionPositions(self, session_start_time,
                            start_offset_secs, duration_secs,
                            time_slice_duration_secs=600):
        start_offset_td = np.timedelta64(start_offset_secs, "s") 
        duration_td = np.timedelta64(duration_secs, "s") 
        time_slice_duration_td = np.timedelta64(time_slice_duration_secs, "s") 

        time_slice_start_minus_dt = session_start_time + start_offset_td - \
            time_slice_duration_td
        time_slice_end_plus_dt = session_start_time + start_offset_td + \
            duration_td + time_slice_duration_td
        session_start_time_str = \
            session_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        time_slice_start_minus_dt_str = \
            time_slice_start_minus_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        time_slice_end_plus_dt_str = \
            time_slice_end_plus_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        sql_stmt = 'SELECT timestamps, position_x, position_y ' \
                   'FROM aeon_analysis.__visit_subject_position__time_slice ' \
                  f'WHERE visit_start="{session_start_time_str}" AND ' \
                      f'time_slice_start>"{time_slice_start_minus_dt_str}" AND ' \
                      f'time_slice_end<"{time_slice_end_plus_dt_str}"'
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        records = cur.fetchall()
        timestamps = x = y = None
        nChunks = len(records)
        for i, row in enumerate(records):
            timestamps_blob = row[0]
            position_x_blob = row[1]
            position_y_blob = row[2]
            start = time.time()
            unpack_timestamps_res = dj.blob.Blob().unpack(blob=timestamps_blob)
            end = time.time()
            print(f"Unpacking timestamps blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            unpack_x_res = dj.blob.Blob().unpack(blob=position_x_blob)
            end = time.time()
            print(f"Unpacking position x blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            unpack_y_res = dj.blob.Blob().unpack(blob=position_y_blob)
            end = time.time()
            print(f"Unpacking position y blob took {end - start} seconds ({i+1}/{nChunks})")
            if timestamps is None:
                timestamps = unpack_timestamps_res
                x = unpack_x_res
                y = unpack_y_res
            else:
                timestamps = np.append(timestamps, unpack_timestamps_res)
                x = np.append(x, unpack_x_res)
                y = np.append(y, unpack_y_res)
        cur.close()
        # timestamps_secs = (timestamps - timestamps[0])/np.timedelta64(1, "s")
        # if duration_secs <0:
            # max_secs = timestamps_secs.max()
        # else:
            # max_secs = duration_secs
        indices_keep = np.where(
            # np.logical_and(start_offset_secs<=timestamps_secs, timestamps_secs<max_secs))[0]
            np.logical_and(session_start_time+start_offset_td<=timestamps,
                           timestamps<session_start_time+start_offset_td+duration_td))[0]
        timestamps = timestamps[indices_keep]
        x = x[indices_keep]
        y = y[indices_keep]
        answer = pd.DataFrame(data={"x": x, "y": y}, index=timestamps)
        return answer

    def getWheelAngles(self, experiment_name,
                       session_start_time, start_offset_secs, duration_secs,
                       patch_label, chunk_duration_secs=3600):
        start_offset_td = np.timedelta64(start_offset_secs, "s") 
        duration_td = np.timedelta64(duration_secs, "s") 
        chunk_duration_td = np.timedelta64(chunk_duration_secs, "s") 

        chunk_start_minus_dt = session_start_time + start_offset_td - \
            chunk_duration_td
        chunck_end_plus_dt = session_start_time + start_offset_td + \
            duration_td + chunk_duration_td
        session_start_time_str = \
            session_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        chunk_start_minus_dt_str = \
            chunk_start_minus_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        chunck_end_plus_dt_str = \
            chunck_end_plus_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        sql_stmt = f'SELECT timestamps, angle FROM aeon_acquisition._food_patch_wheel WHERE experiment_name="{experiment_name}" AND chunk_start BETWEEN "{chunk_start_minus_dt_str}" AND "{chunck_end_plus_dt_str}" AND food_patch_serial_number="{patch_label}"'
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
            timestamps = dj.blob.Blob().unpack(blob=timestamps_blob)
            end = time.time()
            print(f"Unpacking timestamps blob took {end - start} seconds ({i+1}/{nChunks})")
            start = time.time()
            angle = dj.blob.Blob().unpack(blob=angle_blob)
            end = time.time()
            print(f"Unpacking angles blob took {end - start} seconds ({i+1}/{nChunks})")
            angle_series = pd.Series(angle, index=timestamps)
            if all_angles is None:
                all_angles = angle_series
            else:
                all_angles = pd.concat((all_angles, angle_series))
        cur.close()
        indices_keep = np.where(
            # np.logical_and(start_offset_secs<=timestamps_secs, timestamps_secs<max_secs))[0]
            np.logical_and(session_start_time+start_offset_td<=all_angles.index,
                           all_angles.index<session_start_time+start_offset_td+duration_td))[0]
        all_angles = all_angles[indices_keep]
        return all_angles

    def getFoodPatchEventTimes(self, start_time_str, end_time_str, event_label, patch_label):
        sql_stmt = 'SELECT event_time FROM aeon_acquisition._food_patch_event ' \
                   'INNER JOIN aeon_acquisition.`#event_type` ON ' \
                     'aeon_acquisition._food_patch_event.event_code=aeon_acquisition.`#event_type`.event_code ' \
                  f'WHERE aeon_acquisition.`#event_type`.event_type="{event_label}" AND '\
                    f'food_patch_serial_number="{patch_label}" AND ' \
                    f'event_time BETWEEN "{start_time_str}" AND "{end_time_str}"'
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
        sql_stmt = f'SELECT DISTINCT experiment_name FROM aeon_acquisition.experiment__directory WHERE directory_type="{directory_type}"'
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


    def getSubjectSessionsStartTimes(self, experiment_name, subject_name):
        sql_stmt = f'SELECT visit_start FROM aeon_analysis.visit WHERE experiment_name="{experiment_name}" AND subject="{subject_name}"'
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


    def getSessionEndTime(self, experiment_name, session_start_time_str):
        sql_stmt = f'SELECT visit_end FROM aeon_analysis.visit_end WHERE experiment_name="{experiment_name}" AND visit_start="{session_start_time_str}"'
        cur = self._conn.cursor()
        print("Executing: " + sql_stmt)
        start = time.time()
        cur.execute(sql_stmt)
        end = time.time()
        print(f"Elapsed time: {end - start} seconds")
        session_end_time = cur.fetchone()[0]
        cur.close()
        return session_end_time


    def getSessionDuration(self, experiment_name, subject_name, session_start):
        sql_stmt = f'SELECT visit_duration FROM aeon_analysis.visit_end WHERE experiment_name="{experiment_name}" AND subject="{subject_name}" AND visit_start="{session_start}"'
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


