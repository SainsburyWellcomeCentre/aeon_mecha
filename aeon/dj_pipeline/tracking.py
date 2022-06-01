import datajoint as dj
import pandas as pd
import numpy as np
from matplotlib import path

from aeon.io import api as aeon_api

from . import lab, acquisition, qc
from . import get_schema_name, dict_to_uuid


schema = dj.schema(get_schema_name('tracking'))

pixel_scale = 0.00192  # 1 px = 1.92 mm
arena_center_x, arena_center_y = 1.475, 1.075  # center
arena_inner_radius = 0.93  # inner
arena_outer_radius = 0.97  # outer


# ---------- Tracking Method ------------------

@schema
class TrackingMethod(dj.Lookup):
    definition = """
    tracking_method: varchar(16)  
    ---
    tracking_method_description: varchar(256)
    """

    contents = [('DLC', 'Online DeepLabCut as part of Bonsai workflow')]


@schema
class TrackingParamSet(dj.Lookup):
    definition = """  # Parameter set used in a particular TrackingMethod
    tracking_paramset_id:  smallint
    ---
    -> TrackingMethod    
    paramset_description: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, tracking_method: str, paramset_description: str,
                          params: dict, tracking_paramset_id: int = None):
        if tracking_paramset_id is None:
            tracking_paramset_id = (dj.U().aggr(cls, n='max(tracking_paramset_id)').fetch1('n') or 0) + 1

        param_dict = {'tracking_method': tracking_method,
                      'tracking_paramset_id': tracking_paramset_id,
                      'paramset_description': paramset_description,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(
                          {**params, 'tracking_method': tracking_method})
                      }
        param_query = cls & {'param_set_hash': param_dict['param_set_hash']}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1('tracking_paramset_id')
            if existing_paramset_idx == tracking_paramset_id:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f'The specified param-set already exists'
                    f' - with tracking_paramset_id: {existing_paramset_idx}')
        else:
            if {'tracking_paramset_id': tracking_paramset_id} in cls.proj():
                raise dj.DataJointError(
                    f'The specified tracking_paramset_id {tracking_paramset_id} already exists,'
                    f' please pick a different one.')
            cls.insert1(param_dict)

# ---------- Video Tracking ------------------


@schema
class CameraTracking(dj.Imported):
    definition = """  # Tracked objects position data from a particular camera, using a particular tracking method, for a particular chunk
    -> acquisition.Chunk
    -> acquisition.ExperimentCamera
    -> TrackingParamSet
    """

    class Object(dj.Part):
        definition = """  # Position data of object tracked by a particular camera tracking
        -> master
        object_id: int    # object with id = -1 means "unknown/not sure", could potentially be the same object as those with other id value
        ---
        timestamps:        longblob  # (datetime) timestamps of the position data
        position_x:        longblob  # (px) object's x-position, in the arena's coordinate frame
        position_y:        longblob  # (px) object's y-position, in the arena's coordinate frame
        area=null:         longblob  # (px^2) object's size detected in the camera
        """

    @property
    def key_source(self):
        ks = acquisition.Chunk * acquisition.ExperimentCamera * TrackingParamSet
        return (ks
                & 'tracking_paramset_id = 0'
                ^ (qc.CameraQC * acquisition.ExperimentCamera & 'camera_description = "FrameTop"')
                )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (acquisition.Chunk & key).fetch1('chunk_start', 'chunk_end', 'directory_type')

        raw_data_dir = acquisition.Experiment.get_data_directory(key, directory_type=dir_type)
        positiondata = aeon_api.positiondata(raw_data_dir.as_posix(),
                                             start=pd.Timestamp(chunk_start),
                                             end=pd.Timestamp(chunk_end))
        # replace id=NaN with -1
        positiondata.fillna({'id': -1}, inplace=True)

        # Correct for frame offsets from Camera QC
        qc_timestamps, qc_frame_offsets, camera_fs = (qc.CameraQC & key).fetch1(
            'timestamps', 'frame_offset', 'camera_sampling_rate')
        qc_time_offsets = qc_frame_offsets / camera_fs
        qc_time_offsets = np.where(np.isnan(qc_time_offsets), 0, qc_time_offsets)  # set NaNs to 0
        positiondata.index += pd.to_timedelta(qc_time_offsets, 's')

        object_positions = []
        for obj_id in set(positiondata.id.values):
            obj_position = positiondata[positiondata.id == obj_id]

            object_positions.append({
                **key,
                'object_id': obj_id,
                'timestamps': obj_position.index.to_pydatetime(),
                'position_x': obj_position.x.values,
                'position_y': obj_position.y.values,
                'area': obj_position.area.values})

        self.insert1(key)
        self.Object.insert(object_positions)

    @classmethod
    def get_object_position(cls, experiment_name, object_id, start, end,
                            camera_name='FrameTop', tracking_paramset_id=0, in_meter=False):
        table = (cls.Object * acquisition.Chunk.proj('chunk_end')
                 & {'experiment_name': experiment_name}
                 & {'tracking_paramset_id': tracking_paramset_id}
                 & (acquisition.ExperimentCamera & {'camera_description': camera_name}))

        return _get_position(table, object_attr='object_id', object_name=object_id,
                             start_attr='chunk_start', end_attr='chunk_end',
                             start=start, end=end,
                             fetch_attrs=['timestamps', 'position_x', 'position_y', 'area'],
                             attrs_to_scale=['position_x', 'position_y'],
                             scale_factor=pixel_scale if in_meter else 1)


# ---------- HELPER ------------------

def compute_distance(position_df, target):
    assert len(target) == 2
    return np.sqrt(np.square(position_df[['x', 'y']] - target).sum(axis=1))


def is_in_patch(position_df, patch_position, wheel_distance_travelled, patch_radius=0.2):
    distance_from_patch = compute_distance(position_df, patch_position)
    in_patch = distance_from_patch < patch_radius
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel_distance_travelled.diff().rolling('1s').sum() > 1).reindex(
        position_df.index, method='pad')
    time_slice = exit_patch.cumsum()
    return in_wheel.groupby(time_slice).apply(lambda x:x.cumsum()) > 0


def is_position_in_nest(position_df, nest_key):
    """
    Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest
    """
    nest_vertices = list(zip(*(lab.ArenaNest.Vertex & nest_key).fetch(
        'vertex_x', 'vertex_y')))
    nest_path = path.Path(nest_vertices)

    return nest_path.contains_points(position_df[['x', 'y']])


def _get_position(table, object_attr: str, object_name: str,
                  start_attr: str, end_attr: str,
                  start: str, end: str, fetch_attrs: list,
                  attrs_to_scale: list, scale_factor=1.0):
    obj_restriction = {object_attr: object_name}

    start_restriction = f'"{start}" BETWEEN {start_attr} AND {end_attr}'
    end_restriction = f'"{end}" BETWEEN {start_attr} AND {end_attr}'

    start_query = table & obj_restriction & start_restriction
    end_query = table & obj_restriction & end_restriction
    if not (start_query and end_query):
        raise ValueError(f'No position data found for {object_name} between {start} and {end}')

    time_restriction = f'{start_attr} >= "{start_query.fetch1(start_attr)}"' \
                       f' AND {start_attr} < "{end_query.fetch1(end_attr)}"'

    # subject's position data in the time slice
    fetched_data = (table & obj_restriction & time_restriction).fetch(
        *fetch_attrs, order_by=start_attr)

    if not len(fetched_data[0]):
        raise ValueError(f'No position data found for {object_name} between {start} and {end}')

    timestamp_attr = next(attr for attr in fetch_attrs if 'timestamps' in attr)

    # stack and structure in pandas DataFrame
    position = pd.DataFrame({k: np.hstack(v) * scale_factor if k in attrs_to_scale else np.hstack(v)
                             for k, v in zip(fetch_attrs, fetched_data)})
    position.set_index(timestamp_attr, inplace=True)

    time_mask = np.logical_and(position.index >= start, position.index < end)

    return position[time_mask]