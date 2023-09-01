from pathlib import Path

import datajoint as dj
import matplotlib.path
import numpy as np
import pandas as pd

from aeon.dj_pipeline import (
    acquisition,
    dict_to_uuid,
    get_schema_name,
    lab,
    qc,
    streams,
)
from aeon.io import api as io_api
from aeon.schema.social import Pose

from . import acquisition, dict_to_uuid, get_schema_name, lab, qc

schema = dj.schema(get_schema_name("tracking"))

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

    contents = [
        ("DLC", "Online DeepLabCut as part of Bonsai workflow"),
        ("SLEAP", "Online SLEAP as part of Bonsai workflow"),
    ]


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

    contents = [
        (
            0,
            "DLC",
            "Default DLC method from online Bonsai - with params as empty dictionary",
            dict_to_uuid({"tracking_method": "DLC"}),
            {},
        ),
        (
            1,
            "SLEAP",
            "Default SLEAP method from online Bonsai - with params as empty dictionary",
            dict_to_uuid({"tracking_method": "SLEAP"}),
            {},
        ),
    ]

    @classmethod
    def insert_new_params(
        cls,
        tracking_method: str,
        paramset_description: str,
        params: dict,
        tracking_paramset_id: int = None,
    ):
        if tracking_paramset_id is None:
            tracking_paramset_id = (
                dj.U().aggr(cls, n="max(tracking_paramset_id)").fetch1("n") or 0
            ) + 1

        param_dict = {
            "tracking_method": tracking_method,
            "tracking_paramset_id": tracking_paramset_id,
            "paramset_description": paramset_description,
            "params": params,
            "param_set_hash": dict_to_uuid(
                {**params, "tracking_method": tracking_method}
            ),
        }
        param_query = cls & {"param_set_hash": param_dict["param_set_hash"]}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1("tracking_paramset_id")
            if (
                existing_paramset_idx == tracking_paramset_id
            ):  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f"The specified param-set already exists"
                    f" - with tracking_paramset_id: {existing_paramset_idx}"
                )
        else:
            if {"tracking_paramset_id": tracking_paramset_id} in cls.proj():
                raise dj.DataJointError(
                    f"The specified tracking_paramset_id {tracking_paramset_id} already exists,"
                    f" please pick a different one."
                )
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
        return (
            ks
            * (
                qc.CameraQC * acquisition.ExperimentCamera
                & f"camera_description in {tuple(set(acquisition._ref_device_mapping.values()))}"
            ).proj()
            & "tracking_paramset_id = 0"
        )

    def make(self, key):
        chunk_start, chunk_end, dir_type = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end", "directory_type"
        )
        camera = (acquisition.ExperimentCamera & key).fetch1("camera_description")

        raw_data_dir = acquisition.Experiment.get_data_directory(
            key, directory_type=dir_type
        )

        device = getattr(
            acquisition._device_schema_mapping[key["experiment_name"]], camera
        )

        positiondata = io_api.load(
            root=raw_data_dir.as_posix(),
            reader=device.Position,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        # replace id=NaN with -1
        positiondata.fillna({"id": -1}, inplace=True)

        # Retrieve frame offsets from Camera QC
        qc_timestamps, qc_frame_offsets, camera_fs = (
            qc.CameraQC * acquisition.ExperimentCamera & key
        ).fetch1("timestamps", "frame_offset", "camera_sampling_rate")

        # For cases where position data is shorter than video data (from QC) - truncate video data
        # - fix for issue: https://github.com/SainsburyWellcomeCentre/aeon_mecha/issues/130
        max_frame_count = min(len(positiondata), len(qc_timestamps))
        qc_frame_offsets = qc_frame_offsets[:max_frame_count]
        positiondata = positiondata[:max_frame_count]

        # Correct for frame offsets from Camera QC
        qc_time_offsets = qc_frame_offsets / camera_fs
        qc_time_offsets = np.where(
            np.isnan(qc_time_offsets), 0, qc_time_offsets
        )  # set NaNs to 0
        positiondata.index += pd.to_timedelta(qc_time_offsets, "s")

        object_positions = []
        for obj_id in set(positiondata.id.values):
            obj_position = positiondata[positiondata.id == obj_id]

            object_positions.append(
                {
                    **key,
                    "object_id": obj_id,
                    "timestamps": obj_position.index.values,
                    "position_x": obj_position.x.values,
                    "position_y": obj_position.y.values,
                    "area": obj_position.area.values,
                }
            )

        self.insert1(key)
        self.Object.insert(object_positions)

    @classmethod
    def get_object_position(
        cls,
        experiment_name,
        object_id,
        start,
        end,
        camera_name="FrameTop",
        tracking_paramset_id=0,
        in_meter=False,
    ):
        table = (
            cls.Object * acquisition.Chunk.proj("chunk_end")
            & {"experiment_name": experiment_name}
            & {"tracking_paramset_id": tracking_paramset_id}
            & (acquisition.ExperimentCamera & {"camera_description": camera_name})
        )

        return _get_position(
            table,
            object_attr="object_id",
            object_name=object_id,
            start_attr="chunk_start",
            end_attr="chunk_end",
            start=start,
            end=end,
            fetch_attrs=["timestamps", "position_x", "position_y", "area"],
            attrs_to_scale=["position_x", "position_y"],
            scale_factor=pixel_scale if in_meter else 1,
        )


# ---------- VideoSource  ------------------


@schema
class VideoSourceTracking(dj.Imported):
    definition = """  # Tracked objects position data from a particular VideoSource for multi-animal experiment using the SLEAP tracking method per chunk
    -> acquisition.Chunk
    -> streams.VideoSource
    -> TrackingParamSet
    """

    class Point(dj.Part):
        definition = """
        -> master
        point_name: varchar(16)   
        ---
        point_x:          longblob
        point_y:          longblob
        point_likelihood: longblob
        """
    
    class Pose(dj.Part):
        definition = """
        -> master
        pose_name: varchar(16)   
        class:                  smallint   
        ---
        class_likelihood:       longblob   
        centroid_x:             longblob  
        centroid_y:             longblob  
        centroid_likelihood:    longblob  
        pose_timestamps:        longblob 
        point_collection=null:  varchar(1000)  # List of point names  
        """
        
    class PointCollection(dj.Part):
        definition = """
        -> master.Pose
        -> master.Point
        """
    
    @property
    def key_source(self):
        return (acquisition.Chunk & "experiment_name='multianimal'" )  * (streams.VideoSourcePosition & (streams.VideoSource & "video_source_name='CameraTop'")) * (TrackingParamSet & "tracking_paramset_id = 1") # SLEAP & CameraTop

    def make(self, key):
        chunk_start, chunk_end, dir_type = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end", "directory_type"
        )
        raw_data_dir = acquisition.Experiment.get_data_directory(
            key, directory_type=dir_type
        )

        # This needs to be modified later
        sleap_reader = Pose(pattern="", columns=["class", "class_confidence", "centroid_x", "centroid_y", "centroid_confidence"])
        tracking_file_path = "/ceph/aeon/aeon/data/processed/test-node1/1234567/2023-08-10T18-31-00/macentroid/test-node1_1234567_2023-08-10T18-31-00_macentroid.bin"  # temp file path for testing

        tracking_df = sleap_reader.read(Path(tracking_file_path))

        pose_list = []
        for part_name in ["body"]:
            
            for class_id in tracking_df["class"].unique():
                
                class_df = tracking_df[tracking_df["class"] == class_id]

                pose_list.append(
                    {
                        **key,
                        "pose_name": part_name,
                        "class": class_id,
                        "class_likelihood": class_df["class_likelihood"].values,
                        "centroid_x": class_df["x"].values,
                        "centroid_y": class_df["y"].values,
                        "centroid_likelihood": class_df["part_likelihood"].values,
                        "pose_timestamps": class_df.index.values,
                        "point_collection": "",
                    }
                )

        self.insert1(key)
        self.Pose.insert(pose_list)


# ---------- HELPER ------------------


def compute_distance(position_df, target, xcol="x", ycol="y"):
    assert len(target) == 2
    return np.sqrt(np.square(position_df[[xcol, ycol]] - target).sum(axis=1))


def is_position_in_patch(
    position_df, patch_position, wheel_distance_travelled, patch_radius=0.2
) -> pd.Series:
    distance_from_patch = compute_distance(position_df, patch_position)
    in_patch = distance_from_patch < patch_radius
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel_distance_travelled.diff().rolling("1s").sum() > 1).reindex(
        position_df.index, method="pad"
    )
    time_slice = exit_patch.cumsum()
    return in_patch & (in_wheel.groupby(time_slice).apply(lambda x: x.cumsum()) > 0)


def is_position_in_nest(position_df, nest_key, xcol="x", ycol="y") -> pd.Series:
    """
    Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest
    """
    nest_vertices = list(
        zip(*(lab.ArenaNest.Vertex & nest_key).fetch("vertex_x", "vertex_y"))
    )
    nest_path = matplotlib.path.Path(nest_vertices)
    position_df["in_nest"] = nest_path.contains_points(position_df[[xcol, ycol]])
    return position_df["in_nest"]


def _get_position(
    table,
    object_attr: str,
    object_name: str,
    start_attr: str,
    end_attr: str,
    start: str,
    end: str,
    fetch_attrs: list,
    attrs_to_scale: list,
    scale_factor=1.0,
):
    obj_restriction = {object_attr: object_name}

    start_restriction = f'"{start}" BETWEEN {start_attr} AND {end_attr}'
    end_restriction = f'"{end}" BETWEEN {start_attr} AND {end_attr}'

    start_query = table & obj_restriction & start_restriction
    end_query = table & obj_restriction & end_restriction
    if not (start_query and end_query):
        raise ValueError(
            f"No position data found for {object_name} between {start} and {end}"
        )

    time_restriction = (
        f'{start_attr} >= "{min(start_query.fetch(start_attr))}"'
        f' AND {start_attr} < "{max(end_query.fetch(end_attr))}"'
    )

    # subject's position data in the time slice
    fetched_data = (table & obj_restriction & time_restriction).fetch(
        *fetch_attrs, order_by=start_attr
    )

    if not len(fetched_data[0]):
        raise ValueError(
            f"No position data found for {object_name} between {start} and {end}"
        )

    timestamp_attr = next(attr for attr in fetch_attrs if "timestamps" in attr)

    # stack and structure in pandas DataFrame
    position = pd.DataFrame(
        {
            k: np.hstack(v) * scale_factor if k in attrs_to_scale else np.hstack(v)
            for k, v in zip(fetch_attrs, fetched_data)
        }
    )
    position.set_index(timestamp_attr, inplace=True)

    time_mask = np.logical_and(position.index >= start, position.index < end)

    return position[time_mask]
