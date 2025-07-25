"""DataJoint schema for tracking data."""

import gc
from datetime import UTC, datetime, timezone

import datajoint as dj
import matplotlib.path
import numpy as np
import pandas as pd
from swc.aeon.io import api as io_api

from aeon.dj_pipeline import (
    acquisition,
    dict_to_uuid,
    fetch_stream,
    get_schema_name,
    lab,
    streams,
)
from aeon.dj_pipeline.utils import tracking_utils

aeon_schemas = acquisition.aeon_schemas

schema = dj.schema(get_schema_name("tracking"))
logger = dj.logger

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
        """Insert a new set of parameters for a given tracking method."""
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


# ---------- SLEAP Tracking  ------------------


@schema
class SLEAPTracking(dj.Imported):
    """Tracking data from SLEAP for multi-animal experiments."""

    definition = """ # Position data from a VideoSource for multi-animal experiments using SLEAP per chunk
    -> acquisition.Chunk
    -> streams.SpinnakerVideoSource
    -> TrackingParamSet
    ---
    execution_time=null: datetime  # time of ingestion
    """

    class PoseIdentity(dj.Part):
        definition = """
        -> master
        identity_idx:           smallint
        ---
        identity_name:          varchar(16)
        identity_likelihood:    longblob
        anchor_part:         varchar(16)  # the name of the part used as anchor node for this class
        """

    class AnchorPart(dj.Part):
        definition = """  # Position data of the anchor part of a particular identity
        -> master.PoseIdentity
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        x:          longblob   # (px) x-coordinates of the anchor part
        y:          longblob   # (px) y-coordinates of the anchor part
        likelihood: longblob   # likelihood of the anchor part
        timestamps: longblob   # (datetime) timestamps of the anchor part
        """

    class Part(dj.Part):
        definition = """
        -> master.PoseIdentity
        part_name: varchar(16)
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        x:          longblob   # (px) x-coordinates of the part
        y:          longblob   # (px) y-coordinates of the part
        likelihood: longblob   # likelihood of the part
        timestamps: longblob   # (datetime) timestamps of the part
        """

    @property
    def key_source(self):
        """Return the keys to be processed."""
        return (
            acquisition.Chunk
            * (
                streams.SpinnakerVideoSource.join(
                    streams.SpinnakerVideoSource.RemovalTime, left=True
                )
                & "spinnaker_video_source_name='CameraTop'"
            )
            * (TrackingParamSet & "tracking_paramset_id = 1")
            & "chunk_start >= spinnaker_video_source_install_time"
            & 'chunk_start < IFNULL(spinnaker_video_source_removal_time, "2200-01-01")'
        )  # SLEAP & CameraTop

    def make(self, key):
        """Ingest SLEAP tracking data for a given chunk."""
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end"
        )

        data_dirs = acquisition.Experiment.get_data_directories(key)

        device_name = (streams.SpinnakerVideoSource & key).fetch1(
            "spinnaker_video_source_name"
        )

        devices_schema = getattr(
            aeon_schemas,
            (
                acquisition.Experiment.DevicesSchema
                & {"experiment_name": key["experiment_name"]}
            ).fetch1("devices_schema_name"),
        )

        stream_reader = getattr(devices_schema, device_name).Pose

        pose_data = io_api.load(
            root=data_dirs,
            reader=stream_reader,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
            include_model=False,
        )

        if not len(pose_data):
            raise ValueError(
                f"No SLEAP data found for {key['experiment_name']} - {device_name}"
            )

        # get identity names
        class_names = np.unique(pose_data.identity)
        identity_mapping = {n: i for i, n in enumerate(class_names)}

        # get anchor part
        # ie the body_part with the prefix "anchor_" (there should only be one)
        anchor_part = {
            part for part in pose_data.part.unique() if part.startswith("anchor_")
        }
        if len(anchor_part) != 1:
            raise ValueError(
                f"Anchor part not found or multiple anchor parts found: {anchor_part}"
            )
        anchor_part = anchor_part.pop()

        # ingest parts and classes
        pose_identity_entries, anchor_part_entries, part_entries = [], [], []
        for id_name, id_idx in identity_mapping.items():
            identity_position = pose_data[pose_data["identity"] == id_name]
            if identity_position.empty:
                continue

            for part in set(identity_position.part.values):
                part_position = identity_position[identity_position.part == part]
                if part == anchor_part:
                    identity_likelihood = part_position.identity_likelihood.values
                    if isinstance(identity_likelihood[0], dict):
                        identity_likelihood = np.array(
                            [v[id_name] for v in identity_likelihood]
                        )

                    # assert no duplicate timestamps
                    if len(part_position.index.values) != len(
                        set(part_position.index.values)
                    ):
                        raise ValueError(
                            f"Duplicate timestamps found for identity {id_name} and part {part}"
                            f" - this should not happen - check for chunk-duplicate .bin files"
                        )

                    pose_identity_entries.append(
                        {
                            **key,
                            "identity_idx": id_idx,
                            "identity_name": id_name,
                            "anchor_part": anchor_part,
                            "identity_likelihood": identity_likelihood,
                        }
                    )
                    anchor_part_entries.append(
                        {
                            **key,
                            "identity_idx": id_idx,
                            "x": part_position.x.values,
                            "y": part_position.y.values,
                            "likelihood": part_position.part_likelihood.values,
                            "timestamps": part_position.index.values,
                            "sample_count": len(part_position.index.values),
                        }
                    )
                else:
                    part_entries.append(
                        {
                            **key,
                            "identity_idx": id_idx,
                            "part_name": part,
                            "timestamps": part_position.index.values,
                            "x": part_position.x.values,
                            "y": part_position.y.values,
                            "likelihood": part_position.part_likelihood.values,
                            "sample_count": len(part_position.index.values),
                        }
                    )

        self.insert1({**key, "execution_time": pd.Timestamp.now()})
        self.PoseIdentity.insert(pose_identity_entries)
        self.AnchorPart.insert(anchor_part_entries)
        self.Part.insert(part_entries)

        # explicit garbage collect `pose_data` to avoid memory build up
        del pose_data, identity_position, part_position
        gc.collect()


# ---------- Blob Position Tracking ------------------


@schema
class BlobPosition(dj.Imported):
    definition = """  # Blob object position tracking from a particular camera, for a particular chunk
    -> acquisition.Chunk
    -> streams.SpinnakerVideoSource
    ---
    object_count: int  # number of objects tracked in this chunk
    subject_count: int  # number of subjects present in the arena during this chunk
    subject_names: varchar(256)  # names of subjects present in arena during this chunk
    """

    class Object(dj.Part):
        definition = """  # Position data of object tracked by a particular camera tracking
        -> master
        object_id: int    # id=-1 means "unknown"; could be the same object as those with other values
        ---
        identity_name='': varchar(16)
        sample_count:  int       # number of data points acquired from this stream for a given chunk
        x:             longblob  # (px) object's x-position, in the arena's coordinate frame
        y:             longblob  # (px) object's y-position, in the arena's coordinate frame
        timestamps:    longblob  # (datetime) timestamps of the position data
        area=null:     longblob  # (px^2) object's size detected in the camera
        """

    @property
    def key_source(self):
        """Return the keys to be processed."""
        ks = (
            acquisition.Chunk
            * (
                streams.SpinnakerVideoSource.join(
                    streams.SpinnakerVideoSource.RemovalTime, left=True
                )
                & "spinnaker_video_source_name='CameraTop'"
            )
            & "chunk_start >= spinnaker_video_source_install_time"
            & 'chunk_start < IFNULL(spinnaker_video_source_removal_time, "2200-01-01")'
        )
        return ks - SLEAPTracking  # do this only when SLEAPTracking is not available

    def make(self, key):
        """Ingest blob position data for a given chunk."""
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end"
        )

        data_dirs = acquisition.Experiment.get_data_directories(key)

        device_name = (streams.SpinnakerVideoSource & key).fetch1(
            "spinnaker_video_source_name"
        )

        devices_schema = getattr(
            aeon_schemas,
            (
                acquisition.Experiment.DevicesSchema
                & {"experiment_name": key["experiment_name"]}
            ).fetch1("devices_schema_name"),
        )

        stream_reader = devices_schema.CameraTop.Position

        positiondata = io_api.load(
            root=data_dirs,
            reader=stream_reader,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        if not len(positiondata):
            raise ValueError(
                f"No Blob position data found for {key['experiment_name']} - {device_name}"
            )

        # replace id=NaN with -1
        positiondata.fillna({"id": -1}, inplace=True)
        positiondata["identity_name"] = ""

        # Find animal(s) in the arena during the chunk
        # Get all unique subjects that visited the environment over the entire exp;
        # For each subject, see 'type' of visit most recent to start of block
        # If "Exit", this animal was not in the block.
        subject_visits_df = fetch_stream(
            acquisition.Environment.SubjectVisits
            & {"experiment_name": key["experiment_name"]}
            & f'chunk_start <= "{chunk_start}"'
        )[:chunk_end]
        subject_visits_df = subject_visits_df[subject_visits_df.region == "Environment"]
        subject_visits_df = subject_visits_df[
            ~subject_visits_df.id.str.contains("Test", case=False)
        ]
        subject_names = []
        for subject_name in set(subject_visits_df.id):
            _df = subject_visits_df[subject_visits_df.id == subject_name]
            if _df.type.iloc[-1] != "Exit":
                subject_names.append(subject_name)

        if len(subject_names) == 1:
            # if there is only one known subject, replace all object ids with the subject name
            positiondata["id"] = [0] * len(positiondata)
            positiondata["identity_name"] = subject_names[0]

        object_positions = []
        for obj_id in set(positiondata.id.values):
            obj_position = positiondata[positiondata.id == obj_id]

            object_positions.append(
                {
                    **key,
                    "object_id": obj_id,
                    "identity_name": obj_position.identity_name.values[0],
                    "sample_count": len(obj_position.index.values),
                    "timestamps": obj_position.index.values,
                    "x": obj_position.x.values,
                    "y": obj_position.y.values,
                    "area": obj_position.area.values,
                }
            )

        self.insert1(
            {
                **key,
                "object_count": len(object_positions),
                "subject_count": len(subject_names),
                "subject_names": ",".join(subject_names),
            }
        )
        self.Object.insert(object_positions)


# ---------- Processed Identity Position ------------------


@schema
class DenoisedTracking(dj.Computed):
    definition = """
    -> SLEAPTracking
    ---
    execution_time: datetime
    execution_duration: float  # hours
    """

    class Subject(dj.Part):
        definition = """
        -> master
        subject_name: varchar(32)
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        subject_likelihood: longblob  # likelihood of the subject being identified correctly
        x:          longblob
        y:          longblob
        timestamps: longblob
        likelihood: longblob  # likelihood of the positions (x,y) being identified correctly
        """

    key_source = (
        SLEAPTracking & "experiment_name in ('social0.2-aeon3', 'social0.2-aeon4', "
                        "'social0.3-aeon3', 'social0.3-aeon4', "
                        "'social0.4-aeon3', 'social0.4-aeon4')"
    )

    def make(self, key):
        """Processing of SLEAPTracking data to denoise and clean identity swaps."""
        execution_time = datetime.now(UTC)

        query = (
            SLEAPTracking.PoseIdentity.proj("identity_name", "identity_likelihood")
            * SLEAPTracking.AnchorPart
            & key
        )
        df = fetch_stream(query)

        subject_names = df.identity_name.unique()

        if len(subject_names) > 1:
            # Get arena bounds from database
            active_region_query = acquisition.EpochConfig.ActiveRegion & (
                acquisition.Chunk & key
            )
            df_clean = tracking_utils.clean_swaps(
                df, region_df=active_region_query.fetch(format="frame")
            )
        else:
            df_clean = df

        entries = []
        for subj_name in subject_names:
            subj_df = df_clean[df_clean.identity_name == subj_name]
            if subj_df.empty:
                continue

            entries.append(
                {
                    **key,
                    "subject_name": subj_name,
                    "sample_count": len(subj_df.index.values),
                    "subject_likelihood": subj_df.identity_likelihood.values,
                    "x": subj_df.x.values,
                    "y": subj_df.y.values,
                    "timestamps": subj_df.index.values,
                    "likelihood": subj_df.likelihood.values,
                }
            )

        exec_dur = (datetime.now(UTC) - execution_time).total_seconds() / 3600
        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": exec_dur,
            }
        )
        self.Subject.insert(entries)


# ---------- HELPER ------------------


def compute_distance(position_df, target, xcol="x", ycol="y"):
    """Compute the distance between the position and the target.

    Args:
        position_df (pd.DataFrame): DataFrame containing the position data.
        target (tuple): Tuple of length 2 indicating the target x and y position.
        xcol (str): x column name in ``position_df``. Default is 'x'.
        ycol (str): y column name in ``position_df``. Default is 'y'.
    """
    COORDS = 2  # x, y
    if len(target) != COORDS:
        raise ValueError("Target must be a list of tuple of length 2.")
    return np.sqrt(np.square(position_df[[xcol, ycol]] - target).sum(axis=1))


def is_position_in_patch(
    position_df, patch_position, wheel_distance_travelled, patch_radius=0.2
) -> pd.Series:
    """Returns a boolean array of whether a given position is inside the patch and the wheel is moving.

    Args:
        position_df (pd.DataFrame): DataFrame containing the position data.
        patch_position (tuple): Tuple of length 2 indicating the patch x and y position.
        wheel_distance_travelled (pd.Series): distance travelled by the wheel.
        patch_radius (float): Radius of the patch. Default is 0.2.
    """
    distance_from_patch = compute_distance(position_df, patch_position)
    in_patch = distance_from_patch < patch_radius
    exit_patch = in_patch.astype(np.int8).diff() < 0
    in_wheel = (wheel_distance_travelled.diff().rolling("1s").sum() > 1).reindex(
        position_df.index, method="pad"
    )
    time_slice = exit_patch.cumsum()
    return in_patch & (in_wheel.groupby(time_slice).apply(lambda x: x.cumsum()) > 0)


def is_position_in_nest(position_df, nest_key, xcol="x", ycol="y") -> pd.Series:
    """Check if a position is inside the nest.

    Notes: Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest.
    """
    nest_vertices = list(
        zip(
            *(lab.ArenaNest.Vertex & nest_key).fetch("vertex_x", "vertex_y"),
            strict=False,
        )
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
    """Get the position data for a given object between the specified time range."""
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
            for k, v in zip(fetch_attrs, fetched_data, strict=False)
        }
    )
    position.set_index(timestamp_attr, inplace=True)

    time_mask = np.logical_and(position.index >= start, position.index < end)

    return position[time_mask]
