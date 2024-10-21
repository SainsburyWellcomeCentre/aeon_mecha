from pathlib import Path

import datajoint as dj
import matplotlib.path
import numpy as np
import pandas as pd

from aeon.dj_pipeline import acquisition, dict_to_uuid, get_schema_name, lab, qc, streams
from aeon.io import api as io_api

aeon_schemas = acquisition.aeon_schemas

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
            tracking_paramset_id = (dj.U().aggr(cls, n="max(tracking_paramset_id)").fetch1("n") or 0) + 1

        param_dict = {
            "tracking_method": tracking_method,
            "tracking_paramset_id": tracking_paramset_id,
            "paramset_description": paramset_description,
            "params": params,
            "param_set_hash": dict_to_uuid({**params, "tracking_method": tracking_method}),
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


# ---------- VideoSource  ------------------


@schema
class SLEAPTracking(dj.Imported):
    definition = """  # Tracked objects position data from a particular VideoSource for multi-animal experiment using the SLEAP tracking method per chunk
    -> acquisition.Chunk
    -> streams.SpinnakerVideoSource
    -> TrackingParamSet
    """

    class PoseIdentity(dj.Part):
        definition = """
        -> master
        identity_idx:           smallint
        ---
        identity_name:          varchar(16)
        identity_likelihood:    longblob
        anchor_part:         varchar(16)  # the name of the point used as anchor node for this class
        """

    class Part(dj.Part):
        definition = """
        -> master.PoseIdentity
        part_name: varchar(16)
        ---
        sample_count: int      # number of data points acquired from this stream for a given chunk
        x:          longblob
        y:          longblob
        likelihood: longblob
        timestamps: longblob
        """

    @property
    def key_source(self):
        return (
            acquisition.Chunk
            * (
                streams.SpinnakerVideoSource.join(streams.SpinnakerVideoSource.RemovalTime, left=True)
                & "spinnaker_video_source_name='CameraTop'"
            )
            * (TrackingParamSet & "tracking_paramset_id = 1")
            & "chunk_start >= spinnaker_video_source_install_time"
            & 'chunk_start < IFNULL(spinnaker_video_source_removal_time, "2200-01-01")'
        )  # SLEAP & CameraTop

    def make(self, key):
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

        data_dirs = acquisition.Experiment.get_data_directories(key)

        device_name = (streams.SpinnakerVideoSource & key).fetch1("spinnaker_video_source_name")

        devices_schema = getattr(
            aeon_schemas,
            (acquisition.Experiment.DevicesSchema & {"experiment_name": key["experiment_name"]}).fetch1(
                "devices_schema_name"
            ),
        )

        stream_reader = getattr(getattr(devices_schema, device_name), "Pose")

        # special ingestion case for social0.2 full-pose data (using Pose reader from social03)
        if key["experiment_name"].startswith("social0.2"):
            from aeon.io import reader as io_reader
            stream_reader = getattr(getattr(devices_schema, device_name), "Pose03")
            assert isinstance(stream_reader, io_reader.Pose), "Pose03 is not a Pose reader"
            data_dirs = [acquisition.Experiment.get_data_directory(key, "processed")]

        pose_data = io_api.load(
            root=data_dirs,
            reader=stream_reader,
            start=pd.Timestamp(chunk_start),
            end=pd.Timestamp(chunk_end),
        )

        if not len(pose_data):
            raise ValueError(f"No SLEAP data found for {key['experiment_name']} - {device_name}")

        # get identity names
        class_names = np.unique(pose_data.identity)
        identity_mapping = {n: i for i, n in enumerate(class_names)}

        # ingest parts and classes
        pose_identity_entries, part_entries = [], []
        for identity in identity_mapping:
            identity_position = pose_data[pose_data["identity"] == identity]
            if identity_position.empty:
                continue

            # get anchor part - always the first one of all the body parts
            # FIXME: the logic below to get "anchor_part" is not robust, it relies on the ordering of the unique parts
            #  but if there are missing frames for the actual anchor part, it will be missed
            #  and another part will be incorrectly chosen as "anchor_part"
            anchor_part = np.unique(identity_position.part)[0]

            for part in set(identity_position.part.values):
                part_position = identity_position[identity_position.part == part]
                part_entries.append(
                    {
                        **key,
                        "identity_idx": identity_mapping[identity],
                        "part_name": part,
                        "timestamps": part_position.index.values,
                        "x": part_position.x.values,
                        "y": part_position.y.values,
                        "likelihood": part_position.part_likelihood.values,
                        "sample_count": len(part_position.index.values),
                    }
                )
                if part == anchor_part:
                    identity_likelihood = part_position.identity_likelihood.values
                    if isinstance(identity_likelihood[0], dict):
                        identity_likelihood = np.array([v[identity] for v in identity_likelihood])

            pose_identity_entries.append(
                {
                    **key,
                    "identity_idx": identity_mapping[identity],
                    "identity_name": identity,
                    "anchor_part": anchor_part,
                    "identity_likelihood": identity_likelihood,
                }
            )

        self.insert1(key)
        self.PoseIdentity.insert(pose_identity_entries)
        self.Part.insert(part_entries)


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
    """Given the session key and the position data - arrays of x and y
    return an array of boolean indicating whether or not a position is inside the nest.
    """
    nest_vertices = list(zip(*(lab.ArenaNest.Vertex & nest_key).fetch("vertex_x", "vertex_y")))
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
        raise ValueError(f"No position data found for {object_name} between {start} and {end}")

    time_restriction = (
        f'{start_attr} >= "{min(start_query.fetch(start_attr))}"'
        f' AND {start_attr} < "{max(end_query.fetch(end_attr))}"'
    )

    # subject's position data in the time slice
    fetched_data = (table & obj_restriction & time_restriction).fetch(*fetch_attrs, order_by=start_attr)

    if not len(fetched_data[0]):
        raise ValueError(f"No position data found for {object_name} between {start} and {end}")

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
