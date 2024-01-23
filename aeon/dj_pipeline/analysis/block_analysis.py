import datetime
import datajoint as dj
import pandas as pd
import json

from aeon.analysis import utils as analysis_utils

from aeon.dj_pipeline import get_schema_name, fetch_stream
from aeon.dj_pipeline import acquisition, tracking, streams
from aeon.dj_pipeline.analysis.visit import (
    get_maintenance_periods,
    filter_out_maintenance_periods,
)

schema = dj.schema(get_schema_name("analysis"))


@schema
class Block(dj.Manual):
    definition = """
    -> acquisition.Experiment
    block_start: datetime(6)
    ---
    block_end: datetime(6)
    """


@schema
class BlockAnalysis(dj.Computed):
    definition = """
    -> Block
    """

    class Patch(dj.Part):
        definition = """
        -> master
        patch_name: varchar(36)  # e.g. Patch1, Patch2
        ---
        pellet_count: int
        pellet_timestamps: longblob
        total_distance_travelled: float
        cumulative_distance_travelled: longblob
        wheel_timestamps: longblob
        """

    class Subject(dj.Part):
        definition = """
        -> master
        subject_name: varchar(32)
        ---
        weights: longblob
        weight_timestamps: longblob
        position_x: longblob
        position_y: longblob
        position_likelihood: longblob
        position_timestamps: longblob
        """

    def make(self, key):
        block_start, block_end = (Block & key).fetch1("block_start", "block_end")

        start_restriction = f'"{block_start}" BETWEEN chunk_start AND chunk_end'
        end_restriction = f'"{block_end}" BETWEEN chunk_start AND chunk_end'

        start_query = acquisition.Chunk & start_restriction
        end_query = acquisition.Chunk & end_restriction
        if not (start_query and end_query):
            raise ValueError(f"No Chunk found between {block_start} and {block_end}")

        time_restriction = (
            f'chunk_start >= "{min(start_query.fetch("chunk_start"))}"'
            f' AND chunk_start < "{max(end_query.fetch("chunk_end"))}"'
        )

        self.insert1(key)

        # Patch data - TriggerPellet, DepletionState, Encoder (distancetravelled)
        maintenance_period = get_maintenance_periods(key["experiment_name"], block_start, block_end)

        patch_query = (
            streams.UndergroundFeeder.join(streams.UndergroundFeeder.RemovalTime, left=True)
            & key
            & f'"{block_start}" >= underground_feeder_install_time'
            & f'"{block_end}" < IFNULL(underground_feeder_removal_time, "2200-01-01")'
        )
        patch_keys, patch_names = patch_query.fetch("KEY", "underground_feeder_name")

        for patch_key, patch_name in zip(patch_keys, patch_names):
            delivered_pellet_df = fetch_stream(
                streams.UndergroundFeederBeamBreak & patch_key & time_restriction
            )[block_start:block_end]
            # filter out maintenance period based on logs
            pellet_df = filter_out_maintenance_periods(
                delivered_pellet_df,
                maintenance_period,
                block_end,
                dropna=True,
            )
            # wheel data (encoder)
            encoder_df = fetch_stream(streams.UndergroundFeederEncoder & patch_key & time_restriction)[
                block_start:block_end
            ]
            # filter out maintenance period based on logs
            encoder_df = filter_out_maintenance_periods(encoder_df, maintenance_period, block_end)
            encoder_df["distance_travelled"] = analysis_utils.distancetravelled(encoder_df.angle)
            self.Patch.insert1(
                {
                    **key,
                    "patch_name": patch_name,
                    "pellet_count": len(pellet_df),
                    "pellet_timestamps": pellet_df.index.values,
                    "cumulative_distance_travelled": encoder_df.distance_travelled.values,
                    "total_distance_travelled": encoder_df.distance_travelled.values[-1],
                    "wheel_timestamps": encoder_df.index.values,
                }
            )

        # Subject data
        subject_events_query = acquisition.Environment.SubjectState & key & time_restriction
        subject_events_df = fetch_stream(subject_events_query)

        subject_names = set(subject_events_df.id)
        for subject_name in subject_names:
            # positions - query for CameraTop, identity_name matches subject_name,
            pos_query = (
                streams.SpinnakerVideoSource
                * tracking.SLEAPTracking.PoseIdentity.proj("identity_name", anchor_part="part_name")
                * tracking.SLEAPTracking.Part
                & {
                    "spinnaker_video_source_name": "CameraTop",
                    "identity_name": subject_name,
                }
                & time_restriction
            )
            pos_df = fetch_stream(pos_query)[block_start:block_end]
            pos_df = filter_out_maintenance_periods(pos_df, maintenance_period, block_end)

            # weights
            weight_query = acquisition.Environment.SubjectWeight & key & time_restriction
            weight_df = fetch_stream(weight_query)[block_start:block_end]
            weight_df.query(f"subject_id == '{subject_name}'", inplace=True)

            self.Subject.insert1(
                {
                    **key,
                    "subject_name": subject_name,
                    "weights": weight_df.weight.values,
                    "weight_timestamps": weight_df.index.values,
                    "position_x": pos_df.x.values,
                    "position_y": pos_df.y.values,
                    "position_likelihood": pos_df.likelihood.values,
                    "position_timestamps": pos_df.index.values,
                }
            )


@schema
class BlockPlots(dj.Computed):
    definition = """
    -> BlockAnalysis
    ---
    subject_positions_plot: longblob
    subject_weights_plot: longblob
    patch_distance_travelled_plot: longblob
    """

    def make(self, key):
        import plotly.graph_objs as go

        # For position data , set confidence threshold to return position values and downsample by 5x
        conf_thresh = 0.9
        downsampling_factor = 5

        # Make plotly plots
        weight_fig = go.Figure()
        pos_fig = go.Figure()
        for subject_data in (BlockAnalysis.Subject & key).fetch(as_dict=True):
            weight_fig.add_trace(
                go.Scatter(
                    x=subject_data["weight_timestamps"],
                    y=subject_data["weights"],
                    mode="lines",
                    name=subject_data["subject_name"],
                )
            )
            mask = subject_data["position_likelihood"] > conf_thresh
            pos_fig.add_trace(
                go.Scatter3d(
                    x=subject_data["position_x"][mask][::downsampling_factor],
                    y=subject_data["position_y"][mask][::downsampling_factor],
                    z=subject_data["position_timestamps"][mask][::downsampling_factor],
                    mode="lines",
                    name=subject_data["subject_name"],
                )
            )

        wheel_fig = go.Figure()
        for patch_data in (BlockAnalysis.Patch & key).fetch(as_dict=True):
            wheel_fig.add_trace(
                go.Scatter(
                    x=patch_data["wheel_timestamps"][::2],
                    y=patch_data["cumulative_distance_travelled"][::2],
                    mode="lines",
                    name=patch_data["patch_name"],
                )
            )

        # insert figures as json-formatted plotly plots
        self.insert1(
            {
                **key,
                "subject_positions_plot": json.loads(pos_fig.to_json()),
                "subject_weights_plot": json.loads(weight_fig.to_json()),
                "patch_distance_travelled_plot": json.loads(wheel_fig.to_json()),
            }
        )


@schema
class BlockDetection(dj.Computed):
    definition = """
    -> acquisition.Chunk
    """

    def make(self, key):
        pass
