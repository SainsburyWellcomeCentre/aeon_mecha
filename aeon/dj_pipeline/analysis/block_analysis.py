import datetime
import datajoint as dj
import pandas as pd
import json
import numpy as np

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
    block_end=null: datetime(6)
    """


@schema
class BlockAnalysis(dj.Computed):
    definition = """
    -> Block
    ---
    block_duration: float  # (hour)
    """

    key_source = Block & "block_end IS NOT NULL"

    class Patch(dj.Part):
        definition = """
        -> master
        patch_name: varchar(36)  # e.g. Patch1, Patch2
        ---
        pellet_count: int
        pellet_timestamps: longblob
        wheel_cumsum_distance_travelled: longblob  # wheel's cumulative distance travelled
        wheel_timestamps: longblob
        patch_threshold: longblob
        patch_threshold_timestamps: longblob
        patch_rate: float   
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
        cumsum_distance_travelled: longblob  # subject's cumulative distance travelled
        """

    def make(self, key):
        """
        Restrict, fetch and aggregate data from different streams to produce intermediate data products
            at a per-block level (for different patches and different subjects)
        1. Query data for all chunks within the block
        2. Fetch streams, filter by maintenance period
        3. Fetch subject position data (SLEAP)
        4. Aggregate and insert into the table
        """
        block_start, block_end = (Block & key).fetch1("block_start", "block_end")

        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], block_start, block_end
        )

        self.insert1({**key, "block_duration": (block_end - block_start).total_seconds() / 3600})

        # Patch data - TriggerPellet, DepletionState, Encoder (distancetravelled)
        # For wheel data, downsample by 50x - 10Hz
        wheel_downsampling_factor = 50

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
                streams.UndergroundFeederBeamBreak & patch_key & chunk_restriction
            )[block_start:block_end]
            depletion_state_df = fetch_stream(
                streams.UndergroundFeederDepletionState & patch_key & chunk_restriction
            )[block_start:block_end]
            encoder_df = fetch_stream(streams.UndergroundFeederEncoder & patch_key & chunk_restriction)[
                block_start:block_end
            ]
            # filter out maintenance period based on logs
            pellet_df = filter_out_maintenance_periods(
                delivered_pellet_df,
                maintenance_period,
                block_end,
                dropna=True,
            )
            depletion_state_df = filter_out_maintenance_periods(
                depletion_state_df,
                maintenance_period,
                block_end,
                dropna=True,
            )
            encoder_df = filter_out_maintenance_periods(
                encoder_df, maintenance_period, block_end, dropna=True
            )

            encoder_df["distance_travelled"] = analysis_utils.distancetravelled(encoder_df.angle)

            patch_rate = depletion_state_df.rate.unique()
            assert len(patch_rate) == 1  # expects a single rate for this block
            patch_rate = patch_rate[0]

            self.Patch.insert1(
                {
                    **key,
                    "patch_name": patch_name,
                    "pellet_count": len(pellet_df),
                    "pellet_timestamps": pellet_df.index.values,
                    "wheel_cumsum_distance_travelled": encoder_df.distance_travelled.values[
                        ::wheel_downsampling_factor
                    ],
                    "wheel_timestamps": encoder_df.index.values[::wheel_downsampling_factor],
                    "patch_threshold": depletion_state_df.threshold.values,
                    "patch_threshold_timestamps": depletion_state_df.index.values,
                    "patch_rate": patch_rate,
                }
            )

        # Subject data
        subject_events_query = acquisition.Environment.SubjectState & key & chunk_restriction
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
                & chunk_restriction
            )
            pos_df = fetch_stream(pos_query)[block_start:block_end]
            pos_df = filter_out_maintenance_periods(pos_df, maintenance_period, block_end)

            position_diff = np.sqrt(
                (np.square(np.diff(pos_df.x.astype(float))) + np.square(np.diff(pos_df.y.astype(float))))
            )
            cumsum_distance_travelled = np.concatenate([[0], np.cumsum(position_diff)])

            # weights
            weight_query = acquisition.Environment.SubjectWeight & key & chunk_restriction
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
                    "cumsum_distance_travelled": cumsum_distance_travelled,
                }
            )


@schema
class BlockSubjectAnalysis(dj.Computed):
    definition = """
    -> BlockAnalysis
    """

    class Patch(dj.Part):
        definition = """
        -> master
        -> BlockAnalysis.Patch
        -> BlockAnalysis.Subject
        ---
        in_patch_timestamps: longblob  # timestamps in which a particular subject is spending time at a particular patch
        in_patch_time: float  # total seconds spent in this patch for this block
        pellet_count: int
        pellet_timestamps: longblob
        wheel_distance_travelled: longblob  # wheel's cumulative distance travelled
        wheel_timestamps: longblob
        cumulative_sum_preference: longblob  
        windowed_sum_preference: longblob
        """

    def make(self, key):
        pass


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
    -> acquisition.Environment
    """

    def make(self, key):
        """
        On a per-chunk basis, check for the presence of new block, insert into Block table
        """
        # find the 0s
        # that would mark the start of a new block
        # if the 0 is the first index - look back at the previous chunk
        #   if the previous timestamp belongs to a previous epoch -> block_end is the previous timestamp
        #   else block_end is the timestamp of this 0
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        exp_key = {"experiment_name": key["experiment_name"]}
        # only consider the time period between the last block and the current chunk
        previous_block = Block & exp_key & f"block_start <= '{chunk_start}'"
        if previous_block:
            previous_block_key = previous_block.fetch("KEY", limit=1, order_by="block_start DESC")[0]
            previous_block_start = previous_block_key["block_start"]
        else:
            previous_block_key = None
            previous_block_start = (acquisition.Chunk & exp_key).fetch(
                "chunk_start", limit=1, order_by="chunk_start"
            )[0]

        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], previous_block_start, chunk_end
        )

        block_query = acquisition.Environment.BlockState & chunk_restriction
        block_df = fetch_stream(block_query)[previous_block_start:chunk_end]

        block_ends = block_df[block_df.pellet_ct.diff() < 0]

        block_entries = []
        for idx, block_end in enumerate(block_ends.index):
            if idx == 0:
                if previous_block_key:
                    # if there is a previous block - insert "block_end" for the previous block
                    previous_pellet_time = block_df[:block_end].index[-2]
                    previous_epoch = (
                        acquisition.Epoch.join(acquisition.EpochEnd, left=True)
                        & exp_key
                        & f"'{previous_pellet_time}' BETWEEN epoch_start AND IFNULL(epoch_end, '2200-01-01')"
                    ).fetch1("KEY")
                    current_epoch = (
                        acquisition.Epoch.join(acquisition.EpochEnd, left=True)
                        & exp_key
                        & f"'{block_end}' BETWEEN epoch_start AND IFNULL(epoch_end, '2200-01-01')"
                    ).fetch1("KEY")

                    previous_block_key["block_end"] = (
                        block_end if current_epoch == previous_epoch else previous_pellet_time
                    )
                    Block.update1(previous_block_key)
            else:
                block_entries[-1]["block_end"] = block_end
            block_entries.append({**exp_key, "block_start": block_end, "block_end": None})

        Block.insert(block_entries)
        self.insert1(key)
