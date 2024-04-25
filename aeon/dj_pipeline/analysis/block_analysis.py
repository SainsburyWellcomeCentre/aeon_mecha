import json

import datajoint as dj
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import path as mpl_path

from aeon.analysis import utils as analysis_utils
from aeon.dj_pipeline import (acquisition, fetch_stream, get_schema_name,
                              streams, tracking)
from aeon.dj_pipeline.analysis.visit import (filter_out_maintenance_periods,
                                             get_maintenance_periods)

schema = dj.schema(get_schema_name("block_analysis"))
logger = dj.logger


@schema
class Block(dj.Manual):
    definition = """
    -> acquisition.Experiment
    block_start: datetime(6)
    ---
    block_end=null: datetime(6)
    """


@schema
class BlockDetection(dj.Computed):
    definition = """
    -> acquisition.Environment
    """

    def make(self, key):
        """On a per-chunk basis, check for the presence of new block, insert into Block table."""
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


# ---- Block Analysis and Visualization ----

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
        patch_offset: float
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
        """Restrict, fetch and aggregate data from different streams to produce intermediate data products at a per-block level (for different patches and different subjects).
        1. Query data for all chunks within the block.
        2. Fetch streams, filter by maintenance period.
        3. Fetch subject position data (SLEAP).
        4. Aggregate and insert into the table.
        """
        block_start, block_end = (Block & key).fetch1("block_start", "block_end")

        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], block_start, block_end
        )

        # Ensure the relevant streams ingestion are caught up to this block
        chunk_keys = (acquisition.Chunk & key & chunk_restriction).fetch("KEY")
        streams_tables = (
            streams.UndergroundFeederDepletionState,
            streams.UndergroundFeederBeamBreak,
            streams.UndergroundFeederEncoder,
            tracking.SLEAPTracking,
        )
        for streams_table in streams_tables:
            if len(streams_table & chunk_keys) < len(streams_table.key_source & chunk_keys):
                logger.info(
                    f"{streams_table.__name__} not yet fully ingested for block: {key}. Skip BlockAnalysis (to retry later)..."
                )
                return

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

            encoder_df["distance_travelled"] = -1 * analysis_utils.distancetravelled(encoder_df.angle)

            patch_rate = depletion_state_df.rate.unique()
            patch_offset = depletion_state_df.offset.unique()
            assert len(patch_rate) == 1, f"Found multiple patch rates: {patch_rate} for patch: {patch_name}"
            patch_rate = patch_rate[0]
            patch_offset = patch_offset[0]

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
                    "patch_offset": patch_offset,
                }
            )

        # Subject data
        # Get all unique subjects that visited the environment over the entire exp;
        # For each subject, see 'type' of visit most recent to start of block
        # If "Exit", this animal was not in the block.
        subject_visits_df = fetch_stream(
            acquisition.Environment.SubjectVisits
            & key
            & f'chunk_start <= "{chunk_keys[-1]["chunk_start"]}"'
        )[:block_start]
        subject_visits_df = subject_visits_df[subject_visits_df.region == "Environment"]
        subject_names = []
        for subject_name in set(subject_visits_df.id):
            _df = subject_visits_df[subject_visits_df.id == subject_name]
            if _df.type[-1] != "Exit":
                subject_names.append(subject_name)
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
                np.square(np.diff(pos_df.x.astype(float))) + np.square(np.diff(pos_df.y.astype(float)))
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
        wheel_cumsum_distance_travelled: longblob  # wheel's cumulative distance travelled
        """

    class Preference(dj.Part):
        definition = """ # Measure of preference for a particular patch from a particular subject
        -> master
        -> BlockAnalysis.Patch
        -> BlockAnalysis.Subject
        ---
        cumulative_preference_by_wheel: longblob
        windowed_preference_by_wheel: longblob
        cumulative_preference_by_pellet: longblob
        windowed_preference_by_pellet: longblob
        cumulative_preference_by_time: longblob
        windowed_preference_by_time: longblob
        preference_score: float  # one representative preference score for the entire block
        """

    def make(self, key):
        block_patches = (BlockAnalysis.Patch & key).fetch(as_dict=True)
        block_subjects = (BlockAnalysis.Subject & key).fetch(as_dict=True)
        subject_names = [s["subject_name"] for s in block_subjects]
        # Construct subject position dataframe
        subjects_positions_df = pd.concat(
            [
                pd.DataFrame(
                    {"subject_name": [s["subject_name"]] * len(s["position_timestamps"])}
                    | {
                        k: s[k]
                        for k in (
                            "position_timestamps",
                            "position_x",
                            "position_y",
                            "position_likelihood",
                        )
                    }
                )
                for s in block_subjects
            ]
        )
        subjects_positions_df.set_index("position_timestamps", inplace=True)
        # Get frame rate of CameraTop
        camera_fps = int(
            (
                streams.SpinnakerVideoSource * streams.SpinnakerVideoSource.Attribute
                & key
                & 'attribute_name = "SamplingFrequency"'
                & 'spinnaker_video_source_name = "CameraTop"'
                & f'spinnaker_video_source_install_time < "{key["block_start"]}"'
            ).fetch("attribute_value", order_by="spinnaker_video_source_install_time DESC", limit=1)[0]
        )

        self.insert1(key)
        for i, patch in enumerate(block_patches):
            cum_wheel_dist = pd.Series(
                index=patch["wheel_timestamps"], data=patch["wheel_cumsum_distance_travelled"]
            )
            # Get distance-to-patch at each pose data timestep
            patch_region = (
                acquisition.EpochConfig.ActiveRegion
                & key
                & {"region_name": f"{patch['patch_name']}Region"}
                & f'epoch_start < "{key["block_start"]}"'
            ).fetch("region_data", order_by="epoch_start DESC", limit=1)[0]
            patch_xy = list(zip(*[(int(p["X"]), int(p["Y"])) for p in patch_region["ArrayOfPoint"]]))
            patch_center = np.mean(patch_xy[0]).astype(np.uint32), np.mean(patch_xy[1]).astype(np.uint32)
            subjects_xy = subjects_positions_df[["position_x", "position_y"]].values
            dist_to_patch = np.sqrt(np.sum((subjects_xy - patch_center) ** 2, axis=1).astype(float))
            dist_to_patch_df = subjects_positions_df[["subject_name"]].copy()
            dist_to_patch_df["dist_to_patch"] = dist_to_patch
            # Assign pellets and wheel timestamps to subjects
            if len(block_subjects) == 1:
                cum_wheel_dist_dm = cum_wheel_dist.to_frame(name=subject_names[0])
                patch_df_for_pellets_df = pd.DataFrame(
                    index=patch["pellet_timestamps"], data={"subject_name": subject_names[0]}
                )
            else:
                # Assign id based on which subject was closest to patch at time of event
                # Get distance-to-patch at each wheel ts and pel del ts, organized by subject
                dist_to_patch_wheel_ts_id_df = pd.DataFrame(
                    index=cum_wheel_dist.index, columns=subject_names
                )
                dist_to_patch_pel_ts_id_df = pd.DataFrame(
                    index=patch["pellet_timestamps"], columns=subject_names
                )
                for subject_name in subject_names:
                    # Find closest match between pose_df indices and wheel indices
                    if not dist_to_patch_wheel_ts_id_df.empty:
                        dist_to_patch_wheel_ts_subj = pd.merge_asof(
                            left=dist_to_patch_wheel_ts_id_df[subject_name],
                            right=dist_to_patch_df[dist_to_patch_df["subject_name"] == subject_name],
                            left_index=True,
                            right_index=True,
                            direction="forward",
                            tolerance=pd.Timedelta("100ms"),
                        )
                        dist_to_patch_wheel_ts_id_df[subject_name] = dist_to_patch_wheel_ts_subj[
                            "dist_to_patch"
                        ]
                    # Find closest match between pose_df indices and pel indices
                    if not dist_to_patch_pel_ts_id_df.empty:
                        dist_to_patch_pel_ts_subj = pd.merge_asof(
                            left=dist_to_patch_pel_ts_id_df[subject_name],
                            right=dist_to_patch_df[dist_to_patch_df["subject_name"] == subject_name],
                            left_index=True,
                            right_index=True,
                            direction="forward",
                            tolerance=pd.Timedelta("200ms"),
                        )
                        dist_to_patch_pel_ts_id_df[subject_name] = dist_to_patch_pel_ts_subj[
                            "dist_to_patch"
                        ]
                # Get closest subject to patch at each pel del timestep
                patch_df_for_pellets_df = pd.DataFrame(
                    index=patch["pellet_timestamps"],
                    data={"subject_name": dist_to_patch_pel_ts_id_df.idxmin(axis=1).values},
                )

                # Get closest subject to patch at each wheel timestep
                cum_wheel_dist_subj_df = pd.DataFrame(
                    index=cum_wheel_dist.index, columns=subject_names, data=0.0
                )
                closest_subjects = dist_to_patch_wheel_ts_id_df.idxmin(axis=1)
                wheel_dist = cum_wheel_dist.diff().fillna(cum_wheel_dist.iloc[0])
                # Assign wheel dist to closest subject for each wheel timestep
                for subject_name in subject_names:
                    subj_idxs = cum_wheel_dist_subj_df[closest_subjects == subject_name].index
                    cum_wheel_dist_subj_df.loc[subj_idxs, subject_name] = wheel_dist[subj_idxs]
                cum_wheel_dist_dm = cum_wheel_dist_subj_df.cumsum(axis=0)

            # In Patch Time
            patch_bbox = mpl_path.Path(list(zip(*patch_xy)))
            in_patch = subjects_positions_df.apply(
                lambda row: patch_bbox.contains_point((row["position_x"], row["position_y"])), axis=1
            )
            # Insert data
            for subject_name in subject_names:
                pellets = patch_df_for_pellets_df[patch_df_for_pellets_df["subject_name"] == subject_name]
                subject_in_patch = subjects_positions_df[
                    in_patch & (subjects_positions_df["subject_name"] == subject_name)
                ]
                self.Patch.insert1(
                    key
                    | dict(
                        patch_name=patch["patch_name"],
                        subject_name=subject_name,
                        in_patch_timestamps=subject_in_patch.index.values,
                        in_patch_time=len(subject_in_patch) / camera_fps,
                        pellet_count=len(pellets),
                        pellet_timestamps=pellets.index.values,
                        wheel_cumsum_distance_travelled=cum_wheel_dist_dm[subject_name].values,
                    )
                )


@schema
class BlockPlots(dj.Computed):
    definition = """
    -> BlockAnalysis
    ---
    subject_positions_plot: longblob
    subject_weights_plot: longblob
    patch_distance_travelled_plot: longblob
    patch_rate_plot: longblob
    cumulative_pellet_plot: longblob
    """

    def make(self, key):
        # For position data , set confidence threshold to return position values and downsample by 5x
        conf_thresh = 0.9
        downsampling_factor = 5

        # Make plotly plots
        weight_fig = go.Figure()
        pos_fig = go.Figure()
        wheel_fig = go.Figure()
        patch_rate_fig = go.Figure()
        cumulative_pellet_fig = go.Figure()

        for subject_data in BlockAnalysis.Subject & key:
            # Subject weight over time
            weight_fig.add_trace(
                go.Scatter(
                    x=subject_data["weight_timestamps"],
                    y=subject_data["weights"],
                    mode="lines",
                    name=subject_data["subject_name"],
                )
            )
            # Subject position over time
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

        # Cumulative wheel distance travelled over time
        for patch_data in BlockAnalysis.Patch & key:
            wheel_fig.add_trace(
                go.Scatter(
                    x=patch_data["wheel_timestamps"][::2],
                    y=patch_data["wheel_cumsum_distance_travelled"][::2],
                    mode="lines",
                    name=patch_data["patch_name"],
                )
            )

        # Create a bar chart for patch rates
        patch_df = (BlockAnalysis.Patch & key).fetch(format="frame").reset_index()
        patch_rate_fig = px.bar(
            patch_df,
            x="patch_name",
            y="patch_rate",
            color="patch_name",
            title="Patch Stats: Patch Rate for Each Patch",
            labels={"patch_name": "Patch Name", "patch_rate": "Patch Rate"},
            text="patch_rate",
        )
        patch_rate_fig.update_layout(bargap=0.2, width=600, height=400, template="simple_white")

        # Cumulative pellets per patch over time
        for _, row in patch_df.iterrows():
            timestamps = row["pellet_timestamps"]
            total_pellet_count = list(range(1, row["pellet_count"] + 1))

            cumulative_pellet_fig.add_trace(
                go.Scatter(x=timestamps, y=total_pellet_count, mode="lines+markers", name=row["patch_name"])
            )

        cumulative_pellet_fig.update_layout(
            title="Cumulative Pellet Count Over Time",
            xaxis_title="Time",
            yaxis_title="Cumulative Pellet Count",
            width=800,
            height=500,
            legend_title="Patch Name",
            showlegend=True,
            template="simple_white",
        )

        # Insert figures as json-formatted plotly plots
        self.insert1(
            {
                **key,
                "subject_positions_plot": json.loads(pos_fig.to_json()),
                "subject_weights_plot": json.loads(weight_fig.to_json()),
                "patch_distance_travelled_plot": json.loads(wheel_fig.to_json()),
                "patch_rate_plot": json.loads(patch_rate_fig.to_json()),
                "cumulative_pellet_plot": json.loads(cumulative_pellet_fig.to_json()),
            }
        )
