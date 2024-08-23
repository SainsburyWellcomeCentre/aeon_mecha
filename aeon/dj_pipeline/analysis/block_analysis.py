import json
import datajoint as dj
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import path as mpl_path
from datetime import datetime

from aeon.analysis import utils as analysis_utils
from aeon.dj_pipeline import acquisition, fetch_stream, get_schema_name, streams, tracking
from aeon.dj_pipeline.analysis.visit import filter_out_maintenance_periods, get_maintenance_periods

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
        # In the BlockState data - if the 0 is the first index - look back at the previous chunk
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

        # detecting block end times
        # pellet count reset - find 0s in BlockState

        block_state_query = acquisition.Environment.BlockState & exp_key & chunk_restriction
        block_state_df = fetch_stream(block_state_query)
        block_state_df.index = block_state_df.index.round(
            "us"
        )  # timestamp precision in DJ is only at microseconds
        block_state_df = block_state_df.loc[
            (block_state_df.index > previous_block_start) & (block_state_df.index <= chunk_end)
        ]

        block_ends = block_state_df[block_state_df.pellet_ct == 0]
        # account for the double 0s - find any 0s that are within 1 second of each other, remove the 2nd one
        double_0s = block_ends.index.to_series().diff().dt.total_seconds() < 1
        # find the indices of the 2nd 0s and remove
        double_0s = double_0s.shift(-1).fillna(False)
        block_ends = block_ends[~double_0s]

        block_entries = []
        for idx, block_end in enumerate(block_ends.index):
            if idx == 0:
                if previous_block_key:
                    # if there is a previous block - insert "block_end" for the previous block
                    previous_pellet_time = block_state_df[:block_end].index[-1]
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

        Block.insert(block_entries, skip_duplicates=True)
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
            streams.UndergroundFeederDeliverPellet,
            streams.UndergroundFeederEncoder,
            tracking.SLEAPTracking,
        )
        for streams_table in streams_tables:
            if len(streams_table & chunk_keys) < len(streams_table.key_source & chunk_keys):
                raise ValueError(
                    f"BlockAnalysis Not Ready - {streams_table.__name__} not yet fully ingested for block: {key}. Skipping (to retry later)..."
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
            # pellet delivery and patch threshold data
            depletion_state_df = fetch_stream(
                streams.UndergroundFeederDepletionState & patch_key & chunk_restriction
            )[block_start:block_end]

            pellet_ts_threshold_df = get_threshold_associated_pellets(patch_key, block_start, block_end)

            # wheel encoder data
            encoder_df = fetch_stream(streams.UndergroundFeederEncoder & patch_key & chunk_restriction)[
                block_start:block_end
            ]
            # filter out maintenance period based on logs
            pellet_ts_threshold_df = filter_out_maintenance_periods(
                pellet_ts_threshold_df,
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

            if depletion_state_df.empty:
                raise ValueError(f"No depletion state data found for block {key} - patch: {patch_name}")

            if len(depletion_state_df.rate.unique()) > 1:
                # multiple patch rates per block is unexpected, log a note and pick the first rate to move forward
                AnalysisNote.insert1(
                    {
                        "note_timestamp": datetime.utcnow(),
                        "note_type": "Multiple patch rates",
                        "note": f"Found multiple patch rates for block {key} - patch: {patch_name} - rates: {depletion_state_df.rate.unique()}",
                    }
                )

            patch_rate = depletion_state_df.rate.iloc[0]
            patch_offset = depletion_state_df.offset.iloc[0]
            # handles patch rate value being INF
            patch_rate = 999999999 if np.isinf(patch_rate) else patch_rate

            self.Patch.insert1(
                {
                    **key,
                    "patch_name": patch_name,
                    "pellet_count": len(pellet_ts_threshold_df),
                    "pellet_timestamps": pellet_ts_threshold_df.pellet_timestamp.values,
                    "wheel_cumsum_distance_travelled": encoder_df.distance_travelled.values[
                        ::wheel_downsampling_factor
                    ],
                    "wheel_timestamps": encoder_df.index.values[::wheel_downsampling_factor],
                    "patch_threshold": pellet_ts_threshold_df.threshold.values,
                    "patch_threshold_timestamps": pellet_ts_threshold_df.index.values,
                    "patch_rate": patch_rate,
                    "patch_offset": patch_offset,
                }
            )

            # update block_end if last timestamp of encoder_df is before the current block_end
            if encoder_df.index[-1] < block_end:
                block_end = encoder_df.index[-1]

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
            if _df.type.iloc[-1] != "Exit":
                subject_names.append(subject_name)

        for subject_name in subject_names:
            # positions - query for CameraTop, identity_name matches subject_name,
            pos_query = (
                streams.SpinnakerVideoSource
                * tracking.SLEAPTracking.PoseIdentity.proj("identity_name", part_name="anchor_part")
                * tracking.SLEAPTracking.Part
                & key
                & {
                    "spinnaker_video_source_name": "CameraTop",
                    "identity_name": subject_name,
                }
                & chunk_restriction
            )
            pos_df = fetch_stream(pos_query)[block_start:block_end]
            pos_df = filter_out_maintenance_periods(pos_df, maintenance_period, block_end)

            if pos_df.empty:
                continue

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

            # update block_end if last timestamp of pos_df is before the current block_end
            if pos_df.index[-1] < block_end:
                block_end = pos_df.index[-1]

        if block_end != (Block & key).fetch1("block_end"):
            Block.update1({**key, "block_end": block_end})
            self.update1({**key, "block_duration": (block_end - block_start).total_seconds() / 3600})


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
        patch_threshold: longblob  # patch threshold value at each pellet delivery
        wheel_cumsum_distance_travelled: longblob  # wheel's cumulative distance travelled
        """

    class Preference(dj.Part):
        definition = """ # Measure of preference for a particular patch from a particular subject
        -> master
        -> BlockAnalysis.Patch
        -> BlockAnalysis.Subject
        ---
        cumulative_preference_by_wheel: longblob
        cumulative_preference_by_time: longblob
        final_preference_by_wheel=null: float  # cumulative_preference_by_wheel at the end of the block
        final_preference_by_time=null: float  # cumulative_preference_by_time at the end of the block
        """

    key_source = BlockAnalysis & BlockAnalysis.Patch & BlockAnalysis.Subject

    def make(self, key):
        block_patches = (BlockAnalysis.Patch & key).fetch(as_dict=True)
        block_subjects = (BlockAnalysis.Subject & key).fetch(as_dict=True)
        subject_names = [s["subject_name"] for s in block_subjects]
        patch_names = [p["patch_name"] for p in block_patches]
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

        self.insert1(key)

        in_patch_radius = 130  # pixels
        pref_attrs = ["cum_dist", "cum_time", "cum_pref_dist", "cum_pref_time"]
        all_subj_patch_pref_dict = {
            p: {s: {a: pd.Series() for a in pref_attrs} for s in subject_names} for p in patch_names
        }

        for patch in block_patches:
            cum_wheel_dist = pd.Series(
                index=patch["wheel_timestamps"], data=patch["wheel_cumsum_distance_travelled"]
            )
            # Assign pellets and wheel timestamps to subjects
            # Assign id based on which subject was closest to patch at time of event
            # Get distance-to-patch at each wheel ts and pel del ts, organized by subject
            # Get patch x,y from metadata patch rfid loc
            patch_center = (
                streams.RfidReader.join(streams.RfidReader.RemovalTime, left=True)
                * streams.RfidReader.Attribute
                & key
                & f"'{key['block_start']}' >= rfid_reader_install_time"
                & f"'{key['block_start']}' < IFNULL(rfid_reader_removal_time, '2200-01-01')"
                & f"rfid_reader_name LIKE '%{patch['patch_name']}%'"
                & "attribute_name = 'Location'"
            ).fetch1("attribute_value")
            patch_center = (int(patch_center["X"]), int(patch_center["Y"]))
            subjects_xy = subjects_positions_df[["position_x", "position_y"]].values
            dist_to_patch = np.sqrt(np.sum((subjects_xy - patch_center) ** 2, axis=1).astype(float))
            dist_to_patch_df = subjects_positions_df[["subject_name"]].copy()
            dist_to_patch_df["dist_to_patch"] = dist_to_patch

            dist_to_patch_wheel_ts_id_df = pd.DataFrame(index=cum_wheel_dist.index, columns=subject_names)
            dist_to_patch_pel_ts_id_df = pd.DataFrame(
                index=patch["pellet_timestamps"], columns=subject_names
            )
            for subject_name in subject_names:
                # Find closest match between pose_df indices and wheel indices
                if not dist_to_patch_wheel_ts_id_df.empty:
                    dist_to_patch_wheel_ts_subj = pd.merge_asof(
                        left=pd.DataFrame(dist_to_patch_wheel_ts_id_df[subject_name].copy()).reset_index(
                            names="time"
                        ),
                        right=dist_to_patch_df[dist_to_patch_df["subject_name"] == subject_name]
                        .copy()
                        .reset_index(names="time"),
                        on="time",
                        # left_index=True,
                        # right_index=True,
                        direction="nearest",
                        tolerance=pd.Timedelta("100ms"),
                    )
                    dist_to_patch_wheel_ts_id_df[subject_name] = dist_to_patch_wheel_ts_subj[
                        "dist_to_patch"
                    ].values
                # Find closest match between pose_df indices and pel indices
                if not dist_to_patch_pel_ts_id_df.empty:
                    dist_to_patch_pel_ts_subj = pd.merge_asof(
                        left=pd.DataFrame(dist_to_patch_pel_ts_id_df[subject_name].copy()).reset_index(
                            names="time"
                        ),
                        right=dist_to_patch_df[dist_to_patch_df["subject_name"] == subject_name]
                        .copy()
                        .reset_index(names="time"),
                        on="time",
                        # left_index=True,
                        # right_index=True,
                        direction="nearest",
                        tolerance=pd.Timedelta("200ms"),
                    )
                    dist_to_patch_pel_ts_id_df[subject_name] = dist_to_patch_pel_ts_subj[
                        "dist_to_patch"
                    ].values

            # Get closest subject to patch at each pellet timestep
            closest_subjects_pellet_ts = dist_to_patch_pel_ts_id_df.idxmin(axis=1)
            # Get closest subject to patch at each wheel timestep
            cum_wheel_dist_subj_df = pd.DataFrame(
                index=cum_wheel_dist.index, columns=subject_names, data=0.0
            )
            closest_subjects_wheel_ts = dist_to_patch_wheel_ts_id_df.idxmin(axis=1)
            wheel_dist = cum_wheel_dist.diff().fillna(cum_wheel_dist.iloc[0])
            # Assign wheel dist to closest subject for each wheel timestep
            for subject_name in subject_names:
                subj_idxs = cum_wheel_dist_subj_df[closest_subjects_wheel_ts == subject_name].index
                cum_wheel_dist_subj_df.loc[subj_idxs, subject_name] = wheel_dist[subj_idxs]
            cum_wheel_dist_subj_df = cum_wheel_dist_subj_df.cumsum(axis=0)

            # In patch time
            in_patch = dist_to_patch_wheel_ts_id_df < in_patch_radius
            dt = np.median(np.diff(cum_wheel_dist.index)).astype(int) / 1e9  # s
            # Fill in `all_subj_patch_pref`
            for subject_name in subject_names:
                all_subj_patch_pref_dict[patch["patch_name"]][subject_name]["cum_dist"] = (
                    cum_wheel_dist_subj_df[subject_name].values
                )
                subject_in_patch = in_patch[subject_name]
                subject_in_patch_cum_time = subject_in_patch.cumsum().values * dt
                all_subj_patch_pref_dict[patch["patch_name"]][subject_name][
                    "cum_time"
                ] = subject_in_patch_cum_time

                closest_subj_mask = closest_subjects_pellet_ts == subject_name
                subj_pellets = closest_subjects_pellet_ts[closest_subj_mask]
                subj_patch_thresh = patch["patch_threshold"][closest_subj_mask]

                self.Patch.insert1(
                    key
                    | dict(
                        patch_name=patch["patch_name"],
                        subject_name=subject_name,
                        in_patch_timestamps=subject_in_patch.index.values,
                        in_patch_time=subject_in_patch_cum_time[-1],
                        pellet_count=len(subj_pellets),
                        pellet_timestamps=subj_pellets.index.values,
                        patch_threshold=subj_patch_thresh,
                        wheel_cumsum_distance_travelled=cum_wheel_dist_subj_df[subject_name].values,
                    )
                )

        # Now that we have computed all individual patch and subject values, we iterate again through
        # patches and subjects to compute preference scores
        for subject_name in subject_names:
            # Get sum of subj cum wheel dists and cum in patch time
            all_cum_dist = np.sum(
                [all_subj_patch_pref_dict[p][subject_name]["cum_dist"][-1] for p in patch_names]
            )
            all_cum_time = np.sum(
                [all_subj_patch_pref_dict[p][subject_name]["cum_time"][-1] for p in patch_names]
            )

            for patch_name in patch_names:
                cum_pref_dist = (
                    all_subj_patch_pref_dict[patch_name][subject_name]["cum_dist"] / all_cum_dist
                )
                all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_dist"] = cum_pref_dist

                cum_pref_time = (
                    all_subj_patch_pref_dict[patch_name][subject_name]["cum_time"] / all_cum_time
                )
                all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_time"] = cum_pref_time

                self.Preference.insert1(
                    key
                    | dict(
                        patch_name=patch_name,
                        subject_name=subject_name,
                        cumulative_preference_by_time=cum_pref_time,
                        cumulative_preference_by_wheel=cum_pref_dist,
                        final_preference_by_time=cum_pref_time[-1],
                        final_preference_by_wheel=cum_pref_dist[-1],
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


@schema
class BlockSubjectPlots(dj.Computed):
    definition = """
    -> BlockSubjectAnalysis
    ---
    dist_pref_plot: longblob  # Cumulative Patch Preference by Wheel Distance - per subject per patch
    time_pref_plot: longblob  # Cumulative Patch Preference by Time - per subject per patch
    """

    def make(self, key):
        from aeon.analysis.block_plotting import (
            subject_colors,
            patch_markers_linestyles,
            patch_markers,
            gen_hex_grad,
        )

        patch_names, subject_names = (BlockSubjectAnalysis.Preference & key).fetch(
            "patch_name", "subject_name"
        )
        patch_names = np.unique(patch_names)
        subject_names = np.unique(subject_names)

        all_thresh_vals = np.concatenate((BlockAnalysis.Patch & key).fetch("patch_threshold")).astype(float)

        dist_pref_fig, time_pref_fig = go.Figure(), go.Figure()
        for subj_i, subj in enumerate(subject_names):
            for patch_i, p in enumerate(patch_names):
                rate, offset, wheel_ts = (BlockAnalysis.Patch & key & {"patch_name": p}).fetch1(
                    "patch_rate", "patch_offset", "wheel_timestamps"
                )
                patch_thresh, patch_thresh_ts = (BlockAnalysis.Patch & key & {"patch_name": p}).fetch1(
                    "patch_threshold", "patch_threshold_timestamps"
                )

                cum_pref_dist, cum_pref_time = (
                    BlockSubjectAnalysis.Preference & key & {"patch_name": p, "subject_name": subj}
                ).fetch1("cumulative_preference_by_wheel", "cumulative_preference_by_time")
                pellet_ts = (
                    BlockSubjectAnalysis.Patch & key & {"patch_name": p, "subject_name": subj}
                ).fetch1("pellet_timestamps")

                if not len(pellet_ts):
                    continue

                patch_thresh = patch_thresh[np.searchsorted(patch_thresh_ts, pellet_ts) - 1]
                patch_mean = 1 / rate // 100 * 100
                patch_mean_thresh = patch_mean + offset
                cum_pel_ct = pd.DataFrame(
                    index=pellet_ts,
                    data={
                        "counter": np.arange(1, len(pellet_ts) + 1),
                        "threshold": patch_thresh.astype(float),
                        "mean_thresh": patch_mean_thresh,
                        "patch_label": f"{p} μ: {patch_mean_thresh}",
                    },
                )
                cum_pel_ct["norm_thresh_val"] = (
                    (cum_pel_ct["threshold"] - all_thresh_vals.min())
                    / (all_thresh_vals.max() - all_thresh_vals.min())
                ).round(3)

                for fig, cum_pref in zip([dist_pref_fig, time_pref_fig], [cum_pref_dist, cum_pref_time]):
                    fig.add_trace(
                        go.Scatter(
                            x=wheel_ts,
                            y=cum_pref,
                            mode="lines",  # +  markers",
                            line=dict(
                                width=2,
                                color=subject_colors[subj_i],
                                dash=patch_markers_linestyles[patch_i],
                            ),
                            name=f"{subj} - {p}: μ: {patch_mean}",
                        )
                    )
                    # Add markers for each pellet
                    cur_cum_pel_ct = pd.merge_asof(
                        cum_pel_ct.reset_index(names="time"),
                        pd.DataFrame(index=wheel_ts, data={"cum_pref": cum_pref}).reset_index(names="time"),
                        on="time",
                        direction="forward",
                        tolerance=pd.Timedelta("0.1s"),
                    )
                    if not cur_cum_pel_ct.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=cur_cum_pel_ct["time"],
                                y=cur_cum_pel_ct["cum_pref"],
                                mode="markers",
                                marker=dict(
                                    symbol=patch_markers[patch_i],
                                    color=gen_hex_grad(
                                        subject_colors[-1], cur_cum_pel_ct["norm_thresh_val"]
                                    ),
                                    size=8,
                                ),
                                showlegend=False,
                                customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                                hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                            )
                        )

        for fig, title in zip([dist_pref_fig, time_pref_fig], ["Wheel Distance", "Patch Time"]):
            fig.update_layout(
                title=f"Cumulative Patch Preference - {title}",
                xaxis_title="Time",
                yaxis_title="Pref Index",
                yaxis=dict(tickvals=np.arange(0, 1.1, 0.1)),
            )

        # Insert figures as json-formatted plotly plots
        self.insert1(
            {
                **key,
                "dist_pref_plot": json.loads(dist_pref_fig.to_json()),
                "time_pref_plot": json.loads(time_pref_fig.to_json()),
            }
        )


# ---- AnalysisNote ----


@schema
class AnalysisNote(dj.Manual):
    definition = """  # Generic table to catch all notes generated during analysis
    note_timestamp: datetime
    ---
    note_type='': varchar(64)
    note: varchar(3000)
    """

# ---- Helper Functions ----


def get_threshold_associated_pellets(patch_key, start, end):
    """
    Retrieve the pellet delivery timestamps associated with each patch threshold update within the specified start-end time.
    1. Get all patch state update timestamps: let's call these events "A"
    2. Remove all "A" events near manual pellet delivery events (so we don't include manual pellet delivery events in downstream analysis)
    3. For the remaining "A" events, find the nearest delivery event within 1s: for this delivery event, check if there are any repeat delivery events within 0.5 seconds - take the last of these as the pellet delivery timestamp (discard all "A" events that don't have such a corresponding delivery event)
    4. Now for these 'clean' "A" events, go back in time to the SECOND preceding pellet threshold value: this is the threshold value for this pellet delivery (as seen in this image we discussed before)
    """
    chunk_restriction = acquisition.create_chunk_restriction(
        patch_key["experiment_name"], start, end
    )
    # pellet delivery and patch threshold data
    delivered_pellet_df = fetch_stream(
        streams.UndergroundFeederDeliverPellet & patch_key & chunk_restriction
    )[start:end]
    depletion_state_df = fetch_stream(
        streams.UndergroundFeederDepletionState & patch_key & chunk_restriction
    )[start:end]
    # remove NaNs from threshold column
    depletion_state_df = depletion_state_df.dropna(subset=["threshold"])
    # identify & remove invalid indices where the time difference is less than 1 second
    invalid_indices = np.where(depletion_state_df.index.to_series().diff().dt.total_seconds() < 1)[0]
    depletion_state_df = depletion_state_df.drop(depletion_state_df.index[invalid_indices])

    # find pellet times approximately coincide with each threshold update
    # i.e. nearest pellet delivery within 100ms before or after threshold update
    delivered_pellet_ts = delivered_pellet_df.index
    pellet_ts_threshold_df = depletion_state_df.copy()
    pellet_ts_threshold_df["pellet_timestamp"] = pd.NaT
    for threshold_idx in range(len(pellet_ts_threshold_df)):
        threshold_time = pellet_ts_threshold_df.index[threshold_idx]
        within_range_pellet_ts = np.logical_and(delivered_pellet_ts >= threshold_time - pd.Timedelta(milliseconds=100),
                                                delivered_pellet_ts <= threshold_time + pd.Timedelta(milliseconds=100))
        if not within_range_pellet_ts.any():
            continue
        pellet_time = delivered_pellet_ts[within_range_pellet_ts][-1]
        pellet_ts_threshold_df.pellet_timestamp.iloc[threshold_idx] = pellet_time

    # remove rows of threshold updates without corresponding pellet times from i.e. pellet_timestamp is NaN
    pellet_ts_threshold_df = pellet_ts_threshold_df.dropna(subset=["pellet_timestamp"])
    # shift back the pellet_timestamp values by 1 to match the pellet_timestamp with the previous threshold update
    pellet_ts_threshold_df.pellet_timestamp = pellet_ts_threshold_df.pellet_timestamp.shift(-1)
    # remove NaNs from pellet_timestamp column (last row)
    pellet_ts_threshold_df = pellet_ts_threshold_df.dropna(subset=["pellet_timestamp"])

    return pellet_ts_threshold_df
