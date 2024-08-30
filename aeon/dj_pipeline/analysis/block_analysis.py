import json
import datajoint as dj
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import path as mpl_path
from datetime import datetime

from aeon.analysis import utils as analysis_utils
from aeon.analysis.block_plotting import gen_hex_grad, conv2d
from aeon.dj_pipeline import acquisition, fetch_stream, get_schema_name, streams, tracking
from aeon.dj_pipeline.analysis.visit import filter_out_maintenance_periods, get_maintenance_periods

schema = dj.schema(get_schema_name("block_analysis"))
logger = dj.logger

subject_colors = plotly.colors.qualitative.Plotly
subject_colors_dict = {  # @NOTE: really, we shouldn't have to assign each subject to a color explicitly
    "BAA-1104516": subject_colors[0],
    "BAA-1104519": subject_colors[1],
    "BAA-1104568": subject_colors[2],
    "BAA-1104569": subject_colors[3],
}
patch_colors = plotly.colors.qualitative.Dark2
patch_markers = [
    "circle",
    "bowtie",
    "square",
    "hourglass",
    "diamond",
    "cross",
    "x",
    "triangle",
    "star",
]
patch_markers_symbols = ["●", "⧓", "■", "⧗", "♦", "✖", "×", "▲", "★"]
patch_markers_dict = {marker: symbol for marker, symbol in zip(patch_markers, patch_markers_symbols)}
patch_markers_linestyles = ["solid", "dash", "dot", "dashdot", "longdashdot"]


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
            streams.UndergroundFeederBeamBreak,
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
            beam_break_df = fetch_stream(
                streams.UndergroundFeederBeamBreak & patch_key & chunk_restriction
            )[block_start:block_end]
            depletion_state_df = fetch_stream(
                streams.UndergroundFeederDepletionState & patch_key & chunk_restriction
            )[block_start:block_end]
            # remove NaNs from threshold column
            depletion_state_df = depletion_state_df.dropna(subset=["threshold"])
            # identify & remove invalid indices where the time difference is less than 1 second
            invalid_indices = np.where(depletion_state_df.index.to_series().diff().dt.total_seconds() < 1)[0]
            depletion_state_df = depletion_state_df.drop(depletion_state_df.index[invalid_indices])

            # find pellet times associated with each threshold update
            #   for each threshold, find the time of the next threshold update,
            #   find the closest beam break after this update time,
            #   and use this beam break time as the delivery time for the initial threshold
            pellet_ts_threshold_df = depletion_state_df.copy()
            pellet_ts_threshold_df["pellet_timestamp"] = pd.NaT
            for threshold_idx in range(len(pellet_ts_threshold_df) - 1):
                if np.isnan(pellet_ts_threshold_df.threshold.iloc[threshold_idx]):
                    continue
                next_threshold_time = pellet_ts_threshold_df.index[threshold_idx + 1]
                post_thresh_pellet_ts = beam_break_df.index[beam_break_df.index > next_threshold_time]
                if post_thresh_pellet_ts.empty:
                    break
                next_beam_break = post_thresh_pellet_ts[np.searchsorted(post_thresh_pellet_ts, next_threshold_time)]
                pellet_ts_threshold_df.pellet_timestamp.iloc[threshold_idx] = next_beam_break
            # remove NaNs from pellet_timestamp column (last row)
            pellet_ts_threshold_df = pellet_ts_threshold_df.dropna(subset=["pellet_timestamp"])

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
                * tracking.SLEAPTracking.PoseIdentity.proj("identity_name", anchor_part="part_name")
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

        # Constant values for In Patch Time
        in_patch_radius = 130  # pixels

        # Fetch data
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
class BlockPlotsNew(dj.Computed):
    definition = """
     -> BlockAnalysis
    ---
    patch_stats_plot: longblob
    weights_block_plot: longblob
    poisition_block_plot: longblob
    position_heatmap_plot: longblob
    """

    def make(self, key):

        # Constant values for (5.) Position ethograms
        patch_radius = 120  # in px
        gate_radius = 30  # in px

        block_start = (Block & key & f"block_start >= '{one_day_into_social}'").fetch(
            "block_start", limit=1
        )[0]
        block_end = (Block & key & f"block_start >= '{one_day_into_social}'").fetch("block_end", limit=1)[0]
        key = (
            key
            | {"block_start": str(pd.Timestamp(block_start))}
            | {"block_end": str(pd.Timestamp(block_end))}
        )
        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], block_start, block_end
        )

        # Create dataframe for plotting patch stats
        subj_patch_info = (BlockSubjectAnalysis.Patch & key).fetch(format="frame")
        patch_info = (BlockAnalysis.Patch & key).fetch(
            "patch_name", "patch_rate", "patch_offset", as_dict=True
        )
        patch_names = list(subj_patch_info.index.get_level_values("patch_name").unique())
        subject_names = list(subj_patch_info.index.get_level_values("subject_name").unique())

        # Convert `subj_patch_info` into a form amenable to plotting
        reset_subj_patch_info = subj_patch_info.reset_index()  # reset to turn MultiIndex into columns
        min_subj_patch_info = reset_subj_patch_info[  # select only relevant columns
            ["patch_name", "subject_name", "pellet_timestamps", "patch_threshold"]
        ]
        min_subj_patch_info = (
            min_subj_patch_info.explode(["pellet_timestamps", "patch_threshold"], ignore_index=True)
            .dropna()
            .reset_index(drop=True)
        )
        # Rename and reindex columns
        min_subj_patch_info.columns = ["patch", "subject", "time", "threshold"]
        min_subj_patch_info = min_subj_patch_info.reindex(columns=["time", "patch", "threshold", "subject"])

        # Add patch mean values and block-normalized delivery times to pellet info
        n_patches = len(patch_info)
        patch_mean_info = pd.DataFrame(index=np.arange(n_patches), columns=min_subj_patch_info.columns)
        patch_mean_info["subject"] = "mean"
        patch_mean_info["patch"] = [d["patch_name"] for d in patch_info]
        patch_mean_info["threshold"] = [((1 / d["patch_rate"]) + d["patch_offset"]) for d in patch_info]
        patch_mean_info["time"] = subj_patch_info.index.get_level_values("block_start")[0]

        min_subj_patch_info_plus = pd.concat((patch_mean_info, min_subj_patch_info)).reset_index(drop=True)
        min_subj_patch_info_plus["norm_time"] = (
            (min_subj_patch_info_plus["time"] - min_subj_patch_info_plus["time"].iloc[0])
            / (min_subj_patch_info_plus["time"].iloc[-1] - min_subj_patch_info_plus["time"].iloc[0])
        ).round(3)

        ## Create cumulative pellet count by subject
        cum_pel_ct = min_subj_patch_info_plus.sort_values("time").copy().reset_index(drop=True)

        def cumsum_helper(group):
            group["counter"] = np.arange(len(group)) + 1
            return group

        patch_means = cum_pel_ct.loc[0:3][["patch", "threshold"]].rename(
            columns={"threshold": "mean_thresh"}
        )
        patch_means["mean_thresh"] = patch_means["mean_thresh"].astype(float).round(1)
        cum_pel_ct = cum_pel_ct.merge(patch_means, on="patch", how="left")

        cum_pel_ct = cum_pel_ct[~cum_pel_ct["subject"].str.contains("mean")].reset_index(drop=True)
        cum_pel_ct = (
            cum_pel_ct.groupby("subject", group_keys=False).apply(cumsum_helper).reset_index(drop=True)
        )

        make_float_cols = ["threshold", "mean_thresh", "norm_time"]
        cum_pel_ct[make_float_cols] = cum_pel_ct[make_float_cols].astype(float)

        cum_pel_ct["patch_label"] = (
            cum_pel_ct["patch"] + " μ: " + cum_pel_ct["mean_thresh"].astype(float).round(1).astype(str)
        )

        cum_pel_ct["norm_thresh_val"] = (
            (cum_pel_ct["threshold"] - cum_pel_ct["threshold"].min())
            / (cum_pel_ct["threshold"].max() - cum_pel_ct["threshold"].min())
        ).round(3)

        # Sort by 'time' col
        cum_pel_ct = cum_pel_ct.sort_values("time")

        ## Get wheel timestamps for each patch
        wheel_ts = (BlockAnalysis.Patch() & key).fetch("patch_name", "wheel_timestamps", as_dict=True)
        wheel_ts = {d["patch_name"]: d["wheel_timestamps"] for d in wheel_ts}

        ## Get subject patch data
        subject_patch_data = (BlockSubjectAnalysis.Patch() & key).fetch(format="frame")
        subject_patch_data.reset_index(level=["experiment_name", "block_start"], drop=True, inplace=True)

        ## 2. Get subject weights in block
        weights_block = fetch_stream(acquisition.Environment.SubjectWeight & key)

        ## 3. Animal position in block
        pose_query = (
            streams.SpinnakerVideoSource
            * tracking.SLEAPTracking.PoseIdentity.proj(
                "identity_name", "identity_likelihood", anchor_part="part_name"
            )
            * tracking.SLEAPTracking.Part
            & {"spinnaker_video_source_name": "CameraTop"}
            & key
            & chunk_restriction
        )
        centroid_df = fetch_stream(pose_query)[block_start:block_end]
        centroid_df = (
            centroid_df.groupby("identity_name")
            .resample("100ms")
            .first()
            .droplevel("identity_name")
            .dropna()
            .sort_index()
        )
        centroid_df.drop(columns=["spinnaker_video_source_name"], inplace=True)
        centroid_df["x"], centroid_df["y"] = (
            centroid_df["x"].astype(np.int32),
            centroid_df["y"].astype(np.int32),
        )

        # 4. Position heatmaps per subject
        max_x, max_y = int(centroid_df["x"].max()), int(centroid_df["y"].max())
        heatmaps = []
        for id_i, (id_val, id_grp) in enumerate(centroid_df.groupby("identity_name")):
            # <s Add counts of x,y points to a grid that will be used for heatmap
            img_grid = np.zeros((max_x + 1, max_y + 1))
            points, counts = np.unique(id_grp[["x", "y"]].values, return_counts=True, axis=0)
            for point, count in zip(points, counts):
                img_grid[point[0], point[1]] = count
            img_grid /= img_grid.max()  # normalize
            # /s>
            # <s Smooth `img_grid`
            # Mice can go ~450 cm/s, we've downsampled to 10 frames/s, we have 200 px / 1000 cm,
            # so 45 cm/frame ~= 9 px/frame
            win_sz = 9  # in pixels  (ensure odd for centering)
            kernel = np.ones((win_sz, win_sz)) / win_sz**2  # moving avg kernel
            img_grid_p = np.pad(img_grid, win_sz // 2, mode="edge")  # pad for full output from convolution
            img_grid_smooth = conv2d(img_grid_p, kernel)
            heatmaps.append((id_val, img_grid_smooth))

        # 5. Position ethogram per subject
        rois_info = (
            acquisition.EpochConfig.ActiveRegion()
            & key
            & {"epoch_start": pd.Timestamp("2024-06-25 10:01:46")}
        ).fetch(as_dict=True)
        roi_locs = {item["region_name"]: item["region_data"] for item in rois_info}
        rfid_info = (
            streams.RfidReader() * streams.RfidReader.Attribute() & key & "attribute_name = 'Location'"
        ).fetch(as_dict=True)
        rfid_locs = {item["rfid_reader_name"]: item["attribute_value"] for item in rfid_info}

        ## Create position ethogram df
        arena_center_x = int(roi_locs["ArenaCenter"]["X"])
        arena_center_y = int(roi_locs["ArenaCenter"]["Y"])
        arena_center = (arena_center_x, arena_center_y)
        arena_inner_radius = int(roi_locs["ArenaInnerRadius"])
        arena_outer_radius = int(roi_locs["ArenaOuterRadius"])

        rfid_names = (streams.RfidReader() * streams.RfidReader.Attribute()).fetch(
            "rfid_reader_name", as_dict=True
        )
        rfid_names = np.unique([item["rfid_reader_name"] for item in rfid_names])

        rois = patch_names + ["Nest", "Gate", "Corridor"]  # ROIs: patches, nest, gate, corridor
        roi_colors = plotly.colors.qualitative.Dark2
        roi_colors_dict = {roi: roi_c for (roi, roi_c) in zip(rois, roi_colors)}
        pos_eth_df = pd.DataFrame(
            columns=(["Subject"] + rois), index=centroid_df.index
        )  # df to create eth fig
        pos_eth_df["Subject"] = centroid_df["identity_name"]

        # For each ROI, compute if within ROI
        for roi in rois:
            if roi == "Corridor":  # special case for corridor, based on between inner and outer radius
                dist = np.linalg.norm(
                    (np.vstack((centroid_df["x"], centroid_df["y"])).T) - arena_center, axis=1
                )
                pos_eth_df[roi] = (dist >= arena_inner_radius) & (dist <= arena_outer_radius)
            elif roi == "Nest":  # special case for nest, based on 4 corners
                nest_corners = roi_locs["NestRegion"]["ArrayOfPoint"]
                nest_br_x, nest_br_y = int(nest_corners[0]["X"]), int(nest_corners[0]["Y"])
                nest_bl_x, nest_bl_y = int(nest_corners[1]["X"]), int(nest_corners[1]["Y"])
                nest_tl_x, nest_tl_y = int(nest_corners[2]["X"]), int(nest_corners[2]["Y"])
                nest_tr_x, nest_tr_y = int(nest_corners[3]["X"]), int(nest_corners[3]["Y"])
                pos_eth_df[roi] = (
                    (centroid_df["x"] <= nest_br_x)
                    & (centroid_df["y"] >= nest_br_y)
                    & (centroid_df["x"] >= nest_bl_x)
                    & (centroid_df["y"] >= nest_bl_y)
                    & (centroid_df["x"] >= nest_tl_x)
                    & (centroid_df["y"] <= nest_tl_y)
                    & (centroid_df["x"] <= nest_tr_x)
                    & (centroid_df["y"] <= nest_tr_y)
                )
            else:
                roi_radius = gate_radius if roi == "Gate" else patch_radius
                # Get ROI coords
                roi_x, roi_y = int(rfid_locs[roi + "Rfid"]["X"]), int(rfid_locs[roi + "Rfid"]["Y"])
                # Check if in ROI
                dist = np.linalg.norm(
                    (np.vstack((centroid_df["x"], centroid_df["y"])).T) - (roi_x, roi_y), axis=1
                )
                pos_eth_df[roi] = dist < roi_radius

        # 11. Running preference by wheel and time
        ## Get subject patch preference data
        patch_pref = (BlockSubjectAnalysis.Preference() & key).fetch(format="frame")
        patch_pref.reset_index(level=["experiment_name", "block_start"], drop=True, inplace=True)
        # Replace small vals with 0
        patch_pref["cumulative_preference_by_wheel"] = patch_pref["cumulative_preference_by_wheel"].apply(
            lambda arr: np.where(np.array(arr) < 1e-3, 0, np.array(arr))
        )
        ## Calculate running preference by wheel and time
        # @NOTE: Store the running preference (as calculated here) in the db as well

        def calculate_running_preference(group, pref_col, out_col):
            total_pref = np.sum(np.vstack(group[pref_col].values), axis=0)  # sum pref at each ts
            group[out_col] = group[pref_col].apply(
                lambda x: np.nan_to_num(x / total_pref, 0.0)
            )  # running pref
            return group

        patch_pref = (
            patch_pref.groupby("subject_name")
            .apply(
                lambda group: calculate_running_preference(
                    group, "cumulative_preference_by_wheel", "running_preference_by_wheel"
                )
            )
            .droplevel(0)
        )

        patch_pref = (
            patch_pref.groupby("subject_name")
            .apply(
                lambda group: calculate_running_preference(
                    group, "cumulative_preference_by_time", "running_preference_by_time"
                )
            )
            .droplevel(0)
        )

        # 12. Patch preference weighted by wheel-distance-spun-TO-pellet-count ratio
        ## Create multi-indexed dataframe with weighted distance for each subject-patch pair.
        pel_patches = [p for p in patch_names if not "dummy" in p.lower()]
        data = []
        for patch in pel_patches:
            for subject in subject_names:
                data.append(
                    {
                        "patch_name": patch,
                        "subject_name": subject,
                        "time": wheel_ts[patch],
                        "weighted_dist": np.empty_like(wheel_ts[patch]),
                    }
                )
        subj_wheel_pel_weighted_dist = pd.DataFrame(data)
        subj_wheel_pel_weighted_dist.set_index(["patch_name", "subject_name"], inplace=True)
        subj_wheel_pel_weighted_dist["weighted_dist"] = np.nan

        ## Calculate weighted distance

        ### Plotting
        # 1. Plot patch stats from dataframe of each pellet threshold per patch
        # block_subjects = (BlockAnalysis.Subject & key).fetch(as_dict=True)
        # subject_names = [s["subject_name"] for s in block_subjects]
        box_colors = ["#0A0A0A"] + subject_colors[0 : len(subject_names)]  # subject colors + mean color

        patch_stats_fig = px.box(
            min_subj_patch_info_plus.sort_values("patch"),
            x="patch",
            y="threshold",
            color="subject",
            hover_data=["norm_time"],
            color_discrete_sequence=box_colors,
            # notched=True,
            points="all",
        )

        patch_stats_fig.update_layout(
            title="Patch Stats: Patch Means and Sampled Threshold Values",
            xaxis_title="Patch",
            yaxis_title="Threshold (cm)",
        )

        # 2. Plot animal weights in block
        weights_block_fig = px.line(
            weights_block,
            x=weights_block.index,
            y="weight",
            color="subject_id",
            color_discrete_map=subject_colors_dict,
            markers=True,
        )

        weights_block_fig.update_traces(line=dict(width=3), marker=dict(size=8))

        weights_block_fig.update_layout(
            title="Weights in block",
            xaxis_title="Time",
            yaxis_title="Weight (g)",
        )

        # 3. Plot position (centroid) over time
        position_block_fig = go.Figure()
        for id_i, (id_val, id_grp) in enumerate(centroid_df.groupby("identity_name")):
            norm_time = (
                (id_grp.index - id_grp.index[0]) / (id_grp.index[-1] - id_grp.index[0])
            ).values.round(3)
            colors = gen_hex_grad(subject_colors[id_i], norm_time)
            position_block_fig.add_trace(
                go.Scatter(
                    x=id_grp["x"],
                    y=id_grp["y"],
                    mode="markers",
                    name=id_val,
                    marker={
                        # "opacity": norm_time,
                        "color": colors,
                        "size": 4,
                    },
                )
            )
        position_block_fig.update_layout(
            title="Position Tracking over Time",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        # 4. Plot position heatmaps per subject
        for id_val, img_grid_smooth in heatmaps:
            pos_heatmap_fig = px.imshow(
                img_grid_smooth.T,
                zmin=0,
                zmax=(img_grid_smooth.max() / 1000),
                x=np.arange(img_grid.shape[0]),
                y=np.arange(img_grid.shape[1]),
                labels=dict(x="X", y="Y", color="Norm Freq / 1e3"),
                aspect="auto",
            )
            pos_heatmap_fig.update_layout(title=f"Position Heatmap ({id_val})")

        # 5. Plot position ethogram per subject
        # Melt df to a single "Loc" column that contains loc for current time (row)
        pos_eth_df = pos_eth_df.iloc[::100]  # downsample to 10s bins
        melted_df = pos_eth_df.reset_index().melt(
            id_vars=["time", "Subject"], var_name="Loc", value_name="Val"
        )
        melted_df = melted_df[melted_df["Val"]]

        # Plot using Plotly Express
        pos_etho_fig = px.scatter(
            melted_df,
            x="time",
            y="Subject",
            color="Loc",
            color_discrete_map=roi_colors_dict,
        )

        pos_etho_fig.update_layout(
            title="Position Ethogram",
            xaxis_title="Time",
            yaxis_title="Subject",
            width=1000,
            height=250,
            yaxis=dict(
                categoryorder="total ascending",
                categoryarray=sorted(melted_df["Subject"].unique()),
                tickmode="array",
                tickvals=sorted(melted_df["Subject"].unique()),
                ticktext=sorted(melted_df["Subject"].unique()),
            ),
        )

        # 6. Cumulative pellet count over time per subject markered by patch
        cum_pl_by_patch_fig = go.Figure()

        for id_val, id_grp in cum_pel_ct.groupby("subject"):
            # Add lines by subject
            cum_pl_by_patch_fig.add_trace(
                go.Scatter(
                    x=id_grp["time"],
                    y=id_grp["counter"],
                    mode="lines",
                    line=dict(width=2, color=subject_colors_dict[id_val]),
                    name=id_val,
                )
            )
        for patch_i, (patch_val, patch_grp) in enumerate(cum_pel_ct.groupby("patch_label")):
            # Add markers by patch
            cum_pl_by_patch_fig.add_trace(
                go.Scatter(
                    x=patch_grp["time"],
                    y=patch_grp["counter"],
                    mode="markers",
                    marker=dict(
                        symbol=patch_markers[patch_i],
                        color=gen_hex_grad("#d8d8d8", patch_grp["norm_thresh_val"]),
                        size=8,
                    ),
                    name=patch_val,
                    customdata=np.stack((patch_grp["threshold"],), axis=-1),
                    hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                )
            )

        cum_pl_by_patch_fig.update_layout(
            title="Cumulative Pellet Count per Subject", xaxis_title="Time", yaxis_title="Count"
        )

        # 7. Cumulative pellet count over time, per subject-patch (one line per combo)
        cum_pl_per_subject_fig = go.Figure()
        for id_val, id_grp in cum_pel_ct.groupby("subject"):
            for patch_i, (patch_val, patch_grp) in enumerate(id_grp.groupby("patch")):
                cur_p_mean = patch_means[patch_means["patch"] == patch_val]["mean_thresh"].values[0]
                cur_p = f"P{patch_i+1}"
                cum_pl_per_subject_fig.add_trace(
                    go.Scatter(
                        x=patch_grp["time"],
                        y=np.arange(1, (len(patch_grp) + 1)),
                        mode="lines+markers",
                        line=dict(
                            width=2,
                            color=subject_colors_dict[id_val],
                            dash=patch_markers_linestyles[patch_i],
                        ),
                        # line=dict(width=2, color=subject_colors_dict[id_val]),
                        marker=dict(
                            symbol=patch_markers[patch_i],
                            color=gen_hex_grad("#d8d8d8", patch_grp["norm_thresh_val"]),
                            size=8,
                        ),
                        name=f"{id_val} - {cur_p} - μ: {cur_p_mean}",
                        customdata=np.stack((patch_grp["threshold"],), axis=-1),
                        hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                    )
                )

        cum_pl_per_subject_fig.update_layout(
            title="Cumulative Pellet Count per Subject-Patch", xaxis_title="Time", yaxis_title="Count"
        )

        # 8. Pellet delivery over time per patch-subject
        pl_delivery_fig = go.Figure()
        for id_i, (id_val, id_grp) in enumerate(cum_pel_ct.groupby("subject")):
            # Add lines by subject
            pl_delivery_fig.add_trace(
                go.Scatter(
                    x=id_grp["time"],
                    y=id_grp["patch_label"],
                    # mode="markers",
                    mode="lines+markers",
                    line=dict(width=2, color=subject_colors_dict[id_val]),
                    marker=dict(
                        symbol=patch_markers[0],
                        color=gen_hex_grad(subject_colors[id_i], id_grp["norm_thresh_val"]),
                        size=8,
                    ),
                    name=id_val,
                    customdata=np.stack((id_grp["threshold"],), axis=-1),
                    hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                )
            )

        pl_delivery_fig.update_layout(
            title="Pellet Delivery Over Time, per Subject and Patch",
            xaxis_title="Time",
            yaxis_title="Patch",
            yaxis={
                "categoryorder": "array",
                "categoryarray": cum_pel_ct.sort_values("mean_thresh")[
                    "patch_label"
                ].unique(),  # sort y-axis by patch threshold mean
            },
        )

        # 9. Pellet threshold vals over time per patch-subject
        pl_threshold_fig = go.Figure()

        for id_val, id_grp in cum_pel_ct.groupby("subject"):
            # Add lines by subject
            pl_threshold_fig.add_trace(
                go.Scatter(
                    x=id_grp["time"],
                    y=id_grp["threshold"],
                    mode="lines",
                    line=dict(width=2, color=subject_colors_dict[id_val]),
                    name=id_val,
                )
            )
        for patch_i, (patch_val, patch_grp) in enumerate(cum_pel_ct.groupby("patch_label")):
            # Add markers by patch
            pl_threshold_fig.add_trace(
                go.Scatter(
                    x=patch_grp["time"],
                    y=patch_grp["threshold"],
                    mode="markers",
                    marker=dict(symbol=patch_markers[patch_i], color="black", size=8),
                    name=patch_val,
                )
            )

        pl_threshold_fig.update_layout(
            title="Pellet Thresholds over Time, per Subject",
            xaxis_title="Time",
            yaxis_title="Threshold (cm)",
        )

        # 10. Cumulative wheel distance over time per patch-subject
        # @NOTE: we can round all wheel values to the nearest 0.1 cm in the db, and use this for all downstream calcs
        cum_wheel_dist_fig = go.Figure()
        # Add trace for each subject-patch combo
        for subj_i, subj in enumerate(subject_names):
            for patch_i, p in enumerate(patch_names):
                cur_cum_wheel_dist = subject_patch_data.loc[p].loc[subj]["wheel_cumsum_distance_travelled"]
                cur_p_mean = patch_means[patch_means["patch"] == p]["mean_thresh"].values[0]
                cur_p = f"P{patch_i+1}"
                cum_wheel_dist_fig.add_trace(
                    go.Scatter(
                        x=wheel_ts[p],
                        y=cur_cum_wheel_dist,
                        mode="lines",  # +  markers",
                        line=dict(
                            width=2, color=subject_colors[subj_i], dash=patch_markers_linestyles[patch_i]
                        ),
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == p)],
                    pd.DataFrame({"time": wheel_ts[p], "cum_wheel_dist": cur_cum_wheel_dist}).sort_values(
                        "time"
                    ),
                    on="time",
                    direction="forward",
                    tolerance=pd.Timedelta("0.1s"),
                )
                if not cur_cum_pel_ct.empty:
                    cum_wheel_dist_fig.add_trace(
                        go.Scatter(
                            x=cur_cum_pel_ct["time"],
                            y=cur_cum_pel_ct["cum_wheel_dist"],
                            mode="markers",
                            marker=dict(
                                symbol=patch_markers[patch_i],
                                color=gen_hex_grad(subject_colors[-1], cur_cum_pel_ct["norm_thresh_val"]),
                                size=8,
                            ),
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        cum_wheel_dist_fig.update_layout(
            title="Cumulative Wheel Distance", xaxis_title="Time", yaxis_title="Distance (cm)"
        )

        # 11. Running preference normalized by wheel and time
        running_pref_by_wd_plot = go.Figure()
        # Add trace for each subject-patch combo
        for subj_i, subj in enumerate(subject_names):
            for patch_i, p in enumerate(patch_names):
                cur_run_wheel_pref = patch_pref.loc[p].loc[subj]["running_preference_by_wheel"]
                cur_p_mean = patch_means[patch_means["patch"] == p]["mean_thresh"].values[0]
                cur_p = f"P{patch_i+1}"
                running_pref_by_wd_plot.add_trace(
                    go.Scatter(
                        x=wheel_ts[p],
                        y=cur_run_wheel_pref,
                        mode="lines",
                        line=dict(
                            width=2, color=subject_colors[subj_i], dash=patch_markers_linestyles[patch_i]
                        ),
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == p)],
                    pd.DataFrame({"time": wheel_ts[p], "run_wheel_pref": cur_run_wheel_pref}).sort_values(
                        "time"
                    ),
                    on="time",
                    direction="forward",
                    tolerance=pd.Timedelta("0.1s"),
                )
                if not cur_cum_pel_ct.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=cur_cum_pel_ct["time"],
                            y=cur_cum_pel_ct["run_wheel_pref"],
                            mode="markers",
                            marker=dict(
                                symbol=patch_markers[patch_i],
                                color=gen_hex_grad(subject_colors[-1], cur_cum_pel_ct["norm_thresh_val"]),
                                size=8,
                            ),
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        running_pref_by_wd_plot.update_layout(
            title="Running Patch Preference - Wheel Distance",
            xaxis_title="Time",
            yaxis_title="Preference",
            yaxis=dict(tickvals=np.arange(0, 1.1, 0.1)),
        )

        # 12. Running preference by time in patch
        running_pref_by_patch_fig = go.Figure()
        # Add trace for each subject-patch combo
        for subj_i, subj in enumerate(subject_names):
            for patch_i, p in enumerate(patch_names):
                cur_run_time_pref = patch_pref.loc[p].loc[subj]["running_preference_by_time"]
                cur_p_mean = patch_means[patch_means["patch"] == p]["mean_thresh"].values[0]
                cur_p = f"P{patch_i+1}"
                running_pref_by_patch_fig.add_trace(
                    go.Scatter(
                        x=wheel_ts[p],
                        y=cur_run_time_pref,
                        mode="lines",
                        line=dict(
                            width=2, color=subject_colors[subj_i], dash=patch_markers_linestyles[patch_i]
                        ),
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == p)],
                    pd.DataFrame({"time": wheel_ts[p], "run_time_pref": cur_run_time_pref}).sort_values(
                        "time"
                    ),
                    on="time",
                    direction="forward",
                    tolerance=pd.Timedelta("0.1s"),
                )
                if not cur_cum_pel_ct.empty:
                    running_pref_by_patch_fig.add_trace(
                        go.Scatter(
                            x=cur_cum_pel_ct["time"],
                            y=cur_cum_pel_ct["run_time_pref"],
                            mode="markers",
                            marker=dict(
                                symbol=patch_markers[patch_i],
                                color=gen_hex_grad(subject_colors[-1], cur_cum_pel_ct["norm_thresh_val"]),
                                size=8,
                            ),
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        running_pref_by_patch_fig.update_layout(
            title="Running Patch Preference - Time in Patch",
            xaxis_title="Time",
            yaxis_title="Preference",
            yaxis=dict(tickvals=np.arange(0, 1.1, 0.1)),
        )

        self.insert1(
            {
                **key,
                "patch_stats_plot": json.loads(patch_stats_fig.to_json()),
                "weights_block_plot": json.loads(weights_block_fig.to_json()),
                "position_block_plot": json.loads(position_block_fig.to_json()),
                "position_heatmap_plot": json.loads(pos_heatmap_fig.to_json()),
                "position_ethogram_plot": json.loads(pos_etho_fig.to_json()),
                "cum_pl_by_patch_plot": json.loads(cum_pl_by_patch_fig.to_json()),
                "cum_pl_per_subject_plot": json.loads(cum_pl_per_subject_fig.to_json()),
                "pellet_delivery_plot": json.loads(pl_delivery_fig.to_json()),
                "pellet_threshold_plot": json.loads(pl_threshold_fig.to_json()),
                "cum_wheel_dist_plot": json.loads(cum_wheel_dist_fig.to_json()),
                "running_pref_by_wheel_dist_plot": json.loads(running_pref_by_wd_plot.to_json()),
                "running_pref_by_patch_plot": json.loads(running_pref_by_patch_fig.to_json()),
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
