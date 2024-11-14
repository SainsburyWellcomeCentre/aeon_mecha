"""Module for block analysis."""

import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone

import datajoint as dj
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from aeon.analysis import utils as analysis_utils
from aeon.analysis.block_plotting import (
    conv2d,
    gen_hex_grad,
    gen_patch_style_dict,
    gen_subject_colors_dict,
    subject_colors,
)
from aeon.dj_pipeline import (
    acquisition,
    fetch_stream,
    get_schema_name,
    streams,
    tracking,
)
from aeon.dj_pipeline.analysis.visit import (
    filter_out_maintenance_periods,
    get_maintenance_periods,
)
from aeon.io import api as io_api

schema = dj.schema(get_schema_name("block_analysis"))
logger = dj.logger


@schema
class Block(dj.Manual):
    definition = """
    -> acquisition.Experiment
    block_start: datetime(6)
    ---
    block_end=null: datetime(6)
    block_duration_hr=null: decimal(6, 3)  # (hour)
    """


@schema
class BlockDetection(dj.Computed):
    definition = """
    -> acquisition.Environment
    """

    key_source = acquisition.Environment - {"experiment_name": "social0.1-aeon3"}

    def make(self, key):
        """On a per-chunk basis, check for the presence of new block, insert into Block table.

        High level logic
        1. Find the 0s in `pellet_ct` (these are times when the pellet count reset - i.e. new block)
        2. Remove any double 0s (0s within 1 second of each other) (pick the first 0)
        3. Calculate block end_times (use due_time) and durations
        4. Insert into Block table
        """
        # find the 0s in `pellet_ct` (these are times when the pellet count reset - i.e. new block)
        # that would mark the start of a new block

        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")
        exp_key = {"experiment_name": key["experiment_name"]}

        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], chunk_start, chunk_end
        )

        block_state_query = acquisition.Environment.BlockState & exp_key & chunk_restriction
        block_state_df = fetch_stream(block_state_query)
        if block_state_df.empty:
            self.insert1(key)
            return

        block_state_df.index = block_state_df.index.round(
            "us"
        )  # timestamp precision in DJ is only at microseconds
        block_state_df = block_state_df.loc[
            (block_state_df.index > chunk_start) & (block_state_df.index <= chunk_end)
        ]

        blocks_df = block_state_df[block_state_df.pellet_ct == 0]
        # account for the double 0s - find any 0s that are within 1 second of each other, remove the 2nd one
        double_0s = blocks_df.index.to_series().diff().dt.total_seconds() < 1
        # keep the first 0s
        blocks_df = blocks_df[~double_0s]

        block_entries = []
        if not blocks_df.empty:
            # calculate block end_times (use due_time) and durations
            blocks_df["end_time"] = blocks_df["due_time"].apply(lambda x: io_api.aeon(x))
            blocks_df["duration"] = (blocks_df["end_time"] - blocks_df.index).dt.total_seconds() / 3600

            for _, row in blocks_df.iterrows():
                block_entries.append(
                    {
                        **exp_key,
                        "block_start": row.name,
                        "block_end": row.end_time,
                        "block_duration_hr": row.duration,
                    }
                )

        Block.insert(block_entries, skip_duplicates=True)
        self.insert1(key)


# ---- Block Analysis and Visualization ----


@schema
class BlockAnalysis(dj.Computed):
    definition = """
    -> Block
    ---
    block_duration: float  # (hour)
    patch_count=null: int  # number of patches in the block
    subject_count=null: int  # number of subjects in the block
    """

    @property
    def key_source(self):
        """Ensures chunk ingestion is complete before processing the block.

        This is done by checking that there exists a chunk that ends after the block end time.
        """
        ks = Block.aggr(acquisition.Chunk, latest_chunk_end="MAX(chunk_end)")
        ks = ks * Block & "latest_chunk_end >= block_end" & "block_end IS NOT NULL"
        return ks

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
        patch_rate=null: float
        patch_offset=null: float
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
        """Collates data from various streams to produce per-block intermediate data products.

        The intermediate data products consist of data for each ``Patch``
        and each ``Subject`` within the  ``Block``.
        The steps to restrict, fetch, and aggregate data from various streams are as follows:

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
        )
        for streams_table in streams_tables:
            if len(streams_table & chunk_keys) < len(streams_table.key_source & chunk_keys):
                raise ValueError(
                    f"BlockAnalysis Not Ready - {streams_table.__name__}"
                    f"not yet fully ingested for block: {key}."
                    f"Skipping (to retry later)..."
                )

        # Check if SLEAPTracking is ready, if not, see if BlobPosition can be used instead
        use_blob_position = False
        if len(tracking.SLEAPTracking & chunk_keys) < len(tracking.SLEAPTracking.key_source & chunk_keys):
            if len(tracking.BlobPosition & chunk_keys) < len(tracking.BlobPosition.key_source & chunk_keys):
                raise ValueError(
                    f"BlockAnalysis Not Ready - SLEAPTracking (and BlobPosition) not yet fully ingested for block: {key}. Skipping (to retry later)..."
                )
            else:
                use_blob_position = True

        # Patch data - TriggerPellet, DepletionState, Encoder (distancetravelled)
        # For wheel data, downsample to 50Hz
        final_encoder_hz = 50
        freq = 1 / final_encoder_hz * 1e3  # in ms

        maintenance_period = get_maintenance_periods(key["experiment_name"], block_start, block_end)

        patch_query = (
            streams.UndergroundFeeder.join(streams.UndergroundFeeder.RemovalTime, left=True)
            & key
            & f'"{block_start}" >= underground_feeder_install_time'
            & f'"{block_end}" < IFNULL(underground_feeder_removal_time, "2200-01-01")'
        )
        patch_keys, patch_names = patch_query.fetch("KEY", "underground_feeder_name")

        block_patch_entries = []
        for patch_key, patch_name in zip(patch_keys, patch_names, strict=False):
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

            # if all dataframes are empty, skip
            if pellet_ts_threshold_df.empty and depletion_state_df.empty and encoder_df.empty:
                continue

            if encoder_df.empty:
                encoder_df["distance_travelled"] = 0
            else:
                # -1 is for placement of magnetic encoder, where wheel movement actually decreases encoder
                encoder_df["distance_travelled"] = -1 * analysis_utils.distancetravelled(encoder_df.angle)
                encoder_df = encoder_df.resample(f"{freq}ms").first()

            if not depletion_state_df.empty:
                if len(depletion_state_df.rate.unique()) > 1:
                    # multiple patch rates per block is unexpected
                    # log a note and pick the first rate to move forward
                    AnalysisNote.insert1(
                        {
                            "note_timestamp": datetime.now(timezone.utc),
                            "note_type": "Multiple patch rates",
                            "note": (
                                f"Found multiple patch rates for block {key} "
                                f"- patch: {patch_name} "
                                f"- rates: {depletion_state_df.rate.unique()}"
                            ),
                        }
                    )

                patch_rate = depletion_state_df.rate.iloc[0]
                patch_offset = depletion_state_df.offset.iloc[0]
                # handles patch rate value being INF
                patch_rate = 999999999 if np.isinf(patch_rate) else patch_rate
            else:
                logger.warning(f"No depletion state data found for block {key} - patch: {patch_name}")
                patch_rate = None
                patch_offset = None

            block_patch_entries.append(
                {
                    **key,
                    "patch_name": patch_name,
                    "pellet_count": len(pellet_ts_threshold_df),
                    "pellet_timestamps": pellet_ts_threshold_df.pellet_timestamp.values,
                    "wheel_cumsum_distance_travelled": encoder_df.distance_travelled.values,
                    "wheel_timestamps": encoder_df.index.values,
                    "patch_threshold": pellet_ts_threshold_df.threshold.values,
                    "patch_threshold_timestamps": pellet_ts_threshold_df.index.values,
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
        subject_visits_df = subject_visits_df[~subject_visits_df.id.str.contains("Test", case=False)]
        subject_names = []
        for subject_name in set(subject_visits_df.id):
            _df = subject_visits_df[subject_visits_df.id == subject_name]
            if _df.type.iloc[-1] != "Exit":
                subject_names.append(subject_name)

        if use_blob_position and len(subject_names) > 1:
            raise ValueError(
                f"Without SLEAPTracking, BlobPosition can only handle single-subject block. Found {len(subject_names)} subjects."
            )

        block_subject_entries = []
        for subject_name in subject_names:
            # positions - query for CameraTop, identity_name matches subject_name,
            if use_blob_position:
                pos_query = (
                    streams.SpinnakerVideoSource
                    * tracking.BlobPosition.Object
                    & key
                    & chunk_restriction
                    & {
                        "spinnaker_video_source_name": "CameraTop",
                        "identity_name": subject_name
                    }
                )
                pos_df = fetch_stream(pos_query)[block_start:block_end]
                pos_df["likelihood"] = np.nan
                # keep only rows with area between 0 and 1000 - likely artifacts otherwise
                pos_df = pos_df[(pos_df.area > 0) & (pos_df.area < 1000)]
            else:
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

            block_subject_entries.append(
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
            block_end = min(pos_df.index[-1], block_end)

        self.insert1(
            {
                **key,
                "block_duration": (block_end - block_start).total_seconds() / 3600,
                "patch_count": len(block_patch_entries),
                "subject_count": len(block_subject_entries),
            }
        )
        self.Patch.insert(block_patch_entries)
        self.Subject.insert(block_subject_entries)

        if block_end != (Block & key).fetch1("block_end"):
            self.update1(
                {
                    **key,
                    "block_duration": (block_end - block_start).total_seconds() / 3600,
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
        in_patch_timestamps: longblob # timestamps when a subject is at a specific patch
        in_patch_time: float  # total seconds spent in this patch for this block
        pellet_count: int
        pellet_timestamps: longblob
        patch_threshold: longblob  # patch threshold value at each pellet delivery
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
        running_preference_by_time=null: longblob
        running_preference_by_wheel=null: longblob
        final_preference_by_wheel=null: float  # cumulative_preference_by_wheel at the end of the block
        final_preference_by_time=null: float  # cumulative_preference_by_time at the end of the block
        """

    key_source = BlockAnalysis & BlockAnalysis.Patch & BlockAnalysis.Subject

    def make(self, key):
        """Compute preference scores for each subject at each patch within a block."""
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

        # Ensure wheel_timestamps are of the same length across all patches
        wheel_lens = [len(p["wheel_timestamps"]) for p in block_patches]
        if len(set(wheel_lens)) > 1:
            max_diff = max(wheel_lens) - min(wheel_lens)
            if max_diff > 10:
                # if diff is more than 10 samples, raise error, this is unexpected, some patches crash?
                raise ValueError(f"Wheel data lengths are not consistent across patches ({max_diff} samples diff)")
            for p in block_patches:
                p["wheel_timestamps"] = p["wheel_timestamps"][: min(wheel_lens)]
                p["wheel_cumsum_distance_travelled"] = p["wheel_cumsum_distance_travelled"][: min(wheel_lens)]

        self.insert1(key)

        in_patch_radius = 130  # pixels
        pref_attrs = [
            "cum_dist",
            "cum_time",
            "running_dist_pref",
            "running_time_pref",
            "cum_pref_dist",
            "cum_pref_time",
        ]
        all_subj_patch_pref_dict = {
            p: {s: {a: pd.Series() for a in pref_attrs} for s in subject_names} for p in patch_names
        }

        for patch in block_patches:
            cum_wheel_dist = pd.Series(
                index=patch["wheel_timestamps"],
                data=patch["wheel_cumsum_distance_travelled"],
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
                all_subj_patch_pref_dict[patch["patch_name"]][subject_name]["cum_time"] = (
                    subject_in_patch_cum_time
                )

                closest_subj_mask = closest_subjects_pellet_ts == subject_name
                subj_pellets = closest_subjects_pellet_ts[closest_subj_mask]
                subj_patch_thresh = patch["patch_threshold"][closest_subj_mask]

                self.Patch.insert1(
                    key
                    | {
                        "patch_name": patch["patch_name"],
                        "subject_name": subject_name,
                        "in_patch_timestamps": subject_in_patch[in_patch[subject_name]].index.values,
                        "in_patch_time": subject_in_patch_cum_time[-1],
                        "pellet_count": len(subj_pellets),
                        "pellet_timestamps": subj_pellets.index.values,
                        "patch_threshold": subj_patch_thresh,
                        "wheel_cumsum_distance_travelled": cum_wheel_dist_subj_df[subject_name].values,
                    }
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

            CUM_PREF_DIST_MIN = 1e-3

            for patch_name in patch_names:
                cum_pref_dist = (
                    all_subj_patch_pref_dict[patch_name][subject_name]["cum_dist"] / all_cum_dist
                )
                cum_pref_dist = np.where(cum_pref_dist < CUM_PREF_DIST_MIN, 0, cum_pref_dist)
                all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_dist"] = cum_pref_dist

                cum_pref_time = (
                    all_subj_patch_pref_dict[patch_name][subject_name]["cum_time"] / all_cum_time
                )
                all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_time"] = cum_pref_time

            # sum pref at each ts across patches for each subject
            total_dist_pref = np.sum(
                np.vstack(
                    [all_subj_patch_pref_dict[p][subject_name]["cum_pref_dist"] for p in patch_names]
                ),
                axis=0,
            )
            total_time_pref = np.sum(
                np.vstack(
                    [all_subj_patch_pref_dict[p][subject_name]["cum_pref_time"] for p in patch_names]
                ),
                axis=0,
            )
            for patch_name in patch_names:
                cum_pref_dist = all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_dist"]
                all_subj_patch_pref_dict[patch_name][subject_name]["running_dist_pref"] = np.divide(
                    cum_pref_dist,
                    total_dist_pref,
                    out=np.zeros_like(cum_pref_dist),
                    where=total_dist_pref != 0,
                )
                cum_pref_time = all_subj_patch_pref_dict[patch_name][subject_name]["cum_pref_time"]
                all_subj_patch_pref_dict[patch_name][subject_name]["running_time_pref"] = np.divide(
                    cum_pref_time,
                    total_time_pref,
                    out=np.zeros_like(cum_pref_time),
                    where=total_time_pref != 0,
                )

        self.Preference.insert(
            key
            | {
                "patch_name": p,
                "subject_name": s,
                "cumulative_preference_by_time": all_subj_patch_pref_dict[p][s]["cum_pref_time"],
                "cumulative_preference_by_wheel": all_subj_patch_pref_dict[p][s]["cum_pref_dist"],
                "running_preference_by_time": all_subj_patch_pref_dict[p][s]["running_time_pref"],
                "running_preference_by_wheel": all_subj_patch_pref_dict[p][s]["running_dist_pref"],
                "final_preference_by_time": all_subj_patch_pref_dict[p][s]["cum_pref_time"][-1],
                "final_preference_by_wheel": all_subj_patch_pref_dict[p][s]["cum_pref_dist"][-1],
            }
            for p, s in itertools.product(patch_names, subject_names)
        )


@schema
class BlockPatchPlots(dj.Computed):
    definition = """
    -> BlockSubjectAnalysis
    ---
    patch_stats_plot: longblob
    weights_block_plot: longblob
    cum_pel_by_patch_plot: longblob
    cum_pel_per_subject_plot: longblob
    cum_wheel_dist_plot: longblob
    running_pref_by_wheel_dist_plot: longblob
    running_pref_by_patch_plot: longblob
    weighted_patch_pref_plot: longblob
    """

    def make(self, key):
        """Compute and plot various block-level statistics and visualizations."""
        # Define subject colors and patch styling for plotting
        exp_subject_names = (acquisition.Experiment.Subject & key).fetch("subject", order_by="subject")
        if not len(exp_subject_names):
            raise ValueError(
                "No subjects found in the `acquisition.Experiment.Subject`, missing a manual insert step?."
            )
        subject_colors_dict = gen_subject_colors_dict(exp_subject_names)
        exp_patch_names = np.unique(
            (streams.UndergroundFeeder & key).fetch(
                "underground_feeder_name", order_by="underground_feeder_name"
            )
        )
        patch_style_dict = gen_patch_style_dict(exp_patch_names)
        patch_markers_dict = patch_style_dict["markers"]
        patch_linestyles_dict = patch_style_dict["linestyles"]
        pel_mrkr_col = "#d8d8d8"

        # Figure 1 - Patch stats: patch means and pellet threshold boxplots
        # ---
        subj_patch_info = (
            (BlockSubjectAnalysis.Patch.proj("pellet_timestamps", "patch_threshold") & key)
            .fetch(format="frame")
            .reset_index()
        )
        patch_info = (BlockAnalysis.Patch & key).fetch(
            "patch_name", "patch_rate", "patch_offset", as_dict=True
        )
        patch_names = list(subj_patch_info["patch_name"].unique())
        subject_names = list(subj_patch_info["subject_name"].unique())
        # Convert `subj_patch_info` into a form amenable to plotting
        min_subj_patch_info = subj_patch_info[  # select only relevant columns
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
        patch_mean_info["time"] = subj_patch_info["block_start"][0]
        min_subj_patch_info_plus = pd.concat((patch_mean_info, min_subj_patch_info)).reset_index(drop=True)
        min_subj_patch_info_plus["norm_time"] = (
            (min_subj_patch_info_plus["time"] - min_subj_patch_info_plus["time"].iloc[0])
            / (min_subj_patch_info_plus["time"].iloc[-1] - min_subj_patch_info_plus["time"].iloc[0])
        ).round(3)

        # Plot it
        box_colors = ["#0A0A0A"] + list(subject_colors_dict.values())  # subject colors + mean color
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

        # Figure 2 - Animal weights: over time, per subject
        # ---
        # Get weight data and make df amenable to plotting
        weights_block = (
            (BlockAnalysis.Subject.proj("weights", "weight_timestamps") & key)
            .fetch(format="frame")
            .reset_index()
        )
        weights_block = (
            weights_block.explode(["weights", "weight_timestamps"], ignore_index=True)
            .dropna()
            .reset_index(drop=True)
        )
        weights_block.drop(columns=["experiment_name", "block_start"], inplace=True, errors="ignore")
        weights_block.rename(columns={"weight_timestamps": "time"}, inplace=True)
        weights_block.set_index("time", inplace=True)
        weights_block.sort_index(inplace=True)

        # Plot it
        weights_block_fig = px.line(
            weights_block,
            x=weights_block.index,
            y="weights",
            color="subject_name",
            color_discrete_map=subject_colors_dict,
            markers=True,
        )
        weights_block_fig.update_traces(line={"width": 3}, marker={"size": 8})
        weights_block_fig.update_layout(
            title="Weights in block",
            xaxis_title="Time",
            yaxis_title="Weight (g)",
        )

        # Figure 3 - Cumulative pellet count: over time, per subject, markered by patch
        # ---
        # Create dataframe with cumulative pellet count per subject
        cum_pel_ct = min_subj_patch_info_plus.sort_values("time").copy().reset_index(drop=True)
        patch_means = cum_pel_ct.loc[0:3][["patch", "threshold"]].rename(
            columns={"threshold": "mean_thresh"}
        )
        patch_means["mean_thresh"] = patch_means["mean_thresh"].astype(float).round(1)
        cum_pel_ct = cum_pel_ct.merge(patch_means, on="patch", how="left")
        cum_pel_ct = cum_pel_ct[~cum_pel_ct["subject"].str.contains("mean")].reset_index(drop=True)
        cum_pel_ct = (
            cum_pel_ct.groupby("subject", group_keys=False)
            .apply(lambda group: group.assign(counter=np.arange(len(group)) + 1))
            .reset_index(drop=True)
        )
        # Tidy up the dataframe
        make_float_cols = ["threshold", "mean_thresh", "norm_time"]
        cum_pel_ct[make_float_cols] = cum_pel_ct[make_float_cols].astype(float)
        cum_pel_ct["patch_label"] = (
            cum_pel_ct["patch"] + " μ: " + cum_pel_ct["mean_thresh"].astype(float).round(1).astype(str)
        )
        cum_pel_ct["norm_thresh_val"] = (
            (cum_pel_ct["threshold"] - cum_pel_ct["threshold"].min())
            / (cum_pel_ct["threshold"].max() - cum_pel_ct["threshold"].min())
        ).round(3)
        cum_pel_ct = cum_pel_ct.sort_values("time")

        # Plot it
        cum_pel_by_patch_fig = go.Figure()
        for id_val, id_grp in cum_pel_ct.groupby("subject"):
            # Add lines by subject
            cum_pel_by_patch_fig.add_trace(
                go.Scatter(
                    x=id_grp["time"],
                    y=id_grp["counter"],
                    mode="lines",
                    line={"width": 2, "color": subject_colors_dict[id_val]},
                    name=id_val,
                )
            )
        for patch_val, patch_grp in cum_pel_ct.groupby("patch_label"):
            # Add markers by patch
            cum_pel_by_patch_fig.add_trace(
                go.Scatter(
                    x=patch_grp["time"],
                    y=patch_grp["counter"],
                    mode="markers",
                    marker={
                        "symbol": patch_markers_dict[patch_grp["patch"].iloc[0]],
                        "color": gen_hex_grad(pel_mrkr_col, patch_grp["norm_thresh_val"]),
                        "size": 8,
                    },
                    name=patch_val,
                    customdata=np.stack((patch_grp["threshold"],), axis=-1),
                    hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                )
            )
        cum_pel_by_patch_fig.update_layout(
            title="Cumulative Pellet Count per Subject",
            xaxis_title="Time",
            yaxis_title="Count",
        )

        # Figure 4 - Cumulative pellet count: over time, per subject-patch (one line per combo)
        # ---
        # Plot it
        cum_pel_per_subject_fig = go.Figure()
        for id_val, id_grp in cum_pel_ct.groupby("subject"):
            for patch_val, patch_grp in id_grp.groupby("patch"):
                cur_p_mean = patch_means[patch_means["patch"] == patch_val]["mean_thresh"].values[0]
                cur_p = patch_val.replace("Patch", "P")
                cum_pel_per_subject_fig.add_trace(
                    go.Scatter(
                        x=patch_grp["time"],
                        y=np.arange(1, (len(patch_grp) + 1)),
                        mode="lines+markers",
                        line={
                            "width": 2,
                            "color": subject_colors_dict[id_val],
                            "dash": patch_linestyles_dict[patch_val],
                        },
                        # line=dict(width=2, color=subject_colors_dict[id_val]),
                        marker={
                            "symbol": patch_markers_dict[patch_val],
                            "color": gen_hex_grad(pel_mrkr_col, patch_grp["norm_thresh_val"]),
                            "size": 8,
                        },
                        name=f"{id_val} - {cur_p} - μ: {cur_p_mean}",
                        customdata=np.stack((patch_grp["threshold"],), axis=-1),
                        hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                    )
                )
        cum_pel_per_subject_fig.update_layout(
            title="Cumulative Pellet Count per Subject-Patch",
            xaxis_title="Time",
            yaxis_title="Count",
        )

        # Figure 5 - Cumulative wheel distance: over time, per subject-patch
        # ---
        # Get wheel timestamps for each patch
        wheel_ts = (BlockAnalysis.Patch & key).fetch("patch_name", "wheel_timestamps", as_dict=True)
        wheel_ts = {d["patch_name"]: d["wheel_timestamps"] for d in wheel_ts}
        # Get subject patch data
        subj_wheel_cumsum_dist = (BlockSubjectAnalysis.Patch & key).fetch(
            "subject_name",
            "patch_name",
            "wheel_cumsum_distance_travelled",
            as_dict=True,
        )
        subj_wheel_cumsum_dist = {
            (d["subject_name"], d["patch_name"]): d["wheel_cumsum_distance_travelled"]
            for d in subj_wheel_cumsum_dist
        }

        # Plot it
        cum_wheel_dist_fig = go.Figure()
        # Add trace for each subject-patch combo
        for subj in subject_names:
            for patch_name in patch_names:
                cur_cum_wheel_dist = subj_wheel_cumsum_dist[(subj, patch_name)]
                cur_p_mean = patch_means[patch_means["patch"] == patch_name]["mean_thresh"].values[0]
                cur_p = patch_name.replace("Patch", "P")
                cum_wheel_dist_fig.add_trace(
                    go.Scatter(
                        x=wheel_ts[patch_name],
                        y=cur_cum_wheel_dist,
                        mode="lines",  # +  markers",
                        line={
                            "width": 2,
                            "color": subject_colors_dict[subj],
                            "dash": patch_linestyles_dict[patch_name],
                        },
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == patch_name)],
                    pd.DataFrame(
                        {
                            "time": wheel_ts[patch_name],
                            "cum_wheel_dist": cur_cum_wheel_dist,
                        }
                    ).sort_values("time"),
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
                            marker={
                                "symbol": patch_markers_dict[patch_name],
                                "color": gen_hex_grad(pel_mrkr_col, cur_cum_pel_ct["norm_thresh_val"]),
                                "size": 8,
                            },
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        cum_wheel_dist_fig.update_layout(
            title="Cumulative Wheel Distance",
            xaxis_title="Time",
            yaxis_title="Distance (cm)",
        )

        # Figure 6 - Patch preference: Running, normalized, over time, by wheel dist spun, per subject-patch
        # ---
        # Get and format a dataframe with preference data
        patch_pref = (BlockSubjectAnalysis.Preference & key).fetch(format="frame")
        patch_pref.reset_index(level=["experiment_name", "block_start"], drop=True, inplace=True)
        # Replace small vals with 0
        small_pref_thresh = 1e-3
        patch_pref["cumulative_preference_by_wheel"] = patch_pref["cumulative_preference_by_wheel"].apply(
            lambda arr: np.where(np.array(arr) < small_pref_thresh, 0, np.array(arr))
        )

        def calculate_running_preference(group, pref_col, out_col):
            # Sum pref at each ts
            total_pref = np.sum(np.vstack(group[pref_col].values), axis=0)
            # Calculate running pref
            group[out_col] = group[pref_col].apply(lambda x: np.nan_to_num(x / total_pref, 0.0))
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

        # Plot it
        running_pref_by_wheel_plot = go.Figure()
        # Add trace for each subject-patch combo
        for subj in subject_names:
            for patch_name in patch_names:
                cur_run_wheel_pref = patch_pref.loc[patch_name].loc[subj]["running_preference_by_wheel"]
                cur_p_mean = patch_means[patch_means["patch"] == patch_name]["mean_thresh"].values[0]
                cur_p = patch_name.replace("Patch", "P")
                running_pref_by_wheel_plot.add_trace(
                    go.Scatter(
                        x=wheel_ts[patch_name],
                        y=cur_run_wheel_pref,
                        mode="lines",
                        line={
                            "width": 2,
                            "color": subject_colors_dict[subj],
                            "dash": patch_linestyles_dict[patch_name],
                        },
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == patch_name)],
                    pd.DataFrame(
                        {
                            "time": wheel_ts[patch_name],
                            "run_wheel_pref": cur_run_wheel_pref,
                        }
                    ).sort_values("time"),
                    on="time",
                    direction="forward",
                    tolerance=pd.Timedelta("0.1s"),
                )
                if not cur_cum_pel_ct.empty:
                    running_pref_by_wheel_plot.add_trace(
                        go.Scatter(
                            x=cur_cum_pel_ct["time"],
                            y=cur_cum_pel_ct["run_wheel_pref"],
                            mode="markers",
                            marker={
                                "symbol": patch_markers_dict[patch_name],
                                "color": gen_hex_grad(pel_mrkr_col, cur_cum_pel_ct["norm_thresh_val"]),
                                "size": 8,
                            },
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        running_pref_by_wheel_plot.update_layout(
            title="Running Patch Preference - Wheel Distance",
            xaxis_title="Time",
            yaxis_title="Preference",
            yaxis={"tickvals": np.arange(0, 1.1, 0.1)},
        )

        # Figure 7 - Patch preference: Running, normalized, over time, by in-patch time, per subject-patch
        running_pref_by_patch_fig = go.Figure()
        # Add trace for each subject-patch combo
        for subj in subject_names:
            for patch_name in patch_names:
                cur_run_time_pref = patch_pref.loc[patch_name].loc[subj]["running_preference_by_time"]
                cur_p_mean = patch_means[patch_means["patch"] == patch_name]["mean_thresh"].values[0]
                cur_p = patch_name.replace("Patch", "P")
                running_pref_by_patch_fig.add_trace(
                    go.Scatter(
                        x=wheel_ts[patch_name],
                        y=cur_run_time_pref,
                        mode="lines",
                        line={
                            "width": 2,
                            "color": subject_colors_dict[subj],
                            "dash": patch_linestyles_dict[patch_name],
                        },
                        name=f"{subj} - {cur_p} - μ: {cur_p_mean}",
                    )
                )
                # Add markers for each pellet
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == subj) & (cum_pel_ct["patch"] == patch_name)],
                    pd.DataFrame(
                        {
                            "time": wheel_ts[patch_name],
                            "run_time_pref": cur_run_time_pref,
                        }
                    ).sort_values("time"),
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
                            marker={
                                "symbol": patch_markers_dict[patch_name],
                                "color": gen_hex_grad(pel_mrkr_col, cur_cum_pel_ct["norm_thresh_val"]),
                                "size": 8,
                            },
                            name=f"{subj} - {cur_p} pellets",
                            customdata=np.stack((cur_cum_pel_ct["threshold"],), axis=-1),
                            hovertemplate="Threshold: %{customdata[0]:.2f} cm",
                        )
                    )
        running_pref_by_patch_fig.update_layout(
            title="Running Patch Preference - Time in Patch",
            xaxis_title="Time",
            yaxis_title="Preference",
            yaxis={"tickvals": np.arange(0, 1.1, 0.1)},
        )

        # Figure 8 - Weighted patch preference: weighted by 'wheel_dist_spun : pel_ct' ratio
        # ---
        # Create multi-indexed dataframe with weighted distance for each subject-patch pair
        pel_patches = [p for p in patch_names if "dummy" not in p.lower()]  # exclude dummy patches
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

        # Calculate weighted distance
        subject_patch_data = (BlockSubjectAnalysis.Patch() & key).fetch(format="frame")
        subject_patch_data.reset_index(level=["experiment_name", "block_start"], drop=True, inplace=True)
        subj_wheel_pel_weighted_dist = defaultdict(lambda: defaultdict(dict))
        for s in subject_names:
            for p in pel_patches:
                # Get cumulative wheel distance
                cur_wheel_cum_dist_df = pd.DataFrame(columns=["time", "cum_wheel_dist"])
                cur_wheel_cum_dist_df["time"] = wheel_ts[p]
                cur_wheel_cum_dist_df["cum_wheel_dist"] = (
                    subject_patch_data.loc[p].loc[s]["wheel_cumsum_distance_travelled"] + 1
                )
                # Get cumulative pellet count
                cur_cum_pel_ct = pd.merge_asof(
                    cum_pel_ct[(cum_pel_ct["subject"] == s) & (cum_pel_ct["patch"] == p)],
                    cur_wheel_cum_dist_df.sort_values("time"),
                    on="time",
                    direction="forward",
                    tolerance=pd.Timedelta("0.1s"),
                )
                # Perform weighting
                if len(cur_cum_pel_ct) > 0:
                    cur_cum_pel_ct = cum_pel_ct[
                        (cum_pel_ct["subject"] == s) & (cum_pel_ct["patch"] == p)
                    ].reset_index(drop=True)
                    cur_cum_pel_ct["counter"] = cur_cum_pel_ct.index + 1
                    # Weight `cum_wheel_dist` by `counter`
                    merged_df = pd.merge_asof(
                        cur_wheel_cum_dist_df,
                        cur_cum_pel_ct[["time", "counter"]],
                        on="time",
                        direction="forward",
                    )
                    max_weight = cur_cum_pel_ct.iloc[-1]["counter"] + 1  # for values after last pellet
                    merged_df["counter"] = merged_df["counter"].fillna(max_weight)
                    merged_df["weighted_cum_wheel_dist"] = (
                        merged_df.groupby("counter")
                        .apply(lambda x: x["cum_wheel_dist"] / x["counter"].iloc[0])
                        .reset_index(level=0, drop=True)
                    )
                    weighted_dist = merged_df["weighted_cum_wheel_dist"].values
                else:
                    weighted_dist = cur_wheel_cum_dist_df["cum_wheel_dist"].values
                # Assign to dict
                subj_wheel_pel_weighted_dist[p][s]["time"] = cur_wheel_cum_dist_df["time"].values
                subj_wheel_pel_weighted_dist[p][s]["weighted_dist"] = weighted_dist
        # Convert back to dataframe
        data = []
        for p in pel_patches:
            for s in subject_names:
                data.append(
                    {
                        "patch_name": p,
                        "subject_name": s,
                        "time": subj_wheel_pel_weighted_dist[p][s]["time"],
                        "weighted_dist": subj_wheel_pel_weighted_dist[p][s]["weighted_dist"],
                    }
                )
        subj_wheel_pel_weighted_dist = pd.DataFrame(data)
        subj_wheel_pel_weighted_dist.set_index(["patch_name", "subject_name"], inplace=True)

        # Calculate normalized weighted value
        def norm_inv_norm(group):
            each_weighted_dist = np.vstack(group["weighted_dist"])
            norm_dist = each_weighted_dist / np.sum(each_weighted_dist, axis=0)
            inv_norm_dist = 1 / norm_dist
            inv_norm_dist = inv_norm_dist / (np.sum(inv_norm_dist, axis=0))
            # Map each inv_norm_dist back to patch name.
            return pd.Series(inv_norm_dist.tolist(), index=group.index, name="norm_value")

        subj_wheel_pel_weighted_dist["norm_value"] = (
            subj_wheel_pel_weighted_dist.groupby("subject_name")
            .apply(norm_inv_norm)
            .reset_index(level=0, drop=True)
        )
        subj_wheel_pel_weighted_dist["wheel_pref"] = patch_pref["running_preference_by_wheel"]

        # Plot it
        weighted_patch_pref_fig = make_subplots(
            rows=len(pel_patches),
            cols=len(subject_names),
            subplot_titles=[f"{patch} - {subject}" for patch in pel_patches for subject in subject_names],
            specs=[[{"secondary_y": True}] * len(subject_names)] * len(pel_patches),
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.15,
        )

        df = subj_wheel_pel_weighted_dist
        # Iterate through patches and subjects to create plots
        for i, patch in enumerate(pel_patches, start=1):
            for j, subject in enumerate(subject_names, start=1):
                # Filter data for this patch and subject
                times = df.loc[patch].loc[subject]["time"]
                norm_values = df.loc[patch].loc[subject]["norm_value"]
                wheel_prefs = df.loc[patch].loc[subject]["wheel_pref"]

                # Add wheel_pref trace
                weighted_patch_pref_fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=wheel_prefs,
                        name=f"{subject} - wheel_pref",
                        line={
                            "color": subject_colors[i - 1],
                            "dash": patch_linestyles_dict[patch],
                            "width": 1.5,
                        },
                        legendgroup=f"group{i}{j}",
                    ),
                    row=i,
                    col=j,
                    secondary_y=True,
                )

                # Add norm_value trace
                weighted_patch_pref_fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=norm_values,
                        name=f"{subject} - norm_value",
                        line={
                            "color": subject_colors[i - 1],
                            "dash": patch_linestyles_dict[patch],
                            "width": 1.5,
                        },
                        legendgroup=f"group{i}{j}",
                    ),
                    row=i,
                    col=j,
                    secondary_y=False,
                )

                # Update axes
                weighted_patch_pref_fig.update_xaxes(title_text="Time", row=i, col=j)
                weighted_patch_pref_fig.update_yaxes(
                    title_text="norm_value", secondary_y=False, row=i, col=j
                )
                weighted_patch_pref_fig.update_yaxes(
                    title_text="wheel_pref", secondary_y=True, row=i, col=j
                )

        # Update layout
        weighted_patch_pref_fig.update_layout(
            height=900,
            width=1200,
            title_text="Patch Preference (solid) and Experienced Value (dashed) over Time",
            showlegend=False,
        )

        entry = dict(key)
        for fig, fig_name in zip(
            [
                patch_stats_fig,
                weights_block_fig,
                cum_pel_by_patch_fig,
                cum_pel_per_subject_fig,
                cum_wheel_dist_fig,
                running_pref_by_wheel_plot,
                running_pref_by_patch_fig,
                weighted_patch_pref_fig,
            ],
            [
                "patch_stats_plot",
                "weights_block_plot",
                "cum_pel_by_patch_plot",
                "cum_pel_per_subject_plot",
                "cum_wheel_dist_plot",
                "running_pref_by_wheel_dist_plot",
                "running_pref_by_patch_plot",
                "weighted_patch_pref_plot",
            ],
            strict=True,
        ):
            entry[fig_name] = json.loads(fig.to_json())

        self.insert1(entry)


@schema
class BlockSubjectPositionPlots(dj.Computed):
    definition = """
    -> BlockSubjectAnalysis
    ---
    position_plot: longblob  # position plot (plotly)
    position_heatmap_plot: longblob  # position heatmap plot (plotly)
    position_ethogram_plot: longblob  # position ethogram plot (plotly)
    """

    class InROI(dj.Part):
        definition = """  # time spent in each ROI for each subject
        -> master
        subject_name: varchar(32)
        roi_name: varchar(32)
        ---
        in_roi_time: float  # total seconds spent in this ROI for this block
        in_roi_timestamps: longblob # timestamps when a subject is at a specific ROI
        """

    def make(self, key):
        """Compute and plot various block-level statistics and visualizations."""
        # Get some block info
        block_start, block_end = (Block & key).fetch1("block_start", "block_end")
        chunk_restriction = acquisition.create_chunk_restriction(
            key["experiment_name"], block_start, block_end
        )
        exp_patch_names = np.unique(
            (streams.UndergroundFeeder & key).fetch(
                "underground_feeder_name", order_by="underground_feeder_name"
            )
        )

        # Figure 1 - Position (centroid) over time
        # ---
        # Get animal position data
        pos_cols = {"x": "position_x", "y": "position_y", "time": "position_timestamps"}
        centroid_df = (BlockAnalysis.Subject.proj(**pos_cols)
                       & key).fetch(format="frame").reset_index()
        centroid_df.drop(columns=["experiment_name", "block_start"], inplace=True)
        centroid_df = centroid_df.explode(column=list(pos_cols))
        centroid_df.set_index("time", inplace=True)
        centroid_df = (
            centroid_df.groupby("subject_name")
            .resample("100ms")
            .first()
            .droplevel("subject_name")
            .dropna()
            .sort_index()
        )
        centroid_df["x"] = centroid_df["x"].astype(np.int32)
        centroid_df["y"] = centroid_df["y"].astype(np.int32)

        # Plot it
        position_fig = go.Figure()
        for id_i, (id_val, id_grp) in enumerate(centroid_df.groupby("subject_name")):
            norm_time = (
                (id_grp.index - id_grp.index[0]) / (id_grp.index[-1] - id_grp.index[0])
            ).values.round(3)
            colors = gen_hex_grad(subject_colors[id_i], norm_time)
            position_fig.add_trace(
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
        position_fig.update_layout(
            title="Position Tracking over Time",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
        )

        # Figure 2 - Position heatmap
        # ---
        # Calculate heatmaps
        max_x, max_y = int(centroid_df["x"].max()), int(centroid_df["y"].max())
        heatmaps = []
        for id_val, id_grp in centroid_df.groupby("subject_name"):
            # Add counts of x,y points to a grid that will be used for heatmap
            img_grid = np.zeros((max_x + 1, max_y + 1))
            points, counts = np.unique(id_grp[["x", "y"]].values, return_counts=True, axis=0)
            for point, count in zip(points, counts, strict=True):
                img_grid[point[0], point[1]] = count
            img_grid /= img_grid.max()  # normalize
            # Smooth `img_grid`
            # Mice can go ~450 cm/s, we've downsampled to 10 frames/s, we have 200 px / 1000 cm,
            # so 45 cm/frame ~= 9 px/frame
            win_sz = 9  # in pixels  (ensure odd for centering)
            kernel = np.ones((win_sz, win_sz)) / win_sz**2  # moving avg kernel
            img_grid_p = np.pad(img_grid, win_sz // 2, mode="edge")  # pad for full output from convolution
            img_grid_smooth = conv2d(img_grid_p, kernel)
            heatmaps.append((id_val, img_grid_smooth))

        # Plot em
        pos_heatmap_fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[f"Position Heatmap ({id_val})" for id_val, _ in heatmaps],
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.15,
        )
        for ax, (_, img_grid_smooth) in enumerate(heatmaps):
            heatmap_img = go.Heatmap(
                z=img_grid_smooth.T,
                zmin=0,
                zmax=(img_grid_smooth.max() / 1000),
                x=np.arange(max_x),
                y=np.arange(max_y),
                showscale=(ax == 0),
            )
            pos_heatmap_fig.add_trace(heatmap_img, row=(ax + 1), col=1)
        pos_heatmap_fig.update_layout(width=720, height=1080)

        # Figure 3 - Position ethogram
        # ---
        # Get Active Region (ROI) locations
        epoch_query = acquisition.Epoch & (acquisition.Chunk & key & chunk_restriction).proj("epoch_start")
        active_region_query = acquisition.EpochConfig.ActiveRegion & epoch_query
        roi_locs = dict(zip(*active_region_query.fetch("region_name", "region_data"), strict=True))
        # get RFID reader locations
        recent_rfid_query = (acquisition.Experiment.proj() * streams.Device.proj() & key).aggr(
            streams.RfidReader & f"rfid_reader_install_time <= '{block_start}'",
            rfid_reader_install_time="max(rfid_reader_install_time)",
        )
        rfid_location_query = (
            streams.RfidReader * streams.RfidReader.Attribute
            & recent_rfid_query
            & "attribute_name = 'Location'"
        )
        rfid_locs = dict(
            zip(*rfid_location_query.fetch("rfid_reader_name", "attribute_value"), strict=True)
        )

        ## Create position ethogram df
        arena_center_x = int(roi_locs["ArenaCenter"]["X"])
        arena_center_y = int(roi_locs["ArenaCenter"]["Y"])
        arena_center = (arena_center_x, arena_center_y)
        arena_inner_radius = int(roi_locs["ArenaInnerRadius"])
        arena_outer_radius = int(roi_locs["ArenaOuterRadius"])

        patch_radius, gate_radius = 120, 30  # in px
        rois = list(exp_patch_names) + [
            "Nest",
            "Gate",
            "Corridor",
        ]  # ROIs: patches, nest, gate, corridor
        roi_colors = plotly.colors.qualitative.Dark2
        roi_colors_dict = dict(zip(rois, roi_colors, strict=False))
        pos_eth_df = pd.DataFrame(
            columns=(["Subject"] + rois), index=centroid_df.index
        )  # df to create eth fig
        pos_eth_df["Subject"] = centroid_df["subject_name"]

        # For each ROI, compute if within ROI
        for roi in rois:
            if roi == "Corridor":  # special case for corridor, based on between inner and outer radius
                dist = np.linalg.norm(
                    (np.vstack((centroid_df["x"], centroid_df["y"])).T) - arena_center,
                    axis=1,
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
                    (np.vstack((centroid_df["x"], centroid_df["y"])).T) - (roi_x, roi_y),
                    axis=1,
                )
                pos_eth_df[roi] = dist < roi_radius

        # Melt df to a single "Loc" column that contains loc for current time (row)
        pos_eth_df = pos_eth_df.iloc[::100]  # downsample to 10s bins
        melted_df = pos_eth_df.reset_index().melt(
            id_vars=["time", "Subject"], var_name="Loc", value_name="Val"
        )
        melted_df = melted_df[melted_df["Val"]]

        # Plot it
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
            yaxis={
                "categoryorder": "total ascending",
                "categoryarray": sorted(melted_df["Subject"].unique()),
                "tickmode": "array",
                "tickvals": sorted(melted_df["Subject"].unique()),
                "ticktext": sorted(melted_df["Subject"].unique()),
            },
        )

        entry = dict(key)
        for fig, fig_name in zip(
            [position_fig, pos_heatmap_fig, pos_etho_fig],
            ["position_plot", "position_heatmap_plot", "position_ethogram_plot"],
            strict=True,
        ):
            entry[fig_name] = json.loads(fig.to_json())

        # insert into InROI
        in_roi_entries = []
        for subject_name, roi_name in itertools.product(set(melted_df.Subject), rois):
            df_ = melted_df[(melted_df["Subject"] == subject_name) & (melted_df["Loc"] == roi_name)]

            roi_timestamps = df_["time"].values
            roi_time = len(roi_timestamps) * 0.1  # 100ms per timestamp

            in_roi_entries.append(
                {**key, "subject_name": subject_name,
                 "roi_name": roi_name,
                 "in_roi_time": roi_time,
                 "in_roi_timestamps": roi_timestamps}
            )

        self.insert1(entry)
        self.InROI.insert(in_roi_entries)


# ---- Foraging Bout Analysis ----


@schema
class BlockForaging(dj.Computed):
    definition = """
    -> BlockSubjectAnalysis
    ---
    bout_count: int  # number of foraging bouts in the block
    """

    class Bout(dj.Part):
        definition = """
        -> master
        -> BlockAnalysis.Subject
        bout_start: datetime(6)
        ---
        bout_end: datetime(6)
        pellet_count: int  # number of pellets consumed during the bout
        cum_wheel_dist: float  # cumulative distance travelled during the bout
        """

    def make(self, key):
        """Compute and store foraging bouts for each subject in the block."""
        foraging_bout_df = get_foraging_bouts(key)
        foraging_bout_df.rename(
            columns={
                "subject": "subject_name",
                "start": "bout_start",
                "end": "bout_end",
                "n_pellets": "pellet_count",
                "cum_wheel_dist": "cum_wheel_dist",
            },
            inplace=True,
        )

        self.insert1({**key, "bout_count": len(foraging_bout_df)})
        self.Bout.insert({**key, **row} for _, row in foraging_bout_df.iterrows())


# ---- AnalysisNote ----


@schema
class AnalysisNote(dj.Manual):
    definition = """  # Generic table to catch all notes generated during analysis
    note_timestamp: datetime(6)
    ---
    note_type='': varchar(64)
    note: varchar(3000)
    """


# ---- Helper Functions ----


def get_threshold_associated_pellets(patch_key, start, end):
    """Retrieve the pellet delivery timestamps associated with each patch threshold update within the specified start-end time.

    1. Get all patch state update timestamps (DepletionState): let's call these events "A"
        - Remove all events within 1 second of each other
        - Remove all events without threshold value (NaN)
    2. Get all pellet delivery timestamps (DeliverPellet): let's call these events "B"
        - Find matching beam break timestamps within 1.2s after each pellet delivery
    3. For each event "A", find the nearest event "B" within 100ms before or after the event "A"
        - These are the pellet delivery events "B" associated with the previous threshold update
        event "A"
    4. Shift back the pellet delivery timestamps by 1 to match the pellet delivery with the
    previous threshold update
    5. Remove all threshold updates events "A" without a corresponding pellet delivery event "B"

    Args:
        patch_key (dict): primary key for the patch
        start (datetime): start timestamp
        end (datetime): end timestamp

    Returns:
        pd.DataFrame: DataFrame with the following columns:
        - threshold_update_timestamp (index)
        - pellet_timestamp
        - beam_break_timestamp
        - offset
        - rate
    """  # noqa 501
    chunk_restriction = acquisition.create_chunk_restriction(patch_key["experiment_name"], start, end)

    # Step 1 - fetch data
    # pellet delivery trigger
    delivered_pellet_df = fetch_stream(
        streams.UndergroundFeederDeliverPellet & patch_key & chunk_restriction
    )[start:end]
    # beambreak
    beambreak_df = fetch_stream(streams.UndergroundFeederBeamBreak & patch_key & chunk_restriction)[
        start:end
    ]
    # patch threshold
    depletion_state_df = fetch_stream(
        streams.UndergroundFeederDepletionState & patch_key & chunk_restriction
    )[start:end]
    # manual delivery
    manual_delivery_df = fetch_stream(
        streams.UndergroundFeederManualDelivery & patch_key & chunk_restriction
    )[start:end]

    # Return empty if no data
    if delivered_pellet_df.empty or beambreak_df.empty or depletion_state_df.empty:
        return acquisition.io_api._empty(
            ["threshold", "offset", "rate", "pellet_timestamp", "beam_break_timestamp"]
        )

    # Step 2 - Remove invalid rows (back-to-back events)
    BTB_MIN_TIME_DIFF = 1.2  # pellet delivery trigger - time diff is less than 1.2 seconds
    BB_MIN_TIME_DIFF = 1.0  # beambreak - time difference is less than 1 seconds
    PT_MIN_TIME_DIFF = 1.0  # patch threshold - time difference is less than 1 seconds

    invalid_rows = delivered_pellet_df.index.to_series().diff().dt.total_seconds() < BTB_MIN_TIME_DIFF
    delivered_pellet_df = delivered_pellet_df[~invalid_rows]
    # exclude manual deliveries
    delivered_pellet_df = delivered_pellet_df.loc[
        delivered_pellet_df.index.difference(manual_delivery_df.index)
    ]

    invalid_rows = beambreak_df.index.to_series().diff().dt.total_seconds() < BB_MIN_TIME_DIFF
    beambreak_df = beambreak_df[~invalid_rows]

    depletion_state_df = depletion_state_df.dropna(subset=["threshold"])
    invalid_rows = depletion_state_df.index.to_series().diff().dt.total_seconds() < PT_MIN_TIME_DIFF
    depletion_state_df = depletion_state_df[~invalid_rows]

    # Return empty if no data
    if delivered_pellet_df.empty or beambreak_df.empty or depletion_state_df.empty:
        return acquisition.io_api._empty(
            ["threshold", "offset", "rate", "pellet_timestamp", "beam_break_timestamp"]
        )

    # Step 3 - event matching
    # Find pellet delivery triggers with matching beambreaks within 1.2s after each pellet delivery
    pellet_beam_break_df = (
        pd.merge_asof(
            delivered_pellet_df.reset_index(),
            beambreak_df.reset_index().rename(columns={"time": "beam_break_timestamp"}),
            left_on="time",
            right_on="beam_break_timestamp",
            tolerance=pd.Timedelta("{BTB_MIN_TIME_DIFF}s"),
            direction="forward",
        )
        .set_index("time")
        .dropna(subset=["beam_break_timestamp"])
    )
    pellet_beam_break_df.drop_duplicates(subset="beam_break_timestamp", keep="last", inplace=True)

    # Find pellet delivery triggers that approximately coincide with each threshold update
    # i.e. nearest pellet delivery within 100ms before or after threshold update
    pellet_ts_threshold_df = (
        pd.merge_asof(
            depletion_state_df.reset_index(),
            pellet_beam_break_df.reset_index().rename(columns={"time": "pellet_timestamp"}),
            left_on="time",
            right_on="pellet_timestamp",
            tolerance=pd.Timedelta("100ms"),
            direction="nearest",
        )
        .set_index("time")
        .dropna(subset=["pellet_timestamp"])
    )

    # Clean up the df
    pellet_ts_threshold_df = pellet_ts_threshold_df.drop(columns=["event_x", "event_y"])
    # Shift back the pellet_timestamp values by 1 to match with the previous threshold update
    pellet_ts_threshold_df.pellet_timestamp = pellet_ts_threshold_df.pellet_timestamp.shift(-1)
    pellet_ts_threshold_df.beam_break_timestamp = pellet_ts_threshold_df.beam_break_timestamp.shift(-1)
    pellet_ts_threshold_df = pellet_ts_threshold_df.dropna(
        subset=["pellet_timestamp", "beam_break_timestamp"]
    )
    return pellet_ts_threshold_df


"""Foraging bout function."""


def get_foraging_bouts(
    key: dict,
    min_pellets: int = 3,
    max_inactive_time: pd.Timedelta | None = None,  # seconds
    min_wheel_movement: float = 5,  # cm
) -> pd.DataFrame:
    """Gets foraging bouts for all subjects across all patches within a block.

    Args:
        key: Block key - dict containing keys for 'experiment_name' and 'block_start'.
        min_pellets: Minimum number of pellets for a foraging bout.
        max_inactive_time: Maximum time between `min_wheel_movement`s for a foraging bout.
        min_wheel_movement: Minimum wheel movement for a foraging bout.

    Returns:
        DataFrame containing foraging bouts. Columns: duration, n_pellets, cum_wheel_dist, subject.
    """
    max_inactive_time = pd.Timedelta(seconds=60) if max_inactive_time is None else max_inactive_time
    bout_data = pd.DataFrame(columns=["start", "end", "n_pellets", "cum_wheel_dist", "subject"])
    subject_patch_data = (BlockSubjectAnalysis.Patch() & key).fetch(format="frame")
    if subject_patch_data.empty:
        return bout_data
    subject_patch_data.reset_index(level=["experiment_name"], drop=True, inplace=True)
    wheel_ts = (BlockAnalysis.Patch() & key).fetch("wheel_timestamps")[0]
    # For each subject:
    #   - Create cumulative wheel distance spun sum df combining all patches
    #     - Columns: timestamp, wheel distance, patch
    #   - Discretize into 'possible foraging events' based on `max_inactive_time`, and `min_wheel_movement`
    #       - Look ahead by `max_inactive_time` and compare with current wheel distance;
    #         if the wheel will have moved by `min_wheel_movement`, then it is a foraging event
    #       - Because we "looked ahead" (shifted), we need to readjust the start time of a foraging bout
    #       - For the foraging bout end time, we need to account for the final pellet delivery time
    #   - Filter out events with < `min_pellets`
    #   - For final events, get: duration, n_pellets, cum_wheel_distance -> add to returned DF
    for subject in subject_patch_data.index.unique("subject_name"):
        cur_subject_data = subject_patch_data.xs(subject, level="subject_name")
        n_pels = sum([arr.size for arr in cur_subject_data["pellet_timestamps"].values])
        if n_pels < min_pellets:
            continue
        # Create combined cumulative wheel distance spun
        wheel_vals = cur_subject_data["wheel_cumsum_distance_travelled"].values
        # Ensure equal length wheel_vals across patches and wheel_ts
        min_len = min(*(len(arr) for arr in wheel_vals), len(wheel_ts))
        wheel_vals = [arr[:min_len] for arr in wheel_vals]
        comb_cum_wheel_dist = np.vstack(wheel_vals).sum(axis=0)
        wheel_ts = wheel_ts[:min_len]
        # Ensure monotically increasing wheel dist
        comb_cum_wheel_dist = np.maximum.accumulate(comb_cum_wheel_dist)
        # For each wheel_ts, get the corresponding patch that was spun
        patch_spun = np.full(len(wheel_ts), "", dtype="<U20")
        patch_names = cur_subject_data.index.get_level_values(1)
        wheel_spun_thresh = 0.03  # threshold for wheel movement (cm)
        diffs = np.diff(np.stack(wheel_vals), axis=1, prepend=0)
        spun_indices = np.where(diffs > wheel_spun_thresh)
        patch_spun[spun_indices[1]] = patch_names[spun_indices[0]]
        patch_spun_df = pd.DataFrame(
            {"cum_wheel_dist": comb_cum_wheel_dist, "patch_spun": patch_spun},
            index=wheel_ts,
        )
        wheel_s_r = pd.Timedelta(wheel_ts[1] - wheel_ts[0], unit="ns")
        max_inactive_win_len = int(max_inactive_time / wheel_s_r)
        # Find times when foraging
        max_windowed_wheel_vals = patch_spun_df["cum_wheel_dist"].shift(-(max_inactive_win_len - 1)).ffill()
        foraging_mask = max_windowed_wheel_vals > (patch_spun_df["cum_wheel_dist"] + min_wheel_movement)
        # Discretize into foraging bouts
        bout_start_indxs = np.where(np.diff(foraging_mask, prepend=0) == 1)[0] + (max_inactive_win_len - 1)
        n_samples_in_1s = int(1 / wheel_s_r.total_seconds())
        bout_end_indxs = (
            np.where(np.diff(foraging_mask, prepend=0) == -1)[0]
            + (max_inactive_win_len - 1)
            + n_samples_in_1s
        )
        bout_end_indxs[-1] = min(bout_end_indxs[-1], len(wheel_ts) - 1)  # ensure last bout ends in block
        # Remove bout that starts at block end
        if bout_start_indxs[-1] >= len(wheel_ts):
            bout_start_indxs = bout_start_indxs[:-1]
            bout_end_indxs = bout_end_indxs[:-1]
        if len(bout_start_indxs) != len(bout_end_indxs):
            raise ValueError("Mismatch between the lengths of bout_start_indxs and bout_end_indxs.")
        bout_durations = (wheel_ts[bout_end_indxs] - wheel_ts[bout_start_indxs]).astype(  # in seconds
            "timedelta64[ns]"
        ).astype(float) / 1e9
        bout_starts_ends = np.array(
            [
                (wheel_ts[start_idx], wheel_ts[end_idx])
                for start_idx, end_idx in zip(bout_start_indxs, bout_end_indxs, strict=True)
            ]
        )
        all_pel_ts = np.sort(
            np.concatenate([arr for arr in cur_subject_data["pellet_timestamps"] if len(arr) > 0])
        )
        bout_pellets = np.array(
            [
                len(all_pel_ts[(all_pel_ts >= start) & (all_pel_ts <= end)])
                for start, end in bout_starts_ends
            ]
        )
        # Filter by `min_pellets`
        bout_durations = bout_durations[bout_pellets >= min_pellets]
        bout_starts_ends = bout_starts_ends[bout_pellets >= min_pellets]
        bout_pellets = bout_pellets[bout_pellets >= min_pellets]
        bout_cum_wheel_dist = np.array(
            [
                patch_spun_df.loc[end, "cum_wheel_dist"] - patch_spun_df.loc[start, "cum_wheel_dist"]
                for start, end in bout_starts_ends
            ]
        )
        # Add to returned DF
        bout_data = pd.concat(
            [
                bout_data,
                pd.DataFrame(
                    {
                        "start": bout_starts_ends[:, 0],
                        "end": bout_starts_ends[:, 1],
                        "n_pellets": bout_pellets,
                        "cum_wheel_dist": bout_cum_wheel_dist,
                        "subject": subject,
                    }
                ),
            ]
        )
    return bout_data.sort_values("start").reset_index(drop=True)
