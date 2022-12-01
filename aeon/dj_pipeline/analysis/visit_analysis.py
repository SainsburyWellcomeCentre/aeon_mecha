import datetime
from collections import deque
from datetime import time

import datajoint as dj
import numpy as np
import pandas as pd

from .. import acquisition, dict_to_uuid, get_schema_name, lab, qc, tracking
from .visit import Visit, VisitEnd

logger = dj.logger
schema = dj.schema(get_schema_name("analysis"))


# ---------- Position Filtering Method ------------------


@schema
class PositionFilteringMethod(dj.Lookup):
    definition = """
    pos_filter_method: varchar(16)  
    ---
    pos_filter_method_description: varchar(256)
    """

    contents = [("Kalman", "Online DeepLabCut as part of Bonsai workflow")]


@schema
class PositionFilteringParamSet(dj.Lookup):
    definition = """  # Parameter set used in a particular PositionFilteringMethod
    pos_filter_paramset_id:  smallint
    ---
    -> PositionFilteringMethod    
    paramset_description: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """


# ---------- Animal Position per Visit ------------------


@schema
class AnimalObjectMapping(dj.Manual):
    definition = """
    -> Visit
    ---
    object_id: int
    """


@schema
class VisitSubjectPosition(dj.Computed):
    definition = """  # Animal position during a visit
    -> Visit
    -> acquisition.Chunk
    """

    class TimeSlice(dj.Part):
        definition = """
        # A short time-slice (e.g. 10 minutes) of the recording of a given animal for a visit
        -> master
        time_slice_start: datetime(6)   # datetime of the start of this time slice
        ---
        time_slice_end: datetime(6)     # datetime of the end of this time slice
        timestamps:        longblob     # (datetime) timestamps of the position data
        position_x:        longblob     # (px) animal's x-position, in the arena's coordinate frame
        position_y:        longblob     # (px) animal's y-position, in the arena's coordinate frame
        position_z=null:   longblob     # (px) animal's z-position, in the arena's coordinate frame
        area=null:         longblob     # (px^2) animal's size detected in the camera
        """

    _time_slice_duration = np.timedelta64(10, "m")

    @property
    def key_source(self):
        """
        Chunk for all visits:
        + visit_start during this Chunk - i.e. first chunk of the visit
        + visit_end during this Chunk - i.e. last chunk of the visit
        + chunk starts after visit_start and ends before visit_end (or NOW() - i.e. ongoing visits)
        """
        return (
            Visit.join(VisitEnd, left=True).proj(visit_end="IFNULL(visit_end, NOW())")
            * acquisition.Chunk
            & acquisition.SubjectEnterExit
            & [
                "visit_start BETWEEN chunk_start AND chunk_end",
                "visit_end BETWEEN chunk_start AND chunk_end",
                "chunk_start >= visit_start AND chunk_end <= visit_end",
            ]
            & "chunk_start < chunk_end"  # in some chunks, end timestamp comes before start (timestamp error)
        )

    def make(self, key):
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end"
        )

        # -- Determine the time to start time_slicing in this chunk
        if chunk_start < key["visit_start"] < chunk_end:
            # For chunk containing the visit_start - i.e. first chunk of this visit
            start_time = key["visit_start"]
        else:
            # For chunks after the first chunk of this visit
            start_time = chunk_start

        # -- Determine the time to end time_slicing in this chunk
        if VisitEnd & key:  # finished visit
            visit_end = (VisitEnd & key).fetch1("visit_end")
            end_time = min(chunk_end, visit_end)
        else:  # ongoing visit
            # get the enter/exit events in this chunk that are after the visit_start
            next_enter_exit_events = (
                acquisition.SubjectEnterExit.Time * acquisition.EventType
                & key
                & f'enter_exit_time > "{key["visit_start"]}"'
            )
            if not next_enter_exit_events:
                # No enter/exit event: time_slices from this whole chunk
                end_time = chunk_end
            else:
                next_event = next_enter_exit_events.fetch(
                    as_dict=True, order_by="enter_exit_time DESC", limit=1
                )[0]
                if next_event["event_type"] == "SubjectEnteredArena":
                    raise ValueError(f"Bad Visit - never exited visit")
                end_time = next_event["enter_exit_time"]

        # -- Retrieve position data
        camera_name = acquisition._ref_device_mapping[key["experiment_name"]]

        if len(set((tracking.CameraTracking.Object & key).fetch("object_id"))) == 1:
            object_id = (tracking.CameraTracking.Object & key).fetch1("object_id")
        else:
            logger.info(
                '"More than one unique object ID found - using animal/object mapping from AnimalObjectMapping"'
            )
            if not (AnimalObjectMapping & key):
                raise ValueError(
                    f"More than one unique object ID found for {key} - no AnimalObjectMapping defined"
                )
            object_id = (AnimalObjectMapping & key).fetch1("object_id")

        positiondata = tracking.CameraTracking.get_object_position(
            experiment_name=key["experiment_name"],
            camera_name=camera_name,
            object_id=object_id,
            start=chunk_start,
            end=chunk_end,
        )

        if not len(positiondata):
            raise ValueError(f"No position data between {chunk_start} and {chunk_end}")

        timestamps = positiondata.index.values
        x = positiondata.position_x.values
        y = positiondata.position_y.values
        z = np.full_like(x, 0.0)
        area = positiondata.area.values

        chunk_time_slices = []
        time_slice_start = np.array(start_time, dtype="datetime64[ns]")
        end_time = np.array(end_time, dtype="datetime64[ns]")

        while time_slice_start < end_time:
            time_slice_end = time_slice_start + min(
                self._time_slice_duration, end_time - time_slice_start
            )
            in_time_slice = np.logical_and(
                timestamps >= time_slice_start, timestamps < time_slice_end
            )
            chunk_time_slices.append(
                {
                    **key,
                    "time_slice_start": time_slice_start,
                    "time_slice_end": time_slice_end,
                    "timestamps": timestamps[in_time_slice],
                    "position_x": x[in_time_slice],
                    "position_y": y[in_time_slice],
                    "position_z": z[in_time_slice],
                    "area": area[in_time_slice],
                }
            )
            time_slice_start = time_slice_end

        self.insert1(key)
        self.TimeSlice.insert(chunk_time_slices)

    @classmethod
    def get_position(cls, visit_key=None, subject=None, start=None, end=None):
        """
        Given a key to a single Visit, return a Pandas DataFrame for the position data
        of the subject for the specified Visit time period
        """
        if visit_key is not None:
            assert len(Visit & visit_key) == 1
            start, end = (
                Visit.join(VisitEnd, left=True).proj(
                    visit_end="IFNULL(visit_end, NOW())"
                )
                & visit_key
            ).fetch1("visit_start", "visit_end")
            subject = visit_key["subject"]
        elif all((subject, start, end)):
            start = start
            end = end
            subject = subject
        else:
            raise ValueError(
                f'Either "visit_key" or all three "subject", "start" and "end" has to be specified'
            )

        return tracking._get_position(
            cls.TimeSlice,
            object_attr="subject",
            object_name=subject,
            start_attr="time_slice_start",
            end_attr="time_slice_end",
            start=start,
            end=end,
            fetch_attrs=["timestamps", "position_x", "position_y", "area"],
            attrs_to_scale=["position_x", "position_y"],
            scale_factor=tracking.pixel_scale,
        )


# -------------- Visit-level analysis ---------------------


@schema
class VisitTimeDistribution(dj.Computed):
    definition = """
    -> Visit
    visit_date: date
    ---
    day_duration: float               # total duration (in hours)
    time_fraction_in_corridor: float  # fraction of time the animal spent in the corridor in this visit
    in_corridor: longblob             # array of timestamps for when the animal is in the corridor 
    time_fraction_in_arena: float     # fraction of time the animal spent in the arena in this visit
    in_arena: longblob                # array of timestamps for when the animal is in the arena 
    """

    class Nest(dj.Part):
        definition = """  # Time spent in nest
        -> master
        -> lab.ArenaNest
        ---
        time_fraction_in_nest: float  # fraction of time the animal spent in this nest in this visit
        in_nest: longblob             # array of indices for when the animal is in this nest (index into the position data)
        """

    class FoodPatch(dj.Part):
        definition = """ # Time spent in food patch
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this visit
        in_patch: longblob             # array of timestamps for when the animal is in this patch 
        """

    # Work on finished visits
    key_source = Visit & (
        VisitEnd * VisitSubjectPosition.TimeSlice & "time_slice_end = visit_end"
    )

    def make(self, key):
        visit_start, visit_end = (VisitEnd & key).fetch1("visit_start", "visit_end")
        visit_dates = pd.date_range(
            start=pd.Timestamp(visit_start.date()), end=pd.Timestamp(visit_end.date())
        )
        maintenance_period = get_maintenance_periods(
            key["experiment_name"], visit_start, visit_end
        )

        for visit_date in visit_dates:
            day_start = datetime.datetime.combine(visit_date.date(), time.min)
            day_end = datetime.datetime.combine(
                visit_date.date() + datetime.timedelta(days=1), time.min
            )  # start of the next day
            day_start = max(day_start, visit_start)
            day_end = min(day_end, visit_end)

            # duration of the visit on the date
            day_duration = round(
                (day_end - day_start) / datetime.timedelta(hours=1),
                3,
            )
            # subject's position data in the time_slices per day
            position = VisitSubjectPosition.get_position(
                subject=key["subject"], start=day_start, end=day_end
            )
            # filter out maintenance period based on logs
            position = filter_out_maintenance_periods(
                position, maintenance_period, day_end
            )

            # filter for objects of the correct size
            valid_position = (position.area > 0) & (position.area < 1000)
            position[~valid_position] = np.nan
            position.rename(
                columns={"position_x": "x", "position_y": "y"}, inplace=True
            )
            # in corridor
            distance_from_center = tracking.compute_distance(
                position[["x", "y"]],
                (tracking.arena_center_x, tracking.arena_center_y),
            )
            in_corridor = (distance_from_center < tracking.arena_outer_radius) & (
                distance_from_center > tracking.arena_inner_radius
            )
            in_arena = ~in_corridor
            # in nests - loop through all nests in this experiment
            in_nest_times = []
            for nest_key in (lab.ArenaNest & key).fetch("KEY"):
                in_nest = tracking.is_position_in_nest(position, nest_key)
                in_nest_times.append(
                    {
                        **key,
                        **nest_key,
                        "visit_date": visit_date.date(),
                        "time_fraction_in_nest": in_nest.mean(),
                        "in_nest": in_nest.index.values[in_nest],
                    }
                )
                in_arena = in_arena & ~in_nest
            # in food patches - loop through all in-use patches during this visit
            query = acquisition.ExperimentFoodPatch.join(
                acquisition.ExperimentFoodPatch.RemovalTime, left=True
            )
            food_patch_keys = (
                query
                & (
                    VisitSubjectPosition
                    * acquisition.ExperimentFoodPatch.join(
                        acquisition.ExperimentFoodPatch.RemovalTime, left=True
                    )
                    & key
                    & f'"{day_start}" >= food_patch_install_time'
                    & f'"{day_end}" < IFNULL(food_patch_remove_time, "2200-01-01")'
                ).fetch("KEY")
            ).fetch("KEY")

            in_food_patch_times = []
            for food_patch_key in food_patch_keys:
                # wheel data
                food_patch_description = (
                    acquisition.ExperimentFoodPatch & food_patch_key
                ).fetch1("food_patch_description")
                wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
                    experiment_name=key["experiment_name"],
                    start=pd.Timestamp(day_start),
                    end=pd.Timestamp(day_end),
                    patch_name=food_patch_description,
                    using_aeon_io=True,
                )
                # filter out maintenance period based on logs
                wheel_data = filter_out_maintenance_periods(
                    wheel_data, maintenance_period, day_end
                )
                patch_position = (
                    acquisition.ExperimentFoodPatch.Position & food_patch_key
                ).fetch1("food_patch_position_x", "food_patch_position_y")
                in_patch = tracking.is_position_in_patch(
                    position,
                    patch_position,
                    wheel_data.distance_travelled,
                    patch_radius=0.2,
                )
                in_food_patch_times.append(
                    {
                        **key,
                        **food_patch_key,
                        "visit_date": visit_date.date(),
                        "time_fraction_in_patch": in_patch.mean(),
                        "in_patch": in_patch.index.values[in_patch],
                    }
                )
                in_arena = in_arena & ~in_patch

            self.insert1(
                {
                    **key,
                    "visit_date": visit_date.date(),
                    "day_duration": day_duration,
                    "time_fraction_in_corridor": in_corridor.mean(),
                    "in_corridor": in_corridor.index.values[in_corridor],
                    "time_fraction_in_arena": in_arena.mean(),
                    "in_arena": in_arena.index.values[in_arena],
                }
            )
            self.Nest.insert(in_nest_times)
            self.FoodPatch.insert(in_food_patch_times)


@schema
class VisitSummary(dj.Computed):
    definition = """
    -> Visit
    visit_date: date
    ---
    day_duration: float                     # total duration (in hours)
    total_distance_travelled: float         # (m) total distance the animal travelled during this visit
    total_pellet_count: int                 # total pellet delivered (triggered) for all patches during this visit
    total_wheel_distance_travelled: float   # total wheel travelled distance for all patches
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        pellet_count: int                   # number of pellets being delivered (triggered) by this patch during this visit
        wheel_distance_travelled: float     # wheel travelled distance during this visit for this patch
        """

    # Work on finished visits
    key_source = Visit & (
        VisitEnd * VisitSubjectPosition.TimeSlice & "time_slice_end = visit_end"
    )

    def make(self, key):
        visit_start, visit_end = (VisitEnd & key).fetch1("visit_start", "visit_end")
        visit_dates = pd.date_range(
            start=pd.Timestamp(visit_start.date()), end=pd.Timestamp(visit_end.date())
        )
        maintenance_period = get_maintenance_periods(
            key["experiment_name"], visit_start, visit_end
        )

        for visit_date in visit_dates:
            day_start = datetime.datetime.combine(visit_date.date(), time.min)
            day_end = datetime.datetime.combine(
                visit_date.date() + datetime.timedelta(days=1), time.min
            )  # start of the next day

            day_start = max(day_start, visit_start)
            day_end = min(day_end, visit_end)

            # duration of the visit on the date
            day_duration = round(
                (day_end - day_start) / datetime.timedelta(hours=1),
                3,
            )
            # subject's position data in the time_slices per day
            position = VisitSubjectPosition.get_position(
                subject=key["subject"], start=day_start, end=day_end
            )
            # filter out maintenance period based on logs
            position = filter_out_maintenance_periods(
                position, maintenance_period, day_end
            )
            # filter for objects of the correct size
            valid_position = (position.area > 0) & (position.area < 1000)
            position[~valid_position] = np.nan
            position.rename(
                columns={"position_x": "x", "position_y": "y"}, inplace=True
            )
            position_diff = np.sqrt(
                np.square(np.diff(position.x)) + np.square(np.diff(position.y))
            )
            total_distance_travelled = np.nansum(position_diff)

            # in food patches - loop through all in-use patches during this visit
            query = acquisition.ExperimentFoodPatch.join(
                acquisition.ExperimentFoodPatch.RemovalTime, left=True
            )
            food_patch_keys = (
                query
                & (
                    VisitSubjectPosition
                    * acquisition.ExperimentFoodPatch.join(
                        acquisition.ExperimentFoodPatch.RemovalTime, left=True
                    )
                    & key
                    & f'"{day_start}" >= food_patch_install_time'
                    & f'"{day_end}" < IFNULL(food_patch_remove_time, "2200-01-01")'
                ).fetch("KEY")
            ).fetch("KEY")

            food_patch_statistics = []
            for food_patch_key in food_patch_keys:
                pellet_events = (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{day_start}" AND "{day_end}"'
                ).fetch("event_time")
                # filter out maintenance period based on logs
                pellet_events = filter_out_maintenance_periods(
                    pd.DataFrame(pellet_events).set_index(0),
                    maintenance_period,
                    day_end,
                    dropna=True,
                ).index.values
                # wheel data
                food_patch_description = (
                    acquisition.ExperimentFoodPatch & food_patch_key
                ).fetch1("food_patch_description")
                wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
                    experiment_name=key["experiment_name"],
                    start=pd.Timestamp(day_start),
                    end=pd.Timestamp(day_end),
                    patch_name=food_patch_description,
                    using_aeon_io=True,
                )
                # filter out maintenance period based on logs
                wheel_data = filter_out_maintenance_periods(
                    wheel_data, maintenance_period, day_end
                )

                food_patch_statistics.append(
                    {
                        **key,
                        **food_patch_key,
                        "visit_date": visit_date.date(),
                        "pellet_count": len(pellet_events),
                        "wheel_distance_travelled": wheel_data.distance_travelled.values[
                            -1
                        ],
                    }
                )

            total_pellet_count = np.sum(
                [p["pellet_count"] for p in food_patch_statistics]
            )
            total_wheel_distance_travelled = np.sum(
                [p["wheel_distance_travelled"] for p in food_patch_statistics]
            )

            self.insert1(
                {
                    **key,
                    "visit_date": visit_date.date(),
                    "day_duration": day_duration,
                    "total_pellet_count": total_pellet_count,
                    "total_wheel_distance_travelled": total_wheel_distance_travelled,
                    # "change_in_weight": weight_end - weight_start,
                    "total_distance_travelled": total_distance_travelled,
                }
            )
            self.FoodPatch.insert(food_patch_statistics)


@schema
class VisitForagingBout(dj.Computed):
    definition = """ # A time period spanning the time when the animal enters a food patch and moves the wheel to when it leaves the food patch
    -> Visit
    -> acquisition.ExperimentFoodPatch
    bout_start: datetime(6)                    # start time of bout
    --- 
    bout_end: datetime(6)                      # end time of bout
    bout_duration: float                       # (seconds)
    wheel_distance_travelled: float            # (cm)
    pellet_count: int                          # number of pellet trigger events
    """

    # Work on 24/7 experiments
    key_source = (
        Visit
        & VisitSummary
        & (VisitEnd & f"visit_duration > 24")
        & f"experiment_name= 'exp0.2-r0'"
    ) * acquisition.ExperimentFoodPatch

    def make(self, key):
        visit_start, visit_end = (VisitEnd & key).fetch1("visit_start", "visit_end")

        # get in_patch timestamps
        food_patch_description = (acquisition.ExperimentFoodPatch & key).fetch1(
            "food_patch_description"
        )
        in_patch_times = np.concatenate(
            (
                VisitTimeDistribution.FoodPatch * acquisition.ExperimentFoodPatch & key
            ).fetch("in_patch", order_by="visit_date")
        )
        maintenance_period = get_maintenance_periods(
            key["experiment_name"], visit_start, visit_end
        )
        in_patch_times = filter_out_maintenance_periods(
            pd.DataFrame(
                [[food_patch_description]] * len(in_patch_times),
                columns=["region"],
                index=in_patch_times,
            ),
            maintenance_period,
            visit_end,
            True,
        ).index.values

        # get pellet data
        patch = (
            (
                dj.U("event_time", "event_type")
                & (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & key
                    & f'event_time BETWEEN "{visit_start}" AND "{visit_end}"'
                    & 'event_type = "TriggerPellet"'
                )
            )
            .fetch(order_by="event_time", format="frame")
            .reset_index()
            .set_index("event_time")
        )
        # TODO: handle multiple retries of pellet delivery
        maintenance_period = get_maintenance_periods(
            key["experiment_name"], visit_start, visit_end
        )
        patch = filter_out_maintenance_periods(
            patch, maintenance_period, visit_end, True
        )

        if len(in_patch_times):
            change_ind = (
                np.where((np.diff(in_patch_times) / 1e6) > np.timedelta64(20))[0] + 1
            )  # timestamp index where state changes

            for i in range(len(change_ind) + 1):
                if i == 0:
                    ts_array = in_patch_times[: change_ind[i]]
                elif i == len(change_ind):
                    ts_array = in_patch_times[change_ind[i - 1] :]
                else:
                    ts_array = in_patch_times[change_ind[i - 1] : change_ind[i]]

                wheel_start, wheel_end = ts_array[0], ts_array[-1]
                if wheel_start > wheel_end:  # skip if timestamps were misaligned
                    continue

                wheel_data = acquisition.FoodPatchWheel.get_wheel_data(
                    experiment_name=key["experiment_name"],
                    start=wheel_start,
                    end=wheel_end,
                    patch_name=food_patch_description,
                    using_aeon_io=True,
                )
                maintenance_period = get_maintenance_periods(
                    key["experiment_name"], visit_start, visit_end
                )
                wheel_data = filter_out_maintenance_periods(
                    wheel_data, maintenance_period, visit_end, True
                )
                self.insert1(
                    {
                        **key,
                        "bout_start": ts_array[0],
                        "bout_end": ts_array[-1],
                        "bout_duration": (ts_array[-1] - ts_array[0])
                        / np.timedelta64(1, "s"),
                        "wheel_distance_travelled": wheel_data.distance_travelled[-1],
                        "pellet_count": len(patch.loc[wheel_start:wheel_end]),
                    }
                )


def get_maintenance_periods(experiment_name, start, end):
    # get logs from acquisition.ExperimentLog
    log_df = (
        acquisition.ExperimentLog.Message.proj("message")
        & {"experiment_name": experiment_name}
        & 'message IN ("Maintenance", "Experiment")'
        & f'message_time BETWEEN "{start}" AND "{end}"'
    )
    log_df = log_df.fetch(format="frame", order_by="message_time").reset_index()
    log_df = log_df[log_df["message"].shift() != log_df["message"]].reset_index(
        drop=True
    )  # remove duplicates and keep the first one

    # An experiment starts with visit start (anything before the first maintenance is experiment)
    # Delete the row if it starts with "Experiment"
    if log_df.iloc[0]["message"] == "Experiment":
        log_df.drop(index=0, inplace=True)  # look for the first maintenance

    # Last entry is the visit end
    if log_df.iloc[-1]["message"] == "Maintenance":
        log_df_end = log_df.tail(1)
        log_df_end["message_time"], log_df_end["message"] = (
            pd.Timestamp(end),
            "VisitEnd",
        )
        log_df = pd.concat([log_df, log_df_end])
        log_df.reset_index(drop=True, inplace=True)

    start_timestamps = log_df.loc[
        log_df["message"] == "Maintenance", "message_time"
    ].values
    end_timestamps = log_df.loc[
        log_df["message"] != "Maintenance", "message_time"
    ].values

    return deque(
        [
            (pd.Timestamp(start), pd.Timestamp(end))
            for start, end in zip(start_timestamps, end_timestamps)
        ]
    )  # queue object. pop out from left after use


def filter_out_maintenance_periods(data_df, maintenance_period, end_time, dropna=False):
    while maintenance_period:
        (maintenance_start, maintenance_end) = maintenance_period[0]
        if end_time < maintenance_start:  # no more maintenance for this date
            break
        maintenance_filter = (data_df.index >= maintenance_start) & (
            data_df.index <= maintenance_end
        )
        data_df[maintenance_filter] = np.nan
        if end_time >= maintenance_end:  # remove this range
            maintenance_period.popleft()
        else:
            break
    if dropna:
        data_df.dropna(inplace=True)
    return data_df
