import datajoint as dj
import pandas as pd
import numpy as np
import datetime

from .. import lab, acquisition, tracking, qc
from .. import get_schema_name, dict_to_uuid
from .visit import Visit, VisitEnd

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
class VisitSubjectPosition(dj.Computed):
    definition = """  # Animal position during a visit
    -> Visit
    -> acquisition.Chunk
    """

    class TimeSlice(dj.Part):
        definition = """
        # A short time-slice (e.g. 10 minutes) of the recording of a given animal for a visit
        -> master
        time_slice_start: datetime(6)  # datetime of the start of this time slice
        ---
        time_slice_end: datetime(6)    # datetime of the end of this time slice
        timestamps:        longblob  # (datetime) timestamps of the position data
        position_x:        longblob  # (px) animal's x-position, in the arena's coordinate frame
        position_y:        longblob  # (px) animal's y-position, in the arena's coordinate frame
        position_z=null:   longblob  # (px) animal's z-position, in the arena's coordinate frame
        """

    _time_slice_duration = datetime.timedelta(hours=0, minutes=10, seconds=0)

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
            & 'experiment_name in ("exp0.2-r0")'
        )

    def make(self, key):
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1(
            "chunk_start", "chunk_end"
        )

        # -- Determine the time to start time_slicing in this chunk
        if chunk_start < key["visit_start"] < chunk_end:
            # For chunk containing the visit_start - i.e. first chunk of this session
            start_time = key["visit_start"]
        else:
            # For chunks after the first chunk of this session
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

        assert (
            len(set((tracking.CameraTracking.Object & key).fetch("object_id"))) == 1
        ), "More than one unique object ID found - multiple animal/object mapping not yet supported"

        object_id = (tracking.CameraTracking.Object & key).fetch1("object_id")

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

        chunk_time_slices = []
        time_slice_start = start_time
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
                }
            )
            time_slice_start = time_slice_end

        self.insert1(key)
        self.TimeSlice.insert(chunk_time_slices)

    @classmethod
    def get_position(cls, visit_key):
        """
        Given a key to a single Visit, return a Pandas DataFrame for the position data
        of the subject for the specified Visit time period
        """
        assert len(Visit & visit_key) == 1

        start, end = (
            Visit.join(VisitEnd, left=True).proj(visit_end="IFNULL(visit_end, NOW())")
            & visit_key
        ).fetch1("visit_start", "visit_end")

        return tracking._get_position(
            cls.TimeSlice,
            object_attr="subject",
            object_name=visit_key["subject"],
            start_attr="time_slice_start",
            end_attr="time_slice_end",
            start=start,
            end=end,
            fetch_attrs=["timestamps", "position_x", "position_y"],
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
    time_fraction_in_corridor: float  # fraction of time the animal spent in the corridor in this session
    in_corridor: longblob             # array of indices for when the animal is in the corridor (index into the position data)
    time_fraction_visit: float        # fraction of time the animal spent in the arena in this session
    in_arena: longblob                # array of indices for when the animal is in the arena (index into the position data)
    """

    class Nest(dj.Part):
        definition = """  # Time spent in nest
        -> master
        -> lab.ArenaNest
        ---
        time_fraction_in_nest: float  # fraction of time the animal spent in this nest in this session
        in_nest: longblob             # array of indices for when the animal is in this nest (index into the position data)
        """

    class FoodPatch(dj.Part):
        definition = """ # Time spent in food patch
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        in_patch: longblob             # array of indices for when the animal is in this patch (index into the position data)
        """


@schema
class VisitSummary(dj.Computed):
    definition = """
    -> Visit
    visit_date: date
    ---
    total_distance_travelled: float  # (m) total distance the animal travelled during this session
    total_pellet_count: int  # total pellet delivered (triggered) for all patches during this session
    total_wheel_distance_travelled: float  # total wheel travelled distance for all patches
    change_in_weight: float  # weight change before/after the session
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        pellet_count: int  # number of pellets being delivered (triggered) by this patch during this session
        wheel_distance_travelled: float  # wheel travelled distance during this session for this patch
        """
