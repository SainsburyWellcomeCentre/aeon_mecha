import datajoint as dj
import pandas as pd
import numpy as np
import datetime

from aeon.io import api as aeon_api
from aeon.util import utils as aeon_utils

from .. import lab, acquisition, tracking, qc
from .. import get_schema_name

schema = dj.schema(get_schema_name('analysis'))

__all__ = ['schema', 'InArena', 'NeverExitedArena',
           'InArenaEnd', 'InArenaTimeSlice', 'InArenaSubjectPosition',
           'InArenaTimeDistribution', 'InArenaSummary', 'InArenaRewardRate']

# ------------------- SESSION --------------------


@schema
class InArena(dj.Computed):
    definition = """  # A time period spanning the time when the animal first enters the arena to when it exits the arena
    -> acquisition.Experiment.Subject
    in_arena_start: datetime(6)
    ---
    -> [nullable] acquisition.TaskProtocol
    """

    @property
    def key_source(self):
        return dj.U("experiment_name", "subject", "in_arena_start") & (
            acquisition.SubjectEnterExit.Time * acquisition.EventType
            & 'event_type = "SubjectEnteredArena"'
        ).proj(in_arena_start="enter_exit_time") & {'experiment_name': 'exp0.1-r0'}

    def make(self, key):
        self.insert1(key)


@schema
class NeverExitedArena(dj.Manual):
    definition = """  # Bad InArena where the animal seemed to have never exited
    -> InArena
    """


@schema
class InArenaEnd(dj.Computed):
    definition = """
    -> InArena
    ---
    in_arena_end: datetime(6)
    in_arena_duration: float  # (hour)
    """

    key_source = InArena - NeverExitedArena & (
            InArena.proj() * acquisition.SubjectEnterExit.Time * acquisition.EventType
            & 'event_type = "SubjectExitedArena"'
            & "enter_exit_time > in_arena_start"
    )

    def make(self, key):
        in_arena_start = key["in_arena_start"]
        subject_exit = (
            acquisition.SubjectEnterExit.Time
            & {"subject": key["subject"]}
            & f'enter_exit_time > "{in_arena_start}"'
        ).fetch(as_dict=True, limit=1, order_by="enter_exit_time ASC")[0]

        if subject_exit["event_type"] != "SubjectExitedArena":
            NeverExitedArena.insert1(key, skip_duplicates=True)
            return

        in_arena_end = subject_exit["enter_exit_time"]
        duration = (in_arena_end - in_arena_start).total_seconds() / 3600

        # insert
        self.insert1({**key, "in_arena_end": in_arena_end, "in_arena_duration": duration})


# ------------------- TIMESLICE --------------------


@schema
class InArenaTimeSlice(dj.Computed):
    definition = """
    # A short time-slice (e.g. 10 minutes) of the recording of a given animal in the arena
    -> InArena
    -> acquisition.Chunk
    time_slice_start: datetime(6)  # datetime of the start of this time slice
    ---
    time_slice_end: datetime(6)    # datetime of the end of this time slice
    """

    @property
    def key_source(self):
        """
        Chunk for all sessions:
        + are not "NeverExitedSession"
        + in_arena_start during this Chunk - i.e. first chunk of the session
        + in_arena_end during this Chunk - i.e. last chunk of the session
        + chunk starts after in_arena_start and ends before in_arena_end (or NOW() - i.e. session still on going)
        """
        return (
                InArena.join(InArenaEnd, left=True).proj(
                    in_arena_end="IFNULL(in_arena_end, NOW())"
                )
                * acquisition.Chunk
                - NeverExitedArena
                & acquisition.SubjectEnterExit
                & [
                    "in_arena_start BETWEEN chunk_start AND chunk_end",
                    "in_arena_end BETWEEN chunk_start AND chunk_end",
                    "chunk_start >= in_arena_start AND chunk_end <= in_arena_end",
                ]
        )

    _time_slice_duration = datetime.timedelta(hours=0, minutes=10, seconds=0)

    def make(self, key):
        chunk_start, chunk_end = (acquisition.Chunk & key).fetch1("chunk_start", "chunk_end")

        # -- Determine the time to start time_slicing in this chunk
        if chunk_start < key["in_arena_start"] < chunk_end:
            # For chunk containing the in_arena_start - i.e. first chunk of this session
            start_time = key["in_arena_start"]
        else:
            # For chunks after the first chunk of this session
            start_time = chunk_start

        # -- Determine the time to end time_slicing in this chunk
        # get the enter/exit events in this chunk that are after the in_arena_start
        next_enter_exit_events = (
            acquisition.SubjectEnterExit.Time * acquisition.EventType
            & key & f'enter_exit_time > "{key["in_arena_start"]}"'
        )
        if not next_enter_exit_events:
            # No enter/exit event: time_slices from this whole chunk
            end_time = chunk_end
        else:
            next_event = next_enter_exit_events.fetch(
                as_dict=True, order_by="enter_exit_time DESC", limit=1
            )[0]
            if next_event["event_type"] == "SubjectEnteredArena":
                NeverExitedArena.insert1(
                    key, ignore_extra_fields=True, skip_duplicates=True
                )
                return
            end_time = next_event["enter_exit_time"]

        chunk_time_slices = []
        time_slice_start = start_time
        while time_slice_start < end_time:
            time_slice_end = time_slice_start + min(
                self._time_slice_duration, end_time - time_slice_start
            )
            chunk_time_slices.append(
                {
                    **key,
                    "time_slice_start": time_slice_start,
                    "time_slice_end": time_slice_end,
                }
            )
            time_slice_start = time_slice_end

        self.insert(chunk_time_slices)


# ---------- Subject Position ------------------


@schema
class InArenaSubjectPosition(dj.Imported):
    definition = """
    -> InArenaTimeSlice
    ---
    timestamps:        longblob  # (datetime) timestamps of the position data
    position_x:        longblob  # (px) animal's x-position, in the arena's coordinate frame
    position_y:        longblob  # (px) animal's y-position, in the arena's coordinate frame
    position_z=null:   longblob  # (px) animal's z-position, in the arena's coordinate frame
    area=null:         longblob  # (px^2) animal's size detected in the camera
    speed=null:        longblob  # (px/s) speed
    """

    key_source = InArenaTimeSlice & (qc.CameraQC * acquisition.ExperimentCamera
                                     & 'camera_description = "FrameTop"')

    def make(self, key):
        """
        The ingest logic here relies on the assumption that there is only one subject in the arena at a time
        The positiondata is associated with that one subject currently in the arena at any timepoints
        For multi-animal experiments, a mapping of object_id-to-subject is needed to ingest the right position data
        associated with a particular animal
        """
        time_slice_start, time_slice_end = (InArenaTimeSlice & key).fetch1('time_slice_start', 'time_slice_end')

        positiondata = tracking.CameraTracking.get_object_position(
            experiment_name=key['experiment_name'],
            object_id=-1,
            start=time_slice_start,
            end=time_slice_end
        )

        if not len(positiondata):
            raise ValueError(f'No position data between {time_slice_start} and {time_slice_end}')

        timestamps = positiondata.index.to_pydatetime()
        x = positiondata.position_x.values
        y = positiondata.position_y.values
        z = np.full_like(x, 0.0)
        area = positiondata.area.values

        # speed - TODO: confirm with aeon team if this calculation is sufficient (any smoothing needed?)
        position_diff = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)) + np.square(np.diff(z)))
        time_diff = [t.total_seconds() for t in np.diff(timestamps)]
        speed = position_diff / time_diff
        speed = np.hstack((speed[0], speed))

        self.insert1({**key,
                      'timestamps': timestamps,
                      'position_x': x,
                      'position_y': y,
                      'position_z': z,
                      'area': area,
                      'speed': speed})

    @classmethod
    def get_position(cls, in_arena_key):
        """
        Given a key to a single InArena, return a Pandas DataFrame for the position data
        of the subject for the specified InArena time period
        """
        assert len(InArena & in_arena_key) == 1

        start, end = (InArena * InArenaEnd & in_arena_key).fetch1(
            'in_arena_start', 'in_arena_end')

        return tracking._get_position(
            cls * InArenaTimeSlice.proj('time_slice_end'),
            object_attr='subject', object_name=in_arena_key['subject'],
            start_attr='time_slice_start', end_attr='time_slice_end',
            start=start, end=end,
            fetch_attrs=['timestamps', 'position_x', 'position_y', 'speed', 'area'],
            attrs_to_scale=['position_x', 'position_y', 'speed'],
            scale_factor=tracking.pixel_scale)


# -------------- InArena-level Quality Control ---------------------


@schema
class InArenaQC(dj.Computed):
    definition = """  # Quality control performed on each InArena period
    -> InArena
    """

    class Routine(dj.Part):
        definition = """  # Quality control routine performed on each InArena period
        -> master
        -> qc.QCRoutine
        ---
        -> qc.QCCode
        qc_comment: varchar(255)  
        """

    class BadPeriod(dj.Part):
        definition = """
        -> master.Routine
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """

    def make(self, key):
        # depending on which qc_routine
        # fetch relevant data from upstream
        # import the qc_function from the qc_module
        # call the qc_function - expecting a qc_code back, and a list of bad-periods
        # store qc results
        pass


@schema
class BadInArena(dj.Computed):
    definition = """  # InArena period labelled as BadInArena and excluded from further analysis
    -> InArena
    ---
    comment='': varchar(255)  # any comments for why this is a bad InArena time period - e.g. manually flagged
    """


@schema
class GoodInArena(dj.Computed):
    definition = """  #  InArena determined to be good from quality control assessment
    -> InArenaQC
    ---
    qc_routines: varchar(255)  # concatenated list of all the QC routines used for this good/bad conclusion
    """

    class BadPeriod(dj.Part):
        definition = """
        -> master
        bad_period_start: datetime(6)
        ---
        bad_period_end: datetime(6)
        """

    def make(self, key):
        # aggregate all SessionQC results for this session
        # determine Good or Bad Session
        # insert BadPeriod (if none inserted, no bad period)
        pass


# -------------- InArena-level analysis ---------------------


@schema
class InArenaTimeDistribution(dj.Computed):
    definition = """
    -> InArena
    ---
    time_fraction_in_corridor: float  # fraction of time the animal spent in the corridor in this session
    in_corridor: longblob             # array of boolean for if the animal is in the corridor (same length as position data)
    time_fraction_in_arena: float     # fraction of time the animal spent in the arena in this session
    in_arena: longblob                # array of boolean for if the animal is in the arena (same length as position data)
    """

    class Nest(dj.Part):
        definition = """  # Time spent in nest
        -> master
        -> lab.ArenaNest
        ---
        time_fraction_in_nest: float  # fraction of time the animal spent in this nest in this session
        in_nest: longblob             # array of boolean for if the animal is in this nest (same length as position data)
        """

    class FoodPatch(dj.Part):
        definition = """ # Time spent in food patch
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        time_fraction_in_patch: float  # fraction of time the animal spent on this patch in this session
        in_patch: longblob             # array of boolean for if the animal is in this patch (same length as position data)
        """

    # Work on finished Session with TimeSlice and SubjectPosition fully populated only
    key_source = (InArena
                  & (InArena * InArenaEnd * InArenaTimeSlice
                     & 'time_slice_end = in_arena_end').proj()
                  & (InArena.aggr(InArenaTimeSlice, time_slice_count='count(time_slice_start)')
                     * InArena.aggr(InArenaSubjectPosition, tracking_count='count(time_slice_start)')
                     & 'time_slice_count = tracking_count'))

    def make(self, key):
        raw_data_dir = acquisition.Experiment.get_data_directory(key)
        in_arena_start, in_arena_end = (InArena * InArenaEnd & key).fetch1(
            'in_arena_start', 'in_arena_end')

        # subject's position data in the time_slices
        position = InArenaSubjectPosition.get_position(key)

        # filter for objects of the correct size
        valid_position = (position.area > 0) & (position.area < 1000)
        position[~valid_position] = np.nan

        # in corridor
        distance_from_center = tracking.compute_distance(
            position[['x', 'y']],
            (tracking.arena_center_x, tracking.arena_center_y))
        in_corridor = (distance_from_center < tracking.arena_outer_radius) & (distance_from_center > tracking.arena_inner_radius)

        in_arena = ~in_corridor

        # in nests - loop through all nests in this experiment
        in_nest_times = []
        for nest_key in (lab.ArenaNest & key).fetch('KEY'):
            in_nest = tracking.is_position_in_nest(position, nest_key)
            in_nest_times.append(
                {**key, **nest_key,
                 'time_fraction_in_nest': in_nest.mean(),
                 'in_nest': in_nest})
            in_arena = in_arena & ~in_nest

        # in food patches - loop through all in-use patches during this session
        food_patch_keys = (
                InArena * InArenaEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'in_arena_start >= food_patch_install_time'
                & 'in_arena_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        in_food_patch_times = []
        for food_patch_key in food_patch_keys:
            # wheel data
            food_patch_description = (acquisition.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(in_arena_start),
                                               end=pd.Timestamp(in_arena_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle)

            patch_position = (acquisition.ExperimentFoodPatch.Position & food_patch_key).fetch1(
                'food_patch_position_x', 'food_patch_position_y')

            in_patch = tracking.is_in_patch(position, patch_position,
                                            wheel_distance_travelled, patch_radius=0.2)

            in_food_patch_times.append({
                **key, **food_patch_key,
                'time_fraction_in_patch': in_patch.mean(),
                'in_patch': in_patch.values})

            in_arena = in_arena & ~in_patch

        self.insert1({**key,
                      'time_fraction_in_corridor': in_corridor.mean(),
                      'in_corridor': in_corridor.values,
                      'time_fraction_in_arena': in_arena.mean(),
                      'in_arena': in_arena.values})
        self.Nest.insert(in_nest_times)
        self.FoodPatch.insert(in_food_patch_times)


@schema
class InArenaSummary(dj.Computed):
    definition = """
    -> InArena
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

    # Work on finished Session with TimeSlice and SubjectPosition fully populated only
    key_source = (InArena
                  & (InArena * InArenaEnd * InArenaTimeSlice
                     & 'time_slice_end = in_arena_end').proj()
                  & (InArena.aggr(InArenaTimeSlice, time_slice_count='count(time_slice_start)')
                     * InArena.aggr(InArenaSubjectPosition, tracking_count='count(time_slice_start)')
                     & 'time_slice_count = tracking_count'))

    def make(self, key):
        raw_data_dir = acquisition.Experiment.get_data_directory(key)
        in_arena_start, in_arena_end = (InArena * InArenaEnd & key).fetch1(
            'in_arena_start', 'in_arena_end')

        # subject weights
        weight_start = (acquisition.SubjectWeight.WeightTime
                        & f'weight_time = "{in_arena_start}"').fetch1('weight')
        weight_end = (acquisition.SubjectWeight.WeightTime
                      & f'weight_time = "{in_arena_end}"').fetch1('weight')

        # subject's position data in this session
        position = InArenaSubjectPosition.get_position(key)

        valid_position = (position.area > 0) & (position.area < 1000)  # filter for objects of the correct size
        position = position[valid_position]

        position_diff = np.sqrt(np.square(np.diff(position.x)) + np.square(np.diff(position.y)))
        total_distance_travelled = np.nancumsum(position_diff)[-1]

        # food patch data
        food_patch_keys = (
                InArena * InArenaEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'in_arena_start >= food_patch_install_time'
                & 'in_arena_end < IFNULL(food_patch_remove_time, "2200-01-01")').fetch('KEY')

        food_patch_statistics = []
        for food_patch_key in food_patch_keys:
            pellet_events = (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{in_arena_start}" AND "{in_arena_end}"').fetch(
                'event_time')
            # wheel data
            food_patch_description = (acquisition.ExperimentFoodPatch & food_patch_key).fetch1('food_patch_description')
            encoderdata = aeon_api.encoderdata(raw_data_dir.as_posix(),
                                               device=food_patch_description,
                                               start=pd.Timestamp(in_arena_start),
                                               end=pd.Timestamp(in_arena_end))
            wheel_distance_travelled = aeon_api.distancetravelled(encoderdata.angle).values

            food_patch_statistics.append({
                **key, **food_patch_key,
                'pellet_count': len(pellet_events),
                'wheel_distance_travelled': wheel_distance_travelled[-1]})

        total_pellet_count = np.sum([p['pellet_count'] for p in food_patch_statistics])
        total_wheel_distance_travelled = np.sum([p['wheel_distance_travelled'] for p in food_patch_statistics])

        self.insert1({**key,
                      'total_pellet_count': total_pellet_count,
                      'total_wheel_distance_travelled': total_wheel_distance_travelled,
                      'change_in_weight': weight_end - weight_start,
                      'total_distance_travelled': total_distance_travelled})
        self.FoodPatch.insert(food_patch_statistics)


@schema
class InArenaRewardRate(dj.Computed):
    definition = """
    -> InArena
    ---
    pellet_rate_timestamps: longblob  # timestamps of the pellet rate over time
    patch2_patch1_rate_diff: longblob  # rate differences between Patch 2 and Patch 1
    """

    class FoodPatch(dj.Part):
        definition = """
        -> master
        -> acquisition.ExperimentFoodPatch
        ---
        pellet_rate: longblob  # computed rate of pellet delivery over time
        """

    key_source = InArenaSummary()

    def make(self, key):
        in_arena_start, in_arena_end = (InArena * InArenaEnd & key).fetch1(
            'in_arena_start', 'in_arena_end')

        # food patch data
        food_patch_keys = (
                InArena * InArenaEnd
                * acquisition.ExperimentFoodPatch.join(acquisition.ExperimentFoodPatch.RemovalTime, left=True)
                & key
                & 'in_arena_start >= food_patch_install_time'
                & 'in_arena_end < IFNULL(food_patch_remove_time, "2200-01-01")').proj(
            'food_patch_description').fetch(as_dict=True)

        pellet_rate_timestamps = None
        rates = {}
        food_patch_reward_rates = []
        for food_patch_key in food_patch_keys:
            no_pellets = False
            pellet_events = (
                    acquisition.FoodPatchEvent * acquisition.EventType
                    & food_patch_key
                    & 'event_type = "TriggerPellet"'
                    & f'event_time BETWEEN "{in_arena_start}" AND "{in_arena_end}"').fetch(
                'event_time')

            if not pellet_events.size:
                pellet_events = np.array([in_arena_start, in_arena_end])
                no_pellets = True

            pellet_rate = aeon_utils.get_events_rates(
                events=pd.DataFrame({'event_time': pellet_events}).set_index('event_time'),
                window_len_sec=600,
                start=pd.Timestamp(in_arena_start),
                end=pd.Timestamp(in_arena_end),
                frequency='5s', smooth='120s',
                center=True)

            if no_pellets:
                pellet_rate = pd.Series(index=pellet_rate.index, data=np.full(len(pellet_rate), 0))

            if pellet_rate_timestamps is None:
                pellet_rate_timestamps = pellet_rate.index.to_pydatetime()

            rates[food_patch_key.pop('food_patch_description')] = pellet_rate.values

            food_patch_reward_rates.append({
                **key, **food_patch_key,
                'pellet_rate': pellet_rate.values})

        self.insert1({**key,
                      'pellet_rate_timestamps': pellet_rate_timestamps,
                      'patch2_patch1_rate_diff': rates['Patch2'] - rates['Patch1']})
        self.FoodPatch.insert(food_patch_reward_rates)
