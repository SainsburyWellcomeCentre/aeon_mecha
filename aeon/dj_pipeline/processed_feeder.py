"""DataJoint schema to process data from feeders.

Downstream of ``streams.FeederEncoder``, ``streams.FeederBeamBreak``, and
``streams.FeederDeliverPellet``. Because those stream tables are auto-generated
by ``streams_maker`` only when an experiment registers an ``UndergroundFeeder``
device, this module uses **deferred schema activation**: tables are defined
against an unbound ``dj.Schema()`` at import time, and ``activate()`` only
binds them to the database if the required upstream stream tables exist.

The canonical caller is ``aeon.dj_pipeline.__init__``, which invokes
``activate()`` after ``streams_maker`` has run. If the upstream streams are not
present (e.g. an experiment with no feeders), activation is a logged no-op and
``Encoder``/``DiggingBouts``/``DeliveryEvents`` simply remain undeclared.

If new feeder experiments are registered later in the same Python session,
re-running ``activate()`` picks them up.
"""

import datajoint as dj
import numpy as np
import pandas as pd

from aeon.dj_pipeline import get_schema_name, streams

schema = dj.Schema()
logger = dj.logger


@schema
class Encoder(dj.Computed):
    definition = """
    -> streams.FeederEncoder
    ---
    timestamps: <blob>       # (datetime) timestamps, contains nans to bridge periods of no movement
    distance_cm: <blob>      # Wheel distance in cm, contains nans to bridge periods of no movement
    """

    def make(self, key):
        """Process feeder data to only include movement periods.

        The first and last 2 seconds of each chunk are kept irrespective of movement.
        For other periods, movement is defined as a change in distance of at least
        0.03 cm within a 2 second window.

        Adrian 2026-05-01
        """
        # pd.DataFrame with time, angle, intensity columns
        encoder_df = (streams.FeederEncoder & key).fetch1("stream_df")
        angle = encoder_df["angle"]   # pd.Series with datetime index

        # convert angle to distance
        from swc.aeon.analysis.utils import distancetravelled
        distance = distancetravelled(angle=angle, radius=-4)

        # smooth and resample distance to 100Hz
        from scipy.ndimage import gaussian_filter1d
        smooth_dist = gaussian_filter1d(distance, sigma=5)
        smooth_dist = pd.Series(smooth_dist, index=distance.index)
        resampled = smooth_dist.resample(rule="10ms").first()

        # TODO: add edge arguments to aeon_api.load in streams to avoid this
        # hacky removal of full hour that appear both at start and end
        resampled = resampled.iloc[0:-1]

        # remove periods of the data where the feeder is not moving
        trace = resampled
        keep = np.ones(len(trace), dtype=bool)

        dt = 200    # 2 seconds
        moving = True

        onsets = [0]
        offsets = []

        for i in range(dt, len(trace) - dt):
            now = trace.iloc[i]
            past = trace.iloc[i - dt]
            future = trace.iloc[i + dt]

            if not moving:
                keep[i] = False

                # check for starting
                if np.abs(now - past) >= 0.03 or np.abs(future - now) >= 0.03:
                    moving = True

                    # merge with last period if close enough
                    if offsets[-1] > i - 2 * dt:
                        # remove last stopping and set keep to True
                        keep[offsets[-1]:i + 1] = True
                        offsets.pop()
                    else:
                        onsets.append(i)
            elif moving:
                # check for stopping
                if np.abs(future - now) < 0.02 and np.abs(now - past) < 0.02:
                    moving = False
                    offsets.append(i)

        # add values for end of trace
        offsets.append(len(trace))

        if not moving:
            # add last second to keep
            onsets.append(len(trace) - dt)
        if len(onsets) != len(offsets):
            raise AssertionError("onsets and offsets length mismatch")

        # create time series only with movement periods, separated with nans
        short_trace = []
        for i in range(len(onsets)):
            short_trace.append(trace.iloc[onsets[i]:offsets[i]])

            if i < len(offsets) - 1:
                short_trace.append(trace.iloc[offsets[i]:offsets[i] + 1] * np.nan)

        short_trace = pd.concat(short_trace)

        # add entry to database
        new_entry = {
            **key,
            "timestamps": short_trace.index.values,
            "distance_cm": short_trace.values,
        }
        self.insert1(new_entry)


@schema
class DiggingBouts(dj.Computed):
    definition = """
    -> Encoder
    ---
    nr_bouts: int32                      # number of digging bouts detected
    total_digging_time_s: float32        # total digging time in seconds
    total_digging_distance_cm: float32   # total digging distance in cm
    onset_times: <blob>                  # timestamps of digging bout onsets
    offset_times: <blob>                 # timestamps of digging bout offsets
    digging_durations_s: <blob>          # durations of each digging bout in seconds
    digging_distances_cm: <blob>         # distances of each digging bout in cm
    """

    def make(self, key):
        """Process compressed feeder trace to extract on and offsets."""
        time, dist = (Encoder & key).fetch1("timestamps", "distance_cm")
        dist_series = pd.Series(data=dist, index=time)

        trace = dist_series.copy()
        block_onset = np.zeros(len(trace), dtype=bool)

        s1 = 100     # 1 second
        s2 = 200     # 2 seconds
        onsets = []
        offsets = []
        after_onset = False

        for i in range(s2, len(trace) - s2):
            now = trace.iloc[i]
            t1 = trace.iloc[i - s1]
            t2 = trace.iloc[i - s2 + 50]
            t3 = trace.iloc[i + s2]

            # movement criteria; no onset in last second
            if (
                after_onset is False
                and (now - t1) >= 0.1
                and (now - t2) < 0.2
                and (t3 - now) > 1
                and np.sum(block_onset[i - s1:i]) == 0
            ):
                block_onset[i] = True
                onsets.append(i)
                after_onset = True

            if after_onset and (t3 - now) <= 0.1:
                offsets.append(i)
                after_onset = False

        if len(onsets) == len(offsets) + 1:
            # for now: if last offset is missing, set it to end of hour
            offsets.append(len(trace) - 1)

        new_entry = dict(
            **key,
            nr_bouts=len(onsets),
            total_digging_time_s=np.sum((trace.index[offsets] - trace.index[onsets]).total_seconds()),
            total_digging_distance_cm=np.sum(trace.iloc[offsets].values - trace.iloc[onsets].values),
            onset_times=trace.index[onsets].values,
            offset_times=trace.index[offsets].values,
            digging_durations_s=np.array((trace.index[offsets] - trace.index[onsets]).total_seconds()),
            digging_distances_cm=trace.iloc[offsets].values - trace.iloc[onsets].values,
        )
        self.insert1(new_entry)


@schema
class DeliveryEvents(dj.Computed):
    definition = """
    -> streams.FeederDeliverPellet
    -> streams.FeederBeamBreak
    ---
    nr_events: int32                     # number of requests to deliver a pellet
    nr_pellets: int32                    # number of pellets successfully delivered
    delivery_request_times: <blob>       # timestamps of delivery requests
    actual_delivery_times: <blob>        # timestamps of actual delivery based on motor
    nr_delivery_attempts: <blob>         # number of delivery attempts per request
    successful_deliveries: <blob>        # boolean array indicating successful deliveries
    first_beam_breaks: <blob>            # timestamps of first beam break after each delivery request
    """

    def make(self, key):
        """Process delivery commands and beam breaks."""
        deliver_df = (streams.FeederDeliverPellet & key).fetch1("stream_df")
        deliver_command = deliver_df.index.values  # numpy array of datetime64 timestamps

        beam_df = (streams.FeederBeamBreak & key).fetch1("stream_df")
        beam_break = beam_df.index.values  # numpy array of datetime64 timestamps

        # group delivery commands that are within 2 seconds of each other
        if len(deliver_command) > 0:
            deliver_onsets = [[deliver_command[0]]]
            last_t = deliver_onsets[0][0]

            for i in range(1, len(deliver_command)):
                t = deliver_command[i]
                if t - last_t < pd.Timedelta(seconds=2):
                    # same delivery attempt
                    deliver_onsets[-1].append(t)  # make list entry longer
                else:
                    # new delivery attempt
                    deliver_onsets.append([t])  # start new entry
                last_t = t
        else:
            deliver_onsets = []

        # find the first bream break after each delivery onset
        beam_break_t = []
        for times in deliver_onsets:
            t_min = times[0]
            t_max = times[-1] + pd.Timedelta(seconds=1)
            bb_times = beam_break[(beam_break >= t_min) & (beam_break <= t_max)]
            if len(bb_times) > 0:
                beam_break_t.append(bb_times[0])
            else:
                beam_break_t.append(np.nan)

        # extract variables
        new_entry = dict(
            **key,
            nr_events=len(deliver_onsets),
            nr_pellets=np.sum(~pd.isna(beam_break_t)),
            delivery_request_times=np.array([times[0] for times in deliver_onsets]),
            actual_delivery_times=np.array([times[-1] for times in deliver_onsets]),
            nr_delivery_attempts=np.array([len(times) for times in deliver_onsets]),
            successful_deliveries=~pd.isna(beam_break_t),
            first_beam_breaks=np.array(beam_break_t),
        )
        self.insert1(new_entry)


@schema
class FeederQC(dj.Computed):
    definition = """
    -> streams.FeederEncoder
    ---
    violation_count: int32    # number of temporal violations detected
    violation_times: <blob>   # (datetime) timestamps for temporal violations
    duplicate_count: int32    # number of duplicate timestamps detected
    duplicate_times: <blob>   # (datetime) timestamps for duplicates
    """

    def make(self, key):
        """Detect duplication and violation of temporal order of feeder data."""
        encoder_df = (streams.FeederEncoder & key).fetch1("stream_df")
        t = encoder_df.index.values

        # Get violations
        violation_frames = t.argsort() - np.arange(len(t))
        violation_mask = violation_frames < 0
        violation_count = int(np.sum(violation_mask))
        violation_times = t[violation_mask]

        # Get duplicates
        duplicate_mask = np.diff(t) == np.timedelta64(0)
        duplicate_count = int(np.sum(duplicate_mask))
        duplicate_times = t[np.where(duplicate_mask)[0] + 1]

        new_entry = {
            **key,
            "violation_count": violation_count,
            "violation_times": violation_times,
            "duplicate_count": duplicate_count,
            "duplicate_times": duplicate_times,
        }
        self.insert1(new_entry)


_REQUIRED_STREAMS = ("FeederEncoder", "FeederBeamBreak", "FeederDeliverPellet")


def activate(*, create_schema: bool = True, create_tables: bool = True) -> bool:
    """Bind the processed_feeder schema to the database if upstream streams exist.

    Checks that ``streams.FeederEncoder``, ``streams.FeederBeamBreak``, and
    ``streams.FeederDeliverPellet`` are all present as Python attributes — i.e.
    that ``streams_maker`` has already generated them from a registered
    ``UndergroundFeeder`` device. If any are missing, logs a warning and
    returns ``False`` without touching the database.

    Re-callable: if new feeder experiments are registered later in the same
    Python session and ``streams_maker`` regenerates the missing tables,
    invoking this again will activate the schema.

    Returns:
        True if the schema was activated, False if upstream streams missing.
    """
    if streams is None:
        logger.warning("processed_feeder not activated — streams module not initialized")
        return False
    missing = [t for t in _REQUIRED_STREAMS if not hasattr(streams, t)]
    if missing:
        logger.warning(f"processed_feeder not activated — streams missing: {missing}")
        return False
    schema.activate(
        get_schema_name("processed_feeder"),
        create_schema=create_schema,
        create_tables=create_tables,
    )
    return True
