"""Integration tests for the processed_feeder and processed_movement schemas.

Uses the behavior golden datasets (parametrized via ``golden_dataset_config``)
to populate the upstream streams (``FeederEncoder``, ``FeederBeamBreak``,
``FeederDeliverPellet``, ``CameraPosition``, ``CameraVideo``) with a small
sample, then exercises ``populate()`` on each computed table in
``processed_feeder`` and ``processed_movement``. Assertions are **structural**
(array shape consistency, counter/array length agreement, dtype sanity) —
hardcoded numerical equality would break the first time golden data is
regenerated.

Both processed modules are re-activated inside the fixtures because
``aeon/dj_pipeline/__init__.py`` runs ``activate()`` at module-import time,
before ``streams_maker`` has regenerated ``streams.py`` against the test
catalog. Re-calling ``activate()`` after ``full_pipeline`` finishes is the
intended re-callable behavior of the deferred-activation design.
"""

import numpy as np
import pytest

PROCESSED_POPULATE_LIMIT = 2  # only need a couple of processed rows per table to assert

REQUIRED_FEEDER_STREAMS = ("FeederEncoder", "FeederBeamBreak", "FeederDeliverPellet")
REQUIRED_MOVEMENT_STREAMS = ("CameraPosition", "CameraVideo")


@pytest.fixture(scope="module")
def streams_populated(test_epochs, full_pipeline, golden_dataset_config):
    """Populate prerequisite streams so processed tables have keys to consume.

    Runs ``EpochConfig.populate()`` and ``Chunk.ingest_chunks()`` (idempotent),
    then fully populates the feeder and camera streams that processed_feeder
    and processed_movement depend on. The golden datasets are short (~1-2 h)
    so the per-stream chunk count is small and full populate stays fast.
    Errors are suppressed per stream — empty/missing streams just leave the
    table empty, and downstream tests skip on no rows.
    """
    acquisition = full_pipeline["acquisition"]
    streams = full_pipeline["streams"]
    cfg = golden_dataset_config

    exp_name = cfg["experiment_name"]
    acquisition.EpochConfig.populate({"experiment_name": exp_name})
    acquisition.Chunk.ingest_chunks(exp_name)

    for name in (*REQUIRED_FEEDER_STREAMS, *REQUIRED_MOVEMENT_STREAMS):
        tbl = getattr(streams, name, None)
        if tbl is None:
            continue
        tbl.populate(
            {"experiment_name": exp_name},
            display_progress=False,
            suppress_errors=True,
        )
    return streams


@pytest.fixture(scope="module")
def processed_feeder_activated(streams_populated):
    """Re-activate processed_feeder against the regenerated test streams."""
    from aeon.dj_pipeline import processed_feeder

    if not processed_feeder.activate():
        pytest.skip("processed_feeder cannot activate — required streams missing")
    return processed_feeder


@pytest.fixture(scope="module")
def processed_movement_activated(streams_populated):
    """Re-activate processed_movement against the regenerated test streams."""
    from aeon.dj_pipeline import processed_movement

    if not processed_movement.activate():
        pytest.skip("processed_movement cannot activate — required streams missing")
    return processed_movement


def _restriction(cfg):
    return {"experiment_name": cfg["experiment_name"]}


def _data_bearing_keys(streams_module, stream_names, restriction, limit):
    """Return ``limit`` keys whose row in each ``stream_names`` table has ``sample_count > 0``.

    Restricts populate() to upstream chunks that actually have data, avoiding
    the trap where ``max_calls=N`` happens to pick the first N chunks (which
    may all be empty for a given dataset).
    """
    query = None
    for name in stream_names:
        tbl = getattr(streams_module, name)
        # Project to drop attributes that would otherwise force a join here;
        # we only want the shared primary keys.
        sub = (tbl & restriction & "sample_count > 0").proj()
        query = sub if query is None else query * sub
    return query.fetch("KEY", limit=limit)


class TestProcessedFeeder:
    """End-to-end: golden data -> streams -> processed_feeder.* populate."""

    def test_encoder_populates(
        self, processed_feeder_activated, streams_populated, golden_dataset_config
    ):
        pf = processed_feeder_activated
        restriction = _restriction(golden_dataset_config)
        keys = _data_bearing_keys(
            streams_populated, ("FeederEncoder",), restriction, PROCESSED_POPULATE_LIMIT
        )
        if not keys:
            pytest.skip("No FeederEncoder chunks with sample_count > 0")

        pf.Encoder.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pf.Encoder & keys).to_dicts()
        assert rows, "Encoder.populate inserted no rows for data-bearing FeederEncoder keys"

        for r in rows:
            assert isinstance(r["timestamps"], np.ndarray)
            assert isinstance(r["distance_cm"], np.ndarray)
            assert len(r["timestamps"]) == len(r["distance_cm"])
            # distance_cm includes NaNs to bridge non-movement; finite values must be numeric
            finite = r["distance_cm"][np.isfinite(r["distance_cm"].astype(float))]
            assert finite.dtype.kind in "fc"

    def test_digging_bouts_populates(
        self, processed_feeder_activated, streams_populated, golden_dataset_config
    ):
        pf = processed_feeder_activated
        restriction = _restriction(golden_dataset_config)
        keys = _data_bearing_keys(
            streams_populated, ("FeederEncoder",), restriction, PROCESSED_POPULATE_LIMIT
        )
        if not keys:
            pytest.skip("No FeederEncoder chunks with sample_count > 0")

        # Encoder is the upstream — populate it first (idempotent if already done).
        pf.Encoder.populate(keys, display_progress=False, suppress_errors=True)
        pf.DiggingBouts.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pf.DiggingBouts & keys).to_dicts()
        assert rows, "DiggingBouts.populate inserted no rows from populated Encoder keys"

        for r in rows:
            n = r["nr_bouts"]
            assert isinstance(n, int | np.integer)
            assert n >= 0
            assert len(r["onset_times"]) == n
            assert len(r["offset_times"]) == n
            assert len(r["digging_durations_s"]) == n
            assert len(r["digging_distances_cm"]) == n
            assert r["total_digging_time_s"] >= 0
            assert r["total_digging_distance_cm"] >= 0

    def test_delivery_events_populates(
        self, processed_feeder_activated, streams_populated, golden_dataset_config
    ):
        pf = processed_feeder_activated
        restriction = _restriction(golden_dataset_config)
        # Both DeliverPellet AND BeamBreak need data for DeliveryEvents to make sense.
        keys = _data_bearing_keys(
            streams_populated,
            ("FeederDeliverPellet", "FeederBeamBreak"),
            restriction,
            PROCESSED_POPULATE_LIMIT,
        )
        if not keys:
            pytest.skip(
                "No chunks where both FeederDeliverPellet and FeederBeamBreak have data"
            )

        pf.DeliveryEvents.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pf.DeliveryEvents & keys).to_dicts()
        assert rows, "DeliveryEvents.populate inserted no rows from data-bearing keys"

        for r in rows:
            ne = r["nr_events"]
            assert isinstance(ne, int | np.integer)
            assert ne >= 0
            assert len(r["delivery_request_times"]) == ne
            assert len(r["actual_delivery_times"]) == ne
            assert len(r["nr_delivery_attempts"]) == ne
            assert len(r["successful_deliveries"]) == ne
            assert len(r["first_beam_breaks"]) == ne
            assert 0 <= r["nr_pellets"] <= ne

    def test_feeder_qc_populates(
        self, processed_feeder_activated, streams_populated, golden_dataset_config
    ):
        pf = processed_feeder_activated
        restriction = _restriction(golden_dataset_config)
        keys = _data_bearing_keys(
            streams_populated, ("FeederEncoder",), restriction, PROCESSED_POPULATE_LIMIT
        )
        if not keys:
            pytest.skip("No FeederEncoder chunks with sample_count > 0")

        pf.FeederQC.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pf.FeederQC & keys).to_dicts()
        assert rows, "FeederQC.populate inserted no rows from data-bearing keys"

        for r in rows:
            vc = r["violation_count"]
            dc = r["duplicate_count"]
            assert isinstance(vc, int | np.integer)
            assert vc >= 0
            assert isinstance(dc, int | np.integer)
            assert dc >= 0
            assert len(r["violation_times"]) == vc
            assert len(r["duplicate_times"]) == dc


class TestProcessedMovement:
    """End-to-end: golden data -> streams -> processed_movement.* populate."""

    def test_mouse_position_tracking_populates(
        self, processed_movement_activated, streams_populated, golden_dataset_config
    ):
        pm = processed_movement_activated
        restriction = _restriction(golden_dataset_config)
        # Both CameraPosition AND CameraVideo are read inside make().
        keys = _data_bearing_keys(
            streams_populated,
            ("CameraPosition", "CameraVideo"),
            restriction,
            PROCESSED_POPULATE_LIMIT,
        )
        if not keys:
            pytest.skip(
                "No chunks where both CameraPosition and CameraVideo have data"
            )

        pm.MousePositionTracking.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pm.MousePositionTracking & keys).to_dicts()
        assert rows, "MousePositionTracking.populate inserted no rows from data-bearing keys"

        for r in rows:
            n = len(r["timestamps"])
            assert n > 0
            assert len(r["x"]) == n
            assert len(r["y"]) == n
            assert len(r["area"]) == n
            assert 0 <= int(r["nr_nans"]) <= n
            # The make() pads missing video frames with NaN; the NaN count must match
            assert int(r["nr_nans"]) == int(np.isnan(r["x"].astype(float)).sum())

    def test_video_qc_populates(
        self, processed_movement_activated, streams_populated, golden_dataset_config
    ):
        pm = processed_movement_activated
        restriction = _restriction(golden_dataset_config)
        keys = _data_bearing_keys(
            streams_populated, ("CameraVideo",), restriction, PROCESSED_POPULATE_LIMIT
        )
        if not keys:
            pytest.skip("No CameraVideo chunks with sample_count > 0")

        pm.VideoQC.populate(keys, display_progress=False, suppress_errors=True)
        rows = (pm.VideoQC & keys).to_dicts()
        assert rows, "VideoQC.populate inserted no rows from data-bearing keys"

        for r in rows:
            vc = r["violation_count"]
            dc = r["duplicate_count"]
            assert isinstance(vc, int | np.integer)
            assert vc >= 0
            assert isinstance(dc, int | np.integer)
            assert dc >= 0
            assert len(r["violation_times"]) == vc
            assert len(r["duplicate_times"]) == dc
