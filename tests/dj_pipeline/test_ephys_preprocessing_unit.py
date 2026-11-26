"""Unit tests for the preprocessing pipelines in utils/ephys_preprocessing."""

import numpy as np
import pytest
import spikeinterface.full as si

from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

pytestmark = pytest.mark.unit


def _split_and_preprocess(rec, pipeline_before, pipeline_after, group_channel_ids):
    """Mirror PreProcessing.make_compute: before-stage, select group channels, after-stage."""
    rec = si.apply_preprocessing_pipeline(rec, pipeline_before)
    rec = rec.select_channels(group_channel_ids)
    return si.apply_preprocessing_pipeline(rec, pipeline_after)


def test_default_pipeline_matches_original_recipe():
    """Without si_pipeline, aeon_default is used: same traces as the old hard-coded recipe."""
    pipeline_before, pipeline_after = resolve_preprocessing_pipeline({})
    default = resolve_preprocessing_pipeline({"si_pipeline": "aeon_default"})
    assert (pipeline_before, pipeline_after) == default
    assert pipeline_before == {}

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    original = si.common_reference(si.bandpass_filter(rec, freq_min=300, freq_max=6000), operator="median")
    np.testing.assert_array_equal(
        si.apply_preprocessing_pipeline(rec, pipeline_after).get_traces(), original.get_traces()
    )


def test_staged_pipeline_applies_before_and_after_splitting():
    """before_splitting runs on the full recording, after_splitting on the group's channels."""
    pipeline_before, pipeline_after = resolve_preprocessing_pipeline(
        {
            "si_pipeline": {
                "before_splitting": {"common_reference": {"operator": "median"}},
                "after_splitting": {"bandpass_filter": {"freq_min": 300, "freq_max": 6000}},
            }
        }
    )

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    group_channel_ids = rec.channel_ids[:3]

    expected_recording = si.bandpass_filter(
        si.common_reference(rec, operator="median").select_channels(group_channel_ids),
        freq_min=300,
        freq_max=6000,
    )
    pp_recording = _split_and_preprocess(rec, pipeline_before, pipeline_after, group_channel_ids)
    np.testing.assert_array_equal(pp_recording.get_traces(), expected_recording.get_traces())


@pytest.mark.parametrize(
    ("si_pipeline", "expected_before", "expected_after"),
    [
        (
            {"before_splitting": {"common_reference": {"operator": "median"}}},
            {"common_reference": {"operator": "median"}},
            {},
        ),
        (
            {"after_splitting": {"bandpass_filter": {"freq_min": 300, "freq_max": 6000}}},
            {},
            {"bandpass_filter": {"freq_min": 300, "freq_max": 6000}},
        ),
        # Legacy flat dict: every step runs after splitting.
        (
            {"bandpass_filter": {"freq_min": 300, "freq_max": 6000}},
            {},
            {"bandpass_filter": {"freq_min": 300, "freq_max": 6000}},
        ),
    ],
    ids=["before_only", "after_only", "legacy_flat"],
)
def test_partial_and_legacy_pipelines(si_pipeline, expected_before, expected_after):
    """Tests that user input gets resolved as expected for partial and legacy pipelines."""
    assert resolve_preprocessing_pipeline({"si_pipeline": si_pipeline}) == (expected_before, expected_after)


@pytest.mark.parametrize(
    ("preprocessing_params", "error", "match"),
    [
        ({"si_pipline": "aeon_default"}, ValueError, "si_pipline"),
        ({"si_pipeline": "aeon_defualt"}, ValueError, "aeon_defualt"),
        ({"si_pipeline": ["bandpass_filter"]}, TypeError, "registered name or a pipeline dict"),
        ({"si_pipeline": {"before_splitting": {"not_a_step": {}}}}, TypeError, "not_a_step"),
    ],
)
def test_invalid_preprocessing_params_raise(preprocessing_params, error, match):
    """Typos in the section or the name, and a value that's neither name nor dict, raise."""
    with pytest.raises(error, match=match):
        resolve_preprocessing_pipeline(preprocessing_params)
