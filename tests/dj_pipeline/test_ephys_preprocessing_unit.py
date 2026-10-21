"""Unit tests for the preprocessing pipelines in utils/ephys_preprocessing."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit

AEON_DEFAULT = {
    "bandpass_filter": {"freq_min": 300, "freq_max": 6000},
    "common_reference": {"operator": "median"},
}


def test_default_and_legacy_names_resolve_to_aeon_default():
    """No keys, "aeon_default" and the legacy "ephys_preproc" all select the same pipeline."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    assert resolve_preprocessing_pipeline({}) == AEON_DEFAULT
    assert resolve_preprocessing_pipeline({"SI_PREPROCESSING_METHOD": "aeon_default"}) == AEON_DEFAULT
    assert resolve_preprocessing_pipeline({"SI_PREPROCESSING_METHOD": "ephys_preproc"}) == AEON_DEFAULT


def test_aeon_default_matches_original_recipe():
    """Same traces as the hard-coded bandpass + median CAR it replaces."""
    import spikeinterface.full as si

    from aeon.dj_pipeline.utils.ephys_preprocessing import apply_preprocessing

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    original = si.common_reference(si.bandpass_filter(rec, freq_min=300, freq_max=6000), operator="median")
    np.testing.assert_array_equal(apply_preprocessing(rec, {}).get_traces(), original.get_traces())


def test_overrides_merge_per_step():
    """Overrides target one step; the rest of the registered pipeline is unchanged."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    pipeline = resolve_preprocessing_pipeline(
        {"SI_PREPROCESSING_PARAMS": {"bandpass_filter": {"freq_min": 250, "dtype": "float32"}}}
    )
    assert pipeline == {
        "bandpass_filter": {"freq_min": 250, "freq_max": 6000, "dtype": "float32"},
        "common_reference": {"operator": "median"},
    }
    # The registered pipeline itself is not mutated.
    assert resolve_preprocessing_pipeline({}) == AEON_DEFAULT


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"SI_PREPROCESSING_PARAMS": {"whiten": {"dtype": "float32"}}}, "whiten"),
        ({"SI_PREPROCESSING_PIPELINE": {}, "SI_PREPROCESSING_METHOD": "none"}, "can't be combined"),
    ],
)
def test_invalid_selection_raises(params, match):
    """An override for a step the pipeline lacks, or an inline pipeline plus a method, raises."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    with pytest.raises(ValueError, match=match):
        resolve_preprocessing_pipeline(params)
