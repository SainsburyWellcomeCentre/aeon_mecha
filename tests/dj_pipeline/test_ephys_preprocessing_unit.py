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


def test_none_applies_no_steps():
    import spikeinterface.full as si

    from aeon.dj_pipeline.utils.ephys_preprocessing import apply_preprocessing

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    out = apply_preprocessing(rec, {"SI_PREPROCESSING_METHOD": "none"})
    np.testing.assert_array_equal(out.get_traces(), rec.get_traces())


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


def test_inline_pipeline():
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    inline = {"highpass_filter": {"freq_min": 400}, "common_reference": {}}
    assert resolve_preprocessing_pipeline({"SI_PREPROCESSING_PIPELINE": inline}) == inline


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"SI_PREPROCESSING_METHOD": "aeon_defualt"}, ValueError, "aeon_defualt"),
        ({"SI_PREPROCESSING_PARAMS": {"whiten": {"dtype": "float32"}}}, ValueError, "whiten"),
        (
            {"SI_PREPROCESSING_PIPELINE": {}, "SI_PREPROCESSING_METHOD": "none"},
            ValueError,
            "can't be combined",
        ),
        ({"SI_PREPROCESSING_PIPELINE": {"bandpas_filter": {}}}, TypeError, "bandpas_filter"),
    ],
)
def test_invalid_selection_raises(params, error, match):
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    with pytest.raises(error, match=match):
        resolve_preprocessing_pipeline(params)


def test_bad_step_argument_raises_on_apply():
    """SpikeInterface rejects an argument a step doesn't take."""
    import spikeinterface.full as si

    from aeon.dj_pipeline.utils.ephys_preprocessing import apply_preprocessing

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    with pytest.raises(TypeError, match="freq_mni"):
        apply_preprocessing(rec, {"SI_PREPROCESSING_PARAMS": {"bandpass_filter": {"freq_mni": 250}}})


def test_duplicate_registration_raises():
    from aeon.dj_pipeline.utils.ephys_preprocessing import register_preprocessing

    with pytest.raises(ValueError, match="already registered"):
        register_preprocessing("aeon_default", {})
