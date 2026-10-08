"""Unit tests for the SI_PREPROCESSING_METHOD registry in spike_sorting."""

import pytest

pytestmark = pytest.mark.unit


def test_builtin_methods_registered():
    from aeon.dj_pipeline.spike_sorting import (
        DEFAULT_PREPROCESSING_METHOD,
        aeon_default,
        get_preprocessing_method,
        no_preprocessing,
    )

    assert get_preprocessing_method(DEFAULT_PREPROCESSING_METHOD) is aeon_default
    assert get_preprocessing_method("none") is no_preprocessing


def test_unknown_method_raises():
    from aeon.dj_pipeline.spike_sorting import get_preprocessing_method

    with pytest.raises(ValueError, match="aeon_defualt"):
        get_preprocessing_method("aeon_defualt")


def test_duplicate_registration_raises():
    from aeon.dj_pipeline.spike_sorting import register_preprocessing

    with pytest.raises(ValueError, match="already registered"):
        register_preprocessing("aeon_default")(lambda recording: recording)


def test_none_returns_same_recording():
    from aeon.dj_pipeline.spike_sorting import no_preprocessing

    recording = object()
    assert no_preprocessing(recording) is recording


def test_unused_kwargs_raise():
    """A misspelled or misplaced SI_PREPROCESSING_PARAMS key must fail, not be ignored."""
    from aeon.dj_pipeline.spike_sorting import aeon_default, no_preprocessing

    with pytest.raises(ValueError, match="freq_mni"):
        aeon_default(object(), freq_mni=250)
    with pytest.raises(ValueError, match="freq_min"):
        no_preprocessing(object(), freq_min=250)


def test_aeon_default_kwargs_reach_spikeinterface():
    """Defaults reproduce the original recipe; overrides are applied."""
    import numpy as np
    import spikeinterface.full as si

    from aeon.dj_pipeline.spike_sorting import aeon_default

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    original = si.common_reference(si.bandpass_filter(rec, freq_min=300, freq_max=6000), operator="median")
    np.testing.assert_array_equal(aeon_default(rec).get_traces(), original.get_traces())

    out = aeon_default(rec, freq_min=250, operator="average")
    assert out._kwargs["operator"] == "average"
    assert out._kwargs["recording"]._kwargs["freq_min"] == 250


def test_legacy_ephys_preproc_name_resolves_to_aeon_default():
    """Paramsets from earlier runbooks carry SI_PREPROCESSING_METHOD="ephys_preproc"."""
    from aeon.dj_pipeline.spike_sorting import aeon_default, get_preprocessing_method

    assert get_preprocessing_method("ephys_preproc") is aeon_default
