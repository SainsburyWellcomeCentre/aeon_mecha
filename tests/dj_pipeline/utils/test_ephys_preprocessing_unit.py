"""Unit tests for the preprocessing pipelines in utils/ephys_preprocessing."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_default_pipeline_matches_original_recipe():
    """Without si_pipeline, aeon_default is used: same traces as the old hard-coded recipe."""
    import spikeinterface.full as si

    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    pipeline = resolve_preprocessing_pipeline({})
    assert pipeline == resolve_preprocessing_pipeline({"si_pipeline": "aeon_default"})

    rec = si.generate_recording(num_channels=8, durations=[1.0], seed=0)
    original = si.common_reference(si.bandpass_filter(rec, freq_min=300, freq_max=6000), operator="median")
    np.testing.assert_array_equal(
        si.apply_preprocessing_pipeline(rec, pipeline).get_traces(), original.get_traces()
    )


@pytest.mark.parametrize(
    ("preprocessing_params", "error", "match"),
    [
        ({"si_pipline": "aeon_default"}, ValueError, "si_pipline"),
        ({"si_pipeline": "aeon_defualt"}, ValueError, "aeon_defualt"),
        ({"si_pipeline": ["bandpass_filter"]}, TypeError, "registered name or a pipeline dict"),
    ],
)
def test_invalid_preprocessing_params_raise(preprocessing_params, error, match):
    """Typos in the section or the name, and a value that's neither name nor dict, raise."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    with pytest.raises(error, match=match):
        resolve_preprocessing_pipeline(preprocessing_params)


def test_none_resolves_to_an_empty_pipeline_and_passes_traces_through():
    """The "none" pipeline is registered but nothing exercised it.

    Its failure mode is the one the registry exists to prevent: silently applying
    preprocessing to a paramset that asked for none, or the reverse.
    """
    import spikeinterface.full as si

    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    pipeline = resolve_preprocessing_pipeline({"si_pipeline": "none"})
    assert pipeline == {}

    rec = si.generate_recording(num_channels=4, durations=[0.5], seed=0)
    np.testing.assert_array_equal(
        si.apply_preprocessing_pipeline(rec, pipeline).get_traces(), rec.get_traces()
    )


def test_inline_pipeline_is_used_verbatim():
    """An inline dict bypasses the registry; only its error branches were covered."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    inline = {"bandpass_filter": {"freq_min": 500, "freq_max": 3000}}
    resolved = resolve_preprocessing_pipeline({"si_pipeline": inline})

    # Against a literal, not against `inline`: the inline branch returns the very dict
    # it was handed, so comparing the two would hold even if defaults were merged in.
    assert resolved == {"bandpass_filter": {"freq_min": 500, "freq_max": 3000}}
    assert "common_reference" not in resolved, "the registered pipeline must not leak in"
    assert inline == {"bandpass_filter": {"freq_min": 500, "freq_max": 3000}}, "input mutated"


def test_resolving_does_not_mutate_the_registered_pipeline():
    """A caller that edits what it gets back must not change the next caller's result."""
    from aeon.dj_pipeline.utils.ephys_preprocessing import resolve_preprocessing_pipeline

    first = resolve_preprocessing_pipeline({"si_pipeline": "aeon_default"})
    first_copy = {step: dict(kwargs) for step, kwargs in first.items()}
    first["bandpass_filter"]["freq_min"] = 1
    assert resolve_preprocessing_pipeline({}) == first_copy
