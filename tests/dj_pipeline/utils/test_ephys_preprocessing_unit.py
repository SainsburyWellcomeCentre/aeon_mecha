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
