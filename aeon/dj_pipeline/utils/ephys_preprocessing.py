"""Preprocessing pipelines for spike sorting, selectable per SortingParamSet.

A preprocessing pipeline is a SpikeInterface pipeline dict: ordered step names, each
mapped to that step's keyword arguments (omitted arguments take SpikeInterface's
defaults). See
https://spikeinterface.readthedocs.io/en/stable/how_to/build_pipeline_with_dicts.html

``spike_sorting.PreProcessing`` applies the pipeline to the recording, after the fixed
unsigned-to-signed conversion and before the result is saved for sorting. A paramset
selects it with ``si_pipeline`` in its ``SI_PREPROCESSING_PARAMS`` section, either as
the name of a registered pipeline or as a pipeline dict written inline::

    "SI_PREPROCESSING_PARAMS": {"si_pipeline": "aeon_default"}

    "SI_PREPROCESSING_PARAMS": {
        "si_pipeline": {
            "bandpass_filter": {"freq_min": 250, "freq_max": 6000},
            "common_reference": {"operator": "median"},
        }
    }

Without the section, or without ``si_pipeline``, ``aeon_default`` is used.

Registering a pipeline
----------------------
Register a pipeline under a name, from this module or any other::

    from aeon.dj_pipeline.utils.ephys_preprocessing import register_preprocessing

    register_preprocessing(
        "my_pipeline",
        {
            "highpass_filter": {"freq_min": 400},
            "common_reference": {"operator": "median"},
        },
    )

Registration happens on import. A pipeline registered outside this module must be
imported before running ``PreProcessing.populate()``.

Names are unique; registering an existing name raises. Once a paramset references a
name, don't change its pipeline - register the modified pipeline under a new name.
"""

from typing import Any

PipelineDict = dict[str, dict[str, Any]]

_PREPROCESSING_PIPELINES: dict[str, PipelineDict] = {}

DEFAULT_PREPROCESSING_PIPELINE = "aeon_default"

# Keys allowed in a paramset's SI_PREPROCESSING_PARAMS section.
_PREPROCESSING_PARAMS_KEYS = {"si_pipeline"}


def register_preprocessing(name: str, pipeline: PipelineDict) -> None:
    """Register a SpikeInterface preprocessing pipeline dict under ``name``."""
    if name in _PREPROCESSING_PIPELINES:
        raise ValueError(f"Preprocessing pipeline '{name}' is already registered.")
    _PREPROCESSING_PIPELINES[name] = pipeline


def resolve_preprocessing_pipeline(preprocessing_params: dict[str, Any]) -> PipelineDict:
    """Return the pipeline dict a paramset's ``SI_PREPROCESSING_PARAMS`` section selects.

    Raises:
        ValueError: An unknown key in the section, or an unknown registered name.
        TypeError: ``si_pipeline`` is neither a name nor a dict, or the pipeline has a
            step name SpikeInterface doesn't support.
    """
    from spikeinterface.preprocessing import PreprocessingPipeline

    unknown_keys = set(preprocessing_params) - _PREPROCESSING_PARAMS_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown SI_PREPROCESSING_PARAMS key(s) {sorted(unknown_keys)}. "
            f"Allowed: {sorted(_PREPROCESSING_PARAMS_KEYS)}"
        )

    si_pipeline = preprocessing_params.get("si_pipeline", DEFAULT_PREPROCESSING_PIPELINE)
    if isinstance(si_pipeline, str):
        if si_pipeline not in _PREPROCESSING_PIPELINES:
            raise ValueError(
                f"Unknown preprocessing pipeline '{si_pipeline}'. "
                f"Registered: {sorted(_PREPROCESSING_PIPELINES)}"
            )
        pipeline = _PREPROCESSING_PIPELINES[si_pipeline]
    elif isinstance(si_pipeline, dict):
        pipeline = si_pipeline
    else:
        raise TypeError(
            f"si_pipeline must be a registered name or a pipeline dict, got {type(si_pipeline).__name__}."
        )

    PreprocessingPipeline(pipeline)  # raises TypeError on an unsupported step name
    return pipeline


def insert_bad_channels_into_pipeline(si_pipeline, electrodes_df, recording):
    """Gives a PreProcessingPipeline with bad channels from electrodes_df.

    Usually, one would use algorithms to figure out which channels are bad.
    We generally do not want to do this block-by-block because it might
    give different results for different blocks, leading to complications
    down the line. Instead, we say which channels are bad "globally" using
    `ephys.ElectrodeConfig.Electrode.label`. In this function, we intercept
    the SpikeInterface bad channels steps to supply them with this information.
    """
    for pp_name, pp_kwargs in si_pipeline:
        if pp_name in ["detect_and_interpolate_bad_channels", "detect_and_remove_bad_channels"]:
            bad_channel_indices = electrodes_df.query('label == "bad"')["channel_idx"]
            bad_channel_ids = recording.channel_ids[bad_channel_indices]
            pp_kwargs["bad_channel_ids"] = bad_channel_ids

    return si_pipeline


# ---- Registered pipelines ----

register_preprocessing(
    "aeon_default",
    {
        "bandpass_filter": {"freq_min": 300, "freq_max": 6000},
        "common_reference": {"operator": "median"},
    },
)
# No preprocessing in PreProcessing, e.g. to leave it to the sorter.
register_preprocessing("none", {})
