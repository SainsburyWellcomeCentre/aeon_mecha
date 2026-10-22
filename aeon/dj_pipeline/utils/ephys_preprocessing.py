"""Preprocessing pipelines for spike sorting, selectable per SortingParamSet.

A preprocessing pipeline is a SpikeInterface pipeline dict: ordered step names, each
mapped to that step's keyword arguments (omitted arguments take SpikeInterface's
defaults). See
https://spikeinterface.readthedocs.io/en/stable/how_to/build_pipeline_with_dicts.html

``spike_sorting.PreProcessing`` applies the selected pipeline to the recording, after
the fixed unsigned-to-signed conversion and before the result is saved for sorting.
A paramset selects a pipeline in one of two ways:

1. A registered pipeline, by name, with optional per-step argument overrides::

    params = {
        "SI_PREPROCESSING_METHOD": "aeon_default",  # default if omitted
        "SI_PREPROCESSING_PARAMS": {"bandpass_filter": {"freq_min": 250}},
        ...
    }

2. A pipeline written inline::

    params = {
        "SI_PREPROCESSING_PIPELINE": {
            "bandpass_filter": {"freq_min": 300, "freq_max": 6000},
            "common_reference": {"operator": "median"},
        },
        ...
    }

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
imported before inserting a paramset that names it and before running
``PreProcessing.populate()``.

Names are unique; registering an existing name raises. Once a paramset references a
name, don't change its pipeline - register the modified pipeline under a new name.
"""

from typing import Any

PipelineDict = dict[str, dict[str, Any]]

_PREPROCESSING_PIPELINES: dict[str, PipelineDict] = {}

DEFAULT_PREPROCESSING_METHOD = "aeon_default"


def register_preprocessing(name: str, pipeline: PipelineDict) -> None:
    """Register a SpikeInterface preprocessing pipeline dict under ``name``."""
    if name in _PREPROCESSING_PIPELINES:
        raise ValueError(f"Preprocessing method '{name}' is already registered.")
    _PREPROCESSING_PIPELINES[name] = pipeline


_RECOGNISED_KEYS = frozenset(
    {"SI_PREPROCESSING_METHOD", "SI_PREPROCESSING_PARAMS", "SI_PREPROCESSING_PIPELINE"}
)


def resolve_preprocessing_pipeline(params: dict[str, Any]) -> PipelineDict:
    """Return the pipeline dict a SortingParamSet's ``params`` selects.

    Raises:
        ValueError: An unrecognised ``SI_PREPROCESSING*`` key, an unknown method name,
            an override for a step the pipeline doesn't have, or an inline pipeline
            combined with a method or overrides.
        TypeError: A step name SpikeInterface doesn't support.
    """
    from spikeinterface.preprocessing import PreprocessingPipeline

    # A typo in a KEY is as silent as a typo in a value: SI_PREPROCESSING_METHOOD
    # falls straight through to aeon_default, and the paramset then reads as if it
    # asked for something else while bandpass + CAR run.
    unknown_keys = {k for k in params if k.startswith("SI_PREPROCESSING")} - _RECOGNISED_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unrecognised preprocessing key(s) {sorted(unknown_keys)}. "
            f"Recognised: {sorted(_RECOGNISED_KEYS)}"
        )

    if "SI_PREPROCESSING_PIPELINE" in params:
        conflicting = {"SI_PREPROCESSING_METHOD", "SI_PREPROCESSING_PARAMS"} & set(params)
        if conflicting:
            raise ValueError(f"SI_PREPROCESSING_PIPELINE can't be combined with {sorted(conflicting)}.")
        pipeline = params["SI_PREPROCESSING_PIPELINE"]
    else:
        name = params.get("SI_PREPROCESSING_METHOD", DEFAULT_PREPROCESSING_METHOD)
        if name not in _PREPROCESSING_PIPELINES:
            raise ValueError(
                f"Unknown SI_PREPROCESSING_METHOD '{name}'. "
                f"Registered methods: {sorted(_PREPROCESSING_PIPELINES)}"
            )
        registered = _PREPROCESSING_PIPELINES[name]
        overrides = params.get("SI_PREPROCESSING_PARAMS", {})
        unknown_steps = set(overrides) - set(registered)
        if unknown_steps:
            raise ValueError(
                f"SI_PREPROCESSING_PARAMS step(s) {sorted(unknown_steps)} are not in "
                f"the '{name}' pipeline: {list(registered)}"
            )
        pipeline = {step: {**kwargs, **overrides.get(step, {})} for step, kwargs in registered.items()}

    PreprocessingPipeline(pipeline)  # raises TypeError on an unsupported step name
    return pipeline


def apply_preprocessing(recording: Any, params: dict[str, Any]) -> Any:
    """Apply the pipeline ``params`` selects to ``recording`` and return the (lazy) result."""
    from spikeinterface.preprocessing import apply_preprocessing_pipeline

    return apply_preprocessing_pipeline(recording, resolve_preprocessing_pipeline(params))


# ---- Registered pipelines ----

_AEON_DEFAULT: PipelineDict = {
    "bandpass_filter": {"freq_min": 300, "freq_max": 6000},
    "common_reference": {"operator": "median"},
}

register_preprocessing("aeon_default", _AEON_DEFAULT)
# "ephys_preproc": the name earlier runbooks wrote into SI_PREPROCESSING_METHOD, when the
# key was ignored and this pipeline always ran. Kept so those paramsets resolve unchanged.
register_preprocessing("ephys_preproc", _AEON_DEFAULT)
# No preprocessing in PreProcessing, e.g. to leave it to the sorter.
register_preprocessing("none", {})
