"""Preprocessing methods for spike sorting, selectable per SortingParamSet.

``spike_sorting.PreProcessing`` looks a method up by name and applies it to the
recording, after the fixed unsigned-to-signed conversion and before the result is
saved for sorting. A paramset picks the method and its arguments::

    params = {
        "SI_PREPROCESSING_METHOD": "aeon_default",  # default if omitted
        "SI_PREPROCESSING_PARAMS": {"freq_min": 250},  # passed as **kwargs
        ...
    }

Adding a method
---------------
Write a function in this module and register it under a name::

    @register_preprocessing("my_method")
    def my_method(recording, my_option: float = 1.0, **kwargs) -> Any:
        _reject_unused_kwargs("my_method", kwargs)

        import spikeinterface.preprocessing as spre

        recording = spre.highpass_filter(recording, freq_min=my_option)
        return recording

Rules for a method:

- Take a SpikeInterface recording plus keyword arguments, and return a recording.
- Keep it lazy: chain SpikeInterface preprocessing steps rather than loading traces.
  The result is pickled and reloaded by the sorting step in another process.
- Expose tunable values as keyword arguments with defaults, and call
  ``_reject_unused_kwargs`` so a misspelled key fails instead of being ignored.
- Once a paramset references a name, don't change what it does. Register modified
  logic under a new name.

This module has no database dependency, so methods can be unit-tested directly.
"""

from collections.abc import Callable
from typing import Any

_PREPROCESSING_METHODS: dict[str, Callable[..., Any]] = {}

DEFAULT_PREPROCESSING_METHOD = "aeon_default"


def register_preprocessing(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a preprocessing function under ``name`` for use in SortingParamSet."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in _PREPROCESSING_METHODS:
            raise ValueError(f"Preprocessing method '{name}' is already registered.")
        _PREPROCESSING_METHODS[name] = func
        return func

    return decorator


def get_preprocessing_method(name: str) -> Callable[..., Any]:
    """Return the preprocessing function registered under ``name``."""
    try:
        return _PREPROCESSING_METHODS[name]
    except KeyError:
        raise ValueError(
            f"Unknown SI_PREPROCESSING_METHOD '{name}'. "
            f"Registered methods: {sorted(_PREPROCESSING_METHODS)}"
        ) from None


def _reject_unused_kwargs(method: str, kwargs: dict[str, Any]) -> None:
    """Raise on SI_PREPROCESSING_PARAMS keys that ``method`` does not use."""
    if kwargs:
        raise ValueError(
            f"Preprocessing method '{method}' does not accept SI_PREPROCESSING_PARAMS: {sorted(kwargs)}"
        )


# ---- Methods ----


# "ephys_preproc": the name earlier runbooks wrote into SI_PREPROCESSING_METHOD, when the
# key was ignored and this recipe always ran. Kept so those paramsets resolve unchanged.
@register_preprocessing("ephys_preproc")
@register_preprocessing("aeon_default")
def aeon_default(
    recording, freq_min: float = 300, freq_max: float = 6000, operator: str = "median", **kwargs
) -> Any:
    """Bandpass filter (default 300-6000 Hz), then common average reference (default median)."""
    _reject_unused_kwargs("aeon_default", kwargs)

    import spikeinterface.preprocessing as spre

    recording = spre.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max)
    recording = spre.common_reference(recording=recording, operator=operator)
    return recording


@register_preprocessing("none")
def no_preprocessing(recording, **kwargs) -> Any:
    """Return the recording unchanged, e.g. to leave preprocessing to the sorter."""
    _reject_unused_kwargs("none", kwargs)
    return recording
