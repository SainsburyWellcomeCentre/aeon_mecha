"""Unit test for multitag_view's manual-label writing (regression).

A unit carrying both a quality label and a tag used to lose its quality label on
save: the tag was written under a nested ``labels`` key while the quality label was
written flat, and ``CurationModel`` rebuilds each entry from the nested dict alone,
silently dropping the flat quality (``SortedSpikes`` then falls back to KSLabel).
multitag_view now writes tags flat, so both survive a save.

multitag_view subclasses spikeinterface_gui's ``ViewBase`` (only installed in the
curation GUI environment) and lives in the dj_pipeline package (whose import
activates schemas), so it is loaded here directly from its file with the GUI base
stubbed, keeping this a database- and GUI-free unit test.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "aeon" / "dj_pipeline" / "scripts" / "multitag_view.py"
)

_LABEL_DEFINITIONS = {
    "quality": {"label_options": ["good", "MUA", "noise"], "exclusive": True},
    "tags": {"label_options": ["flag", "intermittent", "amplitude drift"], "exclusive": False},
}


def _load_multitag_view():
    """Load multitag_view.py in isolation, stubbing spikeinterface_gui if it is absent."""
    try:
        import spikeinterface_gui.view_base  # noqa: F401
    except ModuleNotFoundError:
        pkg = types.ModuleType("spikeinterface_gui")
        view_base = types.ModuleType("spikeinterface_gui.view_base")
        view_base.ViewBase = object
        pkg.view_base = view_base
        sys.modules["spikeinterface_gui"] = pkg
        sys.modules["spikeinterface_gui.view_base"] = view_base
    spec = importlib.util.spec_from_file_location("_multitag_view_undertest", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize(manual_labels):
    """Return {unit_id: labels-dict} after CurationModel's manual-label normalization."""
    from spikeinterface.curation.curation_model import CurationModel

    model = CurationModel(
        format_version="2",
        unit_ids=[5, 7],
        label_definitions=_LABEL_DEFINITIONS,
        manual_labels=manual_labels,
    )
    return {ml.unit_id: ml.labels for ml in model.manual_labels}


class _FakeView:
    """Minimal stand-in exposing the one attribute the tag methods read/write."""

    def __init__(self, manual_labels):
        self.controller = types.SimpleNamespace(curation_data={"manual_labels": manual_labels})


def test_tagging_a_labeled_unit_keeps_its_quality_label():
    mtv = _load_multitag_view()
    # a unit the curator already labeled "good" (written flat, as the controller does)
    view = _FakeView([{"unit_id": 5, "quality": ["good"]}])

    mtv.MultiTagView._set_unit_features(view, 5, ["flag"])
    entry = view.controller.curation_data["manual_labels"][0]

    # tags are written flat, the same shape the controller uses for quality (no nested "labels")
    assert "labels" not in entry
    assert entry["quality"] == ["good"]
    assert entry[mtv.TAGS_CATEGORY] == ["flag"]
    assert mtv.MultiTagView._get_unit_features(view, 5) == ["flag"]

    # and the quality label survives CurationModel normalization alongside the tag
    labels = _normalize([dict(entry)])[5]
    assert labels["quality"] == ["good"]
    assert labels["tags"] == ["flag"]


def test_tagging_then_labeling_also_keeps_quality():
    # the other ordering Thinh flagged: tag a unit first (creating a brand-new, flat entry
    # via the append path), then apply a quality label the way the controller does (also flat).
    mtv = _load_multitag_view()
    view = _FakeView([])

    mtv.MultiTagView._set_unit_features(view, 5, ["flag"])
    entry = view.controller.curation_data["manual_labels"][0]
    assert entry == {"unit_id": 5, "tags": ["flag"]}  # new entry written flat, no nested "labels"

    # controller.set_label_to_unit then adds the quality label flat on the same entry
    entry["quality"] = ["good"]

    labels = _normalize([dict(entry)])[5]
    assert labels["quality"] == ["good"]
    assert labels["tags"] == ["flag"]


def test_nested_labels_shape_drops_quality():
    """Documents why tags must be written flat: a nested "labels" entry loses the flat quality."""
    labels = _normalize([{"unit_id": 5, "quality": ["good"], "labels": {"tags": ["flag"]}}])[5]
    assert "quality" not in labels
    assert labels["tags"] == ["flag"]
