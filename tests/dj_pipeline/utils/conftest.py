"""Fixtures for load_metadata tests.

Uses real @data_reader decorator and Pydantic base classes from swc.aeon.schema (swc-aeon).
These are the same classes used in production (e.g., aeon_exp_foragingABC/rig.py).
No mocking of decorator behavior - tests run against actual production patterns.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Real Reader classes from swc-aeon (aeon_api)
from swc.aeon.io.reader import Csv, Harp, Video

# Real Pydantic base classes from swc-aeon (aeon_api)
# These have _resolve_pattern_prefix required by @data_reader
from swc.aeon.schema import BaseSchema, data_reader
from swc.aeon.schema.foraging import UndergroundFeeder
from swc.aeon.schema.video import SpinnakerCamera

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
METADATA_FIXTURES_DIR = FIXTURES_DIR / "metadata"


class TestCamera(SpinnakerCamera):
    """Test camera using real @data_reader decorator."""

    @data_reader
    def video(self, pattern) -> Video:
        return Video(f"{pattern}")

    @data_reader
    def position(self, pattern) -> Harp:
        return Harp(f"{pattern}_200", columns=["x", "y"])


class TestFeeder(UndergroundFeeder):
    """Test feeder using real @data_reader decorator."""

    @data_reader
    def beam_break(self, pattern) -> Harp:
        return Harp(f"{pattern}_32", columns=["state"])

    @data_reader
    def encoder(self, pattern) -> Csv:
        return Csv(f"{pattern}_90", columns=["angle", "intensity"])


class TestRig(BaseSchema):
    """Test Rig using real Pydantic base classes with real @data_reader."""

    cameras: dict[str, TestCamera] = {}
    feeders: dict[str, TestFeeder] = {}


@pytest.fixture
def mock_device_class():
    """Device class using real @data_reader decorator.

    Uses real SpinnakerCamera base class and @data_reader decorator
    from swc.aeon.schema (swc-aeon package).
    """
    return TestCamera


@pytest.fixture
def test_rig():
    """Create a test Rig with real Pydantic structure and real @data_reader."""
    return TestRig(
        cameras={
            "CameraTop": TestCamera(serial_number="21053810"),
            "CameraSide": TestCamera(serial_number="21053811"),
        },
        feeders={
            "Feeder1": TestFeeder(port_name="COM3"),
            "Feeder2": TestFeeder(port_name="COM4"),
        },
    )


# -----------------------------------------------------------------------------
# Sample Metadata Fixtures (small config files checked into git)
# -----------------------------------------------------------------------------


@pytest.fixture
def foraging_abc_metadata_path():
    """Path to sample ForagingABC metadata fixture file."""
    return METADATA_FIXTURES_DIR / "ForagingABC_Metadata.json"


@pytest.fixture
def foraging_abc_metadata(foraging_abc_metadata_path):
    """Load sample ForagingABC metadata as a dict."""
    return json.loads(foraging_abc_metadata_path.read_text())


@pytest.fixture
def mock_xarray_dataset():
    """Mock an xarray.Dataset for testing the ``<xarray@xarray_store>`` codec."""
    n, n_ch = 20, 4
    return xr.Dataset(
        {
            "signal": (("time", "channel"), np.random.randn(n, n_ch).astype("float32")),
            "flag": (("time",), np.random.rand(n) > 0.5),
        },
        coords={
            "time": pd.date_range("2026-06-03", periods=n, freq="20ms"),
            "channel": np.arange(n_ch),
        },
        attrs={"fs": 50.0, "note": "mock"},
    )


@pytest.fixture
def foraging_abc_rig_config(foraging_abc_metadata):
    """Extract the rig config section from ForagingABC sample metadata."""
    return foraging_abc_metadata["rig"]


@pytest.fixture
def foraging_abc_rig(foraging_abc_metadata):
    """Construct a typed ForagingABC Rig from sample metadata.

    Loads the real swc.aeon_exp.foragingABC.experiment:Experiment Pydantic class
    and validates the metadata into a typed Rig. Mirrors the production path
    (acquisition.EpochConfig.make).
    """
    pytest.importorskip("swc.aeon_exp")
    from aeon.dj_pipeline.utils.load_metadata import get_experiment_pydantic

    experiment_class = get_experiment_pydantic("swc.aeon_exp.foragingABC.experiment:Experiment")
    experiment = experiment_class.model_validate(foraging_abc_metadata)
    return experiment.rig
