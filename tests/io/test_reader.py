"""Tests for the Pose stream."""

from pathlib import Path

import pytest

import aeon
from aeon.schema.schemas import social02, social03

pose_path = Path(__file__).parent.parent / "data" / "pose"

@pytest.mark.api
def test_Pose_read_local_model_dir():
    """Test that the Pose stream can read a local model directory."""
    data = aeon.load(pose_path, social02.CameraTop.Pose)
    assert len(data) > 0

@pytest.mark.api
def test_Pose_read_local_model_dir_with_register_prefix():
    """Test that the Pose stream can read a local model directory with a register prefix."""
    data = aeon.load(pose_path, social03.CameraTop.Pose)
    assert len(data) > 0

if __name__ == "__main__":
    pytest.main()
