from pathlib import Path

import pytest
from pytest import mark

import aeon
from aeon.schema.schemas import social02

pose_path = Path(__file__).parent.parent / "data" / "pose"


@mark.api
def test_Pose_read_local_model_dir():
    data = aeon.load(pose_path, social02.CameraTop.Pose)
    assert len(data) > 0


if __name__ == "__main__":
    pytest.main()
