"""Test tracking pipeline."""

import datetime
import pathlib

import datajoint as dj
import numpy as np
import pytest

logger = dj.logger


index = 0
column_name = "position_x"  # data column to run test on
file_name = "exp0.2-r0-20220524090000-21053810-20220524082942-0-0.npy"
# test file to be saved with save_test_data


def save_test_data(pipeline, test_params):
    """Save test dataset fetched from tracking.CameraTracking.Object."""
    tracking = pipeline["tracking"]

    key = tracking.CameraTracking.Object().fetch("KEY")[index]
    file_name = (
        "-".join(
            [
                (
                    v.strftime("%Y%m%d%H%M%S")
                    if isinstance(v, datetime.datetime)
                    else str(v)
                )
                for v in key.values()
            ]
        )
        + ".npy"
    )

    data = (tracking.CameraTracking.Object() & key).fetch(column_name)[0]
    test_file = test_params["test_dir"] + "/" + file_name
    np.save(test_file, data)

    return test_file


@pytest.mark.ingestion
@pytest.mark.tracking
@pytest.mark.usefixtures("_camera_tracking_ingestion")
def test_camera_tracking_ingestion(test_params, pipeline):
    tracking = pipeline["tracking"]

    assert (
        len(tracking.CameraTracking.Object())
        == test_params["camera_tracking_object_count"]
    )

    key = tracking.CameraTracking.Object().fetch("KEY")[index]
    file_name = (
        "-".join(
            [
                (
                    v.strftime("%Y%m%d%H%M%S")
                    if isinstance(v, datetime.datetime)
                    else str(v)
                )
                for v in key.values()
            ]
        )
        + ".npy"
    )

    test_file = pathlib.Path(test_params["test_dir"] + "/" + file_name)
    assert test_file.exists()

    print(f"\nTesting {file_name}")

    data = np.load(test_file)
    assert np.allclose(
        data,
        (tracking.CameraTracking.Object() & key).fetch(column_name)[0],
        equal_nan=True,
    )
