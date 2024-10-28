""" Test tracking pipeline. """

import datetime
import pathlib

import numpy as np
from pytest import mark
import datajoint as dj

logger = dj.logger


index = 0
column_name = "position_x"  # data column to run test on
file_name = "exp0.2-r0-20220524090000-21053810-20220524082942-0-0.npy"  # test file to be saved with save_test_data


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


@mark.ingestion
@mark.tracking
def test_camera_tracking_ingestion(test_params, pipeline, camera_tracking_ingestion):
    tracking = pipeline["tracking"]

    camera_tracking_object_count = len(tracking.CameraTracking.Object())
    if camera_tracking_object_count != test_params["camera_tracking_object_count"]:
        raise AssertionError(
            f"Expected camera tracking object count {test_params['camera_tracking_object_count']},but got {camera_tracking_object_count}."
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
    if not test_file.exists():
        raise AssertionError(f"Test file '{test_file}' does not exist.")

    print(f"\nTesting {file_name}")

    data = np.load(test_file)
    expected_data = (tracking.CameraTracking.Object() & key).fetch(column_name)[0]

    if not np.allclose(data, expected_data, equal_nan=True):
        raise AssertionError(
            f"Loaded data does not match the expected data.nExpected: {expected_data}, but got: {data}."
        )
