""" Tests for the QC pipeline. """

from pytest import mark
import datajoint as dj

logger = dj.logger


@mark.qc
def test_camera_qc_ingestion(test_params, pipeline, camera_qc_ingestion):
    qc = pipeline["qc"]

    camera_qc_count = len(qc.CameraQC())
    expected_camera_qc_count = test_params["camera_qc_count"]

    if camera_qc_count != expected_camera_qc_count:
        raise AssertionError(
            f"Expected camera QC count {expected_camera_qc_count}, but got {camera_qc_count}."
        )
