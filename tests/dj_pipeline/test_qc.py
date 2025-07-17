"""Tests for the QC pipeline."""

import pytest


@pytest.mark.qc()
@pytest.mark.usefixtures("_camera_qc_ingestion")
def test_camera_qc_ingestion(test_params, pipeline):
    qc = pipeline["qc"]

    assert len(qc.CameraQC()) == test_params["camera_qc_count"]
