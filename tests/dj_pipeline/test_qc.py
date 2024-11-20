"""Tests for the QC pipeline."""

import pytest


@pytest.mark.qc
def test_camera_qc_ingestion(test_params, pipeline, camera_qc_ingestion):
    qc = pipeline["qc"]

    assert len(qc.CameraQC()) == test_params["camera_qc_count"]
