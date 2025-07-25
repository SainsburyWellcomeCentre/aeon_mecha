"""Tests for the acquisition pipeline."""

import pytest


@pytest.mark.ingestion
@pytest.mark.usefixtures("_epoch_chunk_ingestion")
def test_epoch_chunk_ingestion(test_params, pipeline):
    acquisition = pipeline["acquisition"]

    assert (
        len(acquisition.Epoch & {"experiment_name": test_params["experiment_name"]})
        == test_params["epoch_count"]
    )
    assert (
        len(acquisition.Chunk & {"experiment_name": test_params["experiment_name"]})
        == test_params["chunk_count"]
    )


@pytest.mark.ingestion
@pytest.mark.usefixtures("_epoch_chunk_ingestion", "_experimentlog_ingestion")
def test_experimentlog_ingestion(test_params, pipeline):
    acquisition = pipeline["acquisition"]

    assert (
        len(
            acquisition.ExperimentLog.Message
            & {"experiment_name": test_params["experiment_name"]}
        )
        == test_params["experiment_log_message_count"]
    )
    assert (
        len(
            acquisition.SubjectEnterExit.Time
            & {"experiment_name": test_params["experiment_name"]}
        )
        == test_params["subject_enter_exit_count"]
    )
    assert (
        len(
            acquisition.SubjectWeight.WeightTime
            & {"experiment_name": test_params["experiment_name"]}
        )
        == test_params["subject_weight_time_count"]
    )
