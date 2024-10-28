""" Tests for the acquisition pipeline. """

from pytest import mark
import datajoint as dj

logger = dj.logger


@mark.ingestion
def test_epoch_chunk_ingestion(test_params, pipeline, epoch_chunk_ingestion):
    acquisition = pipeline["acquisition"]
    epoch_count = len(
        acquisition.Epoch & {"experiment_name": test_params["experiment_name"]}
    )
    chunk_count = len(
        acquisition.Chunk & {"experiment_name": test_params["experiment_name"]}
    )
    if epoch_count != test_params["epoch_count"]:
        raise AssertionError(
            f"Expected {test_params['epoch_count']} epochs, but got {epoch_count}."
        )

    if chunk_count != test_params["chunk_count"]:
        raise AssertionError(
            f"Expected {test_params['chunk_count']} chunks, but got {chunk_count}."
        )


@mark.ingestion
def test_experimentlog_ingestion(
    test_params, pipeline, epoch_chunk_ingestion, experimentlog_ingestion
):
    acquisition = pipeline["acquisition"]

    experiment_log_message_count = len(
        acquisition.ExperimentLog.Message
        & {"experiment_name": test_params["experiment_name"]}
    )
    if experiment_log_message_count != test_params["experiment_log_message_count"]:
        raise AssertionError(
            f"Expected {test_params['experiment_log_message_count']} experiment log messages, but got {experiment_log_message_count}."
        )
    subject_enter_exit_count = len(
        acquisition.SubjectEnterExit.Time
        & {"experiment_name": test_params["experiment_name"]}
    )
    if subject_enter_exit_count != test_params["subject_enter_exit_count"]:
        raise AssertionError(
            f"Expected {test_params['subject_enter_exit_count']} subject enter/exit events, but got {subject_enter_exit_count}."
        )

    subject_weight_time_count = len(
        acquisition.SubjectWeight.WeightTime
        & {"experiment_name": test_params["experiment_name"]}
    )
    if subject_weight_time_count != test_params["subject_weight_time_count"]:
        raise AssertionError(
            f"Expected {test_params['subject_weight_time_count']} subject weight events, but got {subject_weight_time_count}."
        )
