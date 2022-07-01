from . import (
    dj_config,
    pipeline,
    new_experiment,
    test_variables,
    epoch_chunk_ingestion,
    experimentlog_ingestion,
)


def test_epoch_chunk_ingestion(epoch_chunk_ingestion, test_variables, pipeline):
    acquisition = pipeline["acquisition"]

    assert (
        len(acquisition.Epoch & {"experiment_name": test_variables["experiment_name"]})
        == test_variables["epoch_count"]
    )
    assert (
        len(acquisition.Chunk & {"experiment_name": test_variables["experiment_name"]})
        == test_variables["chunk_count"]
    )


def test_epoch_chunk_ingestion(experimentlog_ingestion, test_variables, pipeline):
    acquisition = pipeline["acquisition"]

    assert (
        len(
            acquisition.ExperimentLog.Message
            & {"experiment_name": test_variables["experiment_name"]}
        )
        == test_variables["experiment_log_message_count"]
    )
    assert (
        len(
            acquisition.SubjectEnterExit.Time
            & {"experiment_name": test_variables["experiment_name"]}
        )
        == test_variables["subject_enter_exit_count"]
    )
    assert (
        len(
            acquisition.SubjectWeight.WeightTime
            & {"experiment_name": test_variables["experiment_name"]}
        )
        == test_variables["subject_weight_time_count"]
    )
