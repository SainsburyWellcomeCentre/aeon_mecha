from pytest import mark


@mark.ingestion
def test_epoch_chunk_ingestion(test_params, pipeline, epoch_chunk_ingestion):
    acquisition = pipeline["acquisition"]

    assert (
        len(acquisition.Epoch & {"experiment_name": test_params["experiment_name"]})
        == test_params["epoch_count"]
    )
    assert (
        len(acquisition.Chunk & {"experiment_name": test_params["experiment_name"]})
        == test_params["chunk_count"]
    )


@mark.experimentlog
def test_experimentlog_ingestion(test_params, pipeline, experimentlog_ingestion):
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


@mark.qc
def test_camera_qc_ingestion(test_params, pipeline, camera_qc_ingestion):

    qc = pipeline["qc"]

    assert len(qc.CameraQC()) == test_params["camera_qc_count"]


@mark.tracking
def test_camera_tracking_ingestion(
    test_params,
    pipeline,
    camera_tracking_ingestion,
):

    tracking = pipeline["tracking"]

    import pdb

    pdb.set_trace()
    assert len(tracking.CameraTracking()) == test_params["camera_tracking_count"]
