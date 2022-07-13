from pytest import mark


@mark.instantiation
def test_pipeline_instantiation(pipeline):
    
    assert hasattr(pipeline["acquisition"], "FoodPatchEvent")
    assert hasattr(pipeline["lab"], "Arena")
    assert hasattr(pipeline["qc"], "CameraQC")
    assert hasattr(pipeline["report"], "InArenaSummaryPlot")
    assert hasattr(pipeline["subject"], "Subject")
    assert hasattr(pipeline["tracking"], "CameraTracking")
    
    
@mark.instantiation
def test_exp_creation(pipeline, test_params):
    acquisition = pipeline["acquisition"]
    
    experiment_name = test_params["experiment_name"]
    assert acquisition.Experiment.fetch1("experiment_name") == experiment_name
    raw_dir = (
        acquisition.Experiment.Directory
        & {"experiment_name": experiment_name, "directory_type": "raw"}
    ).fetch1("directory_path")
    assert raw_dir == test_params["raw_dir"]
    exp_subjects = (
        acquisition.Experiment.Subject & {"experiment_name": experiment_name}
    ).fetch("subject")
    assert len(exp_subjects) == test_params["subject_count"]
    assert "BAA-1100701" in exp_subjects
