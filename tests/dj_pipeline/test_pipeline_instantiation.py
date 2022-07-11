from pytest import mark


@mark.instantiation
def test_pipeline_instantiation(pipeline):
    acquisition = pipeline["acquisition"]
    report = pipeline["report"]
    
    assert hasattr(acquisition, "FoodPatchEvent")
    assert hasattr(report, "InArenaSummaryPlot")
    

@mark.instantiation
def test_exp_creation(test_params, pipeline, exp_creation):
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
