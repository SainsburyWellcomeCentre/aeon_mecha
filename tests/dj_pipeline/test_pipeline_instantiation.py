from . import dj_config, pipeline, new_experiment, test_variables


def test_pipeline_instantiation(pipeline):
    acquisition = pipeline["acquisition"]
    report = pipeline["report"]

    assert hasattr(acquisition, "FoodPatchEvent")
    assert hasattr(report, "InArenaSummaryPlot")


def test_experiment_creation(pipeline, new_experiment, test_variables):
    acquisition = pipeline["acquisition"]
    experiment_name = test_variables["experiment_name"]
    assert acquisition.Experiment.fetch1("experiment_name") == experiment_name
    raw_dir = (
        acquisition.Experiment.Directory
        & {"experiment_name": experiment_name, "directory_type": "raw"}
    ).fetch1("directory_path")
    assert raw_dir == test_variables["raw_dir"]
    exp_subjects = (
        acquisition.Experiment.Subject & {"experiment_name": experiment_name}
    ).fetch("subject")
    assert len(exp_subjects) == test_variables["subject_count"]
    assert "BAA-1100701" in exp_subjects
