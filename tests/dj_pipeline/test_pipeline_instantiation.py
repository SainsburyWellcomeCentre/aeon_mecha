""" Tests for pipeline instantiation and experiment creation """

import datajoint as dj

logger = dj.logger

import pytest


@pytest.mark.instantiation
def test_pipeline_instantiation(pipeline):
    if not hasattr(pipeline["acquisition"], "FoodPatchEvent"):
        raise AssertionError("Pipeline acquisition does not have 'FoodPatchEvent' attribute.")

    if not hasattr(pipeline["lab"], "Arena"):
        raise AssertionError("Pipeline lab does not have 'Arena' attribute.")

    if not hasattr(pipeline["qc"], "CameraQC"):
        raise AssertionError("Pipeline qc does not have 'CameraQC' attribute.")

    if not hasattr(pipeline["report"], "InArenaSummaryPlot"):
        raise AssertionError("Pipeline report does not have 'InArenaSummaryPlot' attribute.")

    if not hasattr(pipeline["subject"], "Subject"):
        raise AssertionError("Pipeline subject does not have 'Subject' attribute.")

    if not hasattr(pipeline["tracking"], "CameraTracking"):
        raise AssertionError("Pipeline tracking does not have 'CameraTracking' attribute.")


@pytest.mark.instantiation
def test_experiment_creation(test_params, pipeline, experiment_creation):
    acquisition = pipeline["acquisition"]

    experiment_name = test_params["experiment_name"]
    fetched_experiment_name = acquisition.Experiment.fetch1("experiment_name")
    if fetched_experiment_name != experiment_name:
        raise AssertionError(
            f"Expected experiment name '{experiment_name}', but got '{fetched_experiment_name}'."
        )

    raw_dir = (
        acquisition.Experiment.Directory & {"experiment_name": experiment_name, "directory_type": "raw"}
    ).fetch1("directory_path")
    if raw_dir != test_params["raw_dir"]:
        raise AssertionError(f"Expected raw directory '{test_params['raw_dir']}', but got '{raw_dir}'.")

    exp_subjects = (acquisition.Experiment.Subject & {"experiment_name": experiment_name}).fetch("subject")
    if len(exp_subjects) != test_params["subject_count"]:
        raise AssertionError(
            f"Expected subject count {test_params['subject_count']}, but got {len(exp_subjects)}."
        )

    if "BAA-1100701" not in exp_subjects:
        raise AssertionError("Expected subject 'BAA-1100701' not found in experiment subjects.")
