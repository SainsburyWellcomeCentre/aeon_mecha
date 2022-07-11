"""
# run all tests:
# pytest -sv --cov-report term-missing --cov=aeon_mecha -p no:warnings tests/dj_pipeline

# run one test, debug:
# pytest [above options] --pdb tests/dj_pipeline/test_ingestion.py -k <function_name>

# run test on marker:
# pytest -m <marker_name>
"""

import pathlib

import datajoint as dj
import pytest

_tear_down = True
_populate_settings = {"suppress_errors": True}


@pytest.fixture(autouse=True, scope="session")
def test_params():

    return {
        "start_ts": "2022-05-24 08:29:42",
        "end_ts": "2022-05-24 15:59:00",
        "experiment_name": "exp0.2-r0",
        "raw_dir": "aeon/data/raw/AEON2/experiment0.2",
        "qc_dir": "aeon/data/qc/AEON2/experiment0.2",
        "subject_count": 5,
        "epoch_count": 1,
        "chunk_count": 9,
        "experiment_log_message_count": 0,
        "subject_enter_exit_count": 0,
        "subject_weight_time_count": 0,
        "camera_qc_count": 56,
        "camera_tracking_count": 7,
    }


def load_pipeline():

    from aeon.dj_pipeline import (
        acquisition,
        analysis,
        lab,
        qc,
        report,
        subject,
        tracking,
    )

    return {
        "subject": subject,
        "lab": lab,
        "acquisition": acquisition,
        "qc": qc,
        "tracking": tracking,
        "analysis": analysis,
        "report": report,
    }


def drop_schema():

    _pipeline = load_pipeline()

    _pipeline["report"].schema.drop()
    _pipeline["analysis"].schema.drop()
    _pipeline["tracking"].schema.drop()
    _pipeline["qc"].schema.drop()
    _pipeline["acquisition"].schema.drop()
    _pipeline["subject"].schema.drop()
    _pipeline["lab"].schema.drop()


@pytest.fixture(autouse=True, scope="session")
def dj_config():
    """If dj_local_config exists, load"""
    dj_config_fp = pathlib.Path("dj_local_conf.json")
    assert dj_config_fp.exists()
    dj.config.load(dj_config_fp)
    dj.config["safemode"] = False
    assert "custom" in dj.config
    dj.config["custom"][
        "database.prefix"
    ] = f"u_{dj.config['database.user']}_testsuite_"
    return


@pytest.fixture(autouse=True, scope="session")
def pipeline(dj_config):

    _pipeline = load_pipeline()

    if len(_pipeline["acquisition"].Experiment()):
        drop_schema()
        _pipeline = load_pipeline()

    yield _pipeline

    if _tear_down:

        drop_schema()


@pytest.fixture(scope="session")
def exp_creation(test_params, pipeline):
    from aeon.dj_pipeline.populate import create_experiment_02

    create_experiment_02.main()

    acquisition = pipeline["acquisition"]

    experiment_name = acquisition.Experiment.fetch1("experiment_name")

    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "raw",
            "directory_path": test_params["raw_dir"],
        }
    )
    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "quality-control",
            "directory_path": test_params["qc_dir"],
        }
    )

    return


@pytest.fixture(scope="session")
def epoch_chunk_ingestion(test_params, pipeline, exp_creation):
    acquisition = pipeline["acquisition"]

    test_params["experiment_name"]

    acquisition.Epoch.ingest_epochs(
        experiment_name=test_params["experiment_name"],
        start=test_params["start_ts"],
        end=test_params["end_ts"],
    )

    acquisition.Chunk.ingest_chunks(experiment_name=test_params["experiment_name"])

    return


@pytest.fixture(scope="session")
def experimentlog_ingestion(pipeline, epoch_chunk_ingestion):
    acquisition = pipeline["acquisition"]

    acquisition.ExperimentLog.populate(**_populate_settings)
    acquisition.SubjectEnterExit.populate(**_populate_settings)
    acquisition.SubjectWeight.populate(**_populate_settings)

    return


@pytest.fixture(scope="session")
def camera_qc_ingestion(pipeline, epoch_chunk_ingestion):
    qc = pipeline["qc"]
    qc.CameraQC.populate()

    return


@pytest.fixture(scope="session")
def camera_tracking_ingestion(pipeline, camera_qc_ingestion):
    tracking = pipeline["tracking"]
    tracking.CameraTracking.populate(display_progress=True)

    return
