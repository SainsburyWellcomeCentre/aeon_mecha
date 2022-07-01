import datajoint as dj
import pytest
import pathlib


_tear_down = False
_populate_settings = {"suppress_errors": True}


@pytest.fixture(autouse=True)
def dj_config():
    """If dj_local_config exists, load"""
    dj_config_fp = pathlib.Path("../../dj_local_conf.json")
    assert dj_config_fp.exists()
    dj.config.load(dj_config_fp)
    dj.config["safemode"] = False
    assert "custom" in dj.config
    dj.config["custom"]["database.prefix"] = "aeon_test_"
    return


@pytest.fixture
def pipeline():
    from aeon.dj_pipeline import (
        lab,
        subject,
        acquisition,
        qc,
        tracking,
        analysis,
        report,
    )

    yield {
        "subject": subject,
        "lab": lab,
        "acquisition": acquisition,
        "qc": qc,
        "tracking": tracking,
        "analysis": analysis,
        "report": report,
    }

    if _tear_down:
        subject.Subject.delete()


@pytest.fixture
def test_variables():
    return {
        "experiment_name": "exp0.2-r0",
        "raw_dir": "aeon/data/raw/TEST_SUITE/experiment0.2",
        "qc_dir": "aeon/data/qc/TEST_SUITE/experiment0.2",
        "subject_count": 5,
        "epoch_count": 999,
        "chunk_count": 999,
    }


@pytest.fixture
def new_experiment(pipeline, test_variables):
    from aeon.dj_pipeline.populate import create_experiment_02

    acquisition = pipeline["acquisition"]

    create_experiment_02.main()

    experiment_name = acquisition.Experiment.fetch1("experiment_name")

    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "raw",
            "directory_path": test_variables["raw_dir"],
        }
    )
    acquisition.Experiment.Directory.update1(
        {
            "experiment_name": experiment_name,
            "repository_name": "ceph_aeon",
            "directory_type": "quality-control",
            "directory_path": test_variables["qc_dir"],
        }
    )

    yield

    if _tear_down:
        acquisition.Experiment.delete()


@pytest.fixture
def epoch_chunk_ingestion(pipeline, test_variables):
    acquisition = pipeline["acquisition"]

    acquisition.Epoch.ingest_epochs(experiment_name=test_variables["experiment_name"])
    acquisition.Chunk.ingest_chunks(experiment_name=test_variables["experiment_name"])

    yield

    if _tear_down:
        acquisition.Epoch.delete()


@pytest.fixture
def experimentlog_ingestion(pipeline, test_variables):
    acquisition = pipeline["acquisition"]

    acquisition.ExperimentLog.populate(**_populate_settings)
    acquisition.SubjectEnterExit.populate(**_populate_settings)
    acquisition.SubjectWeight.populate(**_populate_settings)

    yield

    if _tear_down:
        acquisition.ExperimentLog.delete()
        acquisition.SubjectEnterExit.delete()
        acquisition.SubjectWeight.delete()
