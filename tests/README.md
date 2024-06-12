# Pytest for Project Aeon

The following pytest routine will test whether the pre-defined datajoint schemas are properly instantiated and the sample data ingested in key datajoint tables.  

For running the test, make sure you are in ```aeon``` virtual environment.
```
module load miniconda
conda activate aeon
```

Currently, the test will use the following sample dataset as defined in ```test_params``` from ```tests/conftest.py```
```python
        "start_ts": "2022-06-22 08:51:10",
        "end_ts": "2022-06-22 14:00:00",
        "experiment_name": "exp0.2-r0",
        "raw_dir": "aeon/data/raw/AEON2/experiment0.2",
        "qc_dir": "aeon/data/qc/AEON2/experiment0.2",
        "test_dir": data_dir(),
        "subject_count": 5,
        "epoch_count": 1,
        "chunk_count": 7,
        "experiment_log_message_count": 0,
        "subject_enter_exit_count": 0,
        "subject_weight_time_count": 0,
        "camera_qc_count": 40,
        "camera_tracking_object_count": 5,
```

Set `_tear_down=True` in `conftest.py` for proper cleanup of test artifacts after each testing (except for debugging/development purposes).

The test can then be run with the following simple command at the root directory of the repo:
```
pytest 
```

With no command line arguments being specified, the command will run on any modules or functions that start with ```test_```)

You can add in a series of command line arguments to get a more detailed view of the test results like the following:
```
pytest --pdb -sv --cov-report term-missing --cov=aeon_mecha -p no:warnings
```

The test can also be run on a single pytest module,

```
pytest -k <module_name>
```
or a function.

```
pytest -k <function_name>
```

You can also run tests on a specific marker:

```
pytest -m <marker_name>
```

For more detailed guides, please refer to [pytest documentation](https://docs.pytest.org/en/7.1.x/contents.html).