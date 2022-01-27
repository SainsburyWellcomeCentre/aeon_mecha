'''Unit Tests for aeon.preprocessing.api'''

import pytest
import aeon.preprocessing.api as aeon_api

from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Path to exp 0.1 data
exp01_data_root = Path('/ceph/aeon/test2/experiment0.1/')
#def test_data_root():
#    assert Path.is_dir(exp01_data_root)

# <s Test `aeon`
test_data = [
    (1, '01/01/1904, 00:00:01'),
    (3600, '01/01/1904, 01:00:00')
]
@pytest.mark.parametrize("seconds, expected", test_data)
def test_aeon(seconds, expected):
    assert aeon_api.aeon(seconds).strftime("%m/%d/%Y, %H:%M:%S") == expected
# /s>

# <s Test `chunk`
test_datetime1 = datetime(2022, 1, 24, 9, 59, 59)
exp1 = '2022-01-24 09:00:00'
test_datetime2 = datetime(2022, 1, 24, 10, 0, 0)
exp2 = '2022-01-24 10:00:00'
test_pd_datetime = \
    pd.date_range("2022-01-26", periods=2, freq='h')[1]
exp3 = '2022-01-26 01:00:00'
test_data = [
    (test_datetime1, exp1),
    (test_datetime2, exp2),
    (test_pd_datetime, exp3)
]
@pytest.mark.parametrize("datetime, expected", test_data)
def test_chunk(datetime, expected):
    assert str(aeon_api.chunk(datetime)) == expected
# /s>

# <s Test `chunk_key`
test_file1 = Path.joinpath(exp01_data_root,
    '2021-10-27T10-28-19/SessionData/SessionData_2021-10-27T09-00-00.csv')
exp1 = datetime(2021, 10, 27, 9, 0)
test_file2 = Path.joinpath(exp01_data_root,
    '2021-12-14T08-49-29/Patch1/Patch1_State_2021-12-14T17-00-00.csv')
exp2 = datetime(2021, 12, 14, 17, 0)
test_data = [
    (test_file1, exp1),
    (test_file2, exp2)
]
@pytest.mark.parametrize("file, expected", test_data)
def test_chunk_key(file, expected):
    assert Path.is_file(file)
    assert aeon_api.chunk_key(file) == expected
# /s>

# <s Test `chunk_range`

# /s>

# Test `chunk_filter`

# Test `chunk_glob`

# Test `chunkreader`

# Test `chunkdata`

# Test `harpreader`
