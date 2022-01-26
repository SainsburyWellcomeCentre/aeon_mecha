'''Unit Tests for aeon.preprocessing.api'''

import pytest
import aeon.preprocessing.api as aeon_api

from datetime import datetime, timedelta
import pandas as pd

# Path to exp 0.1 data
exp01_data_root = '/ceph/aeon/test2/experiment0.1/'

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
exp_datetime1 = '2022-01-24 09:00:00'
test_datetime2 = datetime(2022, 1, 24, 10, 0, 0)
exp_datetime2 = '2022-01-24 10:00:00'
# @todo: add test for pd.Series
test_data = [
    (test_datetime1, exp_datetime1),
    (test_datetime2, exp_datetime2)
]
@pytest.mark.parametrize("datetime, expected", test_data)
def test_chunk(datetime, expected):
    assert str(aeon_api.chunk(datetime)) == expected
# /s>

# Test `chunk_key`

# Test `chunk_range`

# Test `chunk_filter`

# Test `chunk_glob`

# Test `chunkreader`

# Test `chunkdata`

# Test `harpreader`
