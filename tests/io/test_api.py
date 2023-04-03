import unittest
import aeon.io.api as aeon
from aeon.schema.dataset import exp02
import pandas as pd

class TestLoad(unittest.TestCase):
    def test_load_monotonic(self):
        data = aeon.load('./tests/data/monotonic', exp02.Patch2.Encoder)
        self.assertTrue(data.index.is_monotonic_increasing)

    def test_load_nonmonotonic(self):
        data = aeon.load('./tests/data/nonmonotonic', exp02.Patch2.Encoder)
        self.assertFalse(data.index.is_monotonic_increasing)
        
if __name__ == '__main__':
    unittest.main()