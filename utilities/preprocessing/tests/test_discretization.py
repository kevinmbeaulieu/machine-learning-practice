import pandas as pd
import unittest

from utilities.preprocessing.discretization import discretize

class TestDiscretization(unittest.TestCase):
    def test_discretize_equal_width(self):
        df = pd.DataFrame({
            'col': [1, 2, 3, 80, 81, 82, 98, 99, 100]
        })
        got = discretize(df, 'col', 'equal-width', 3)['col'].to_list()
        expected = [1, 1, 1, 3, 3, 3, 3, 3, 3]
        self.assertListEqual(expected, got)

    def test_discretize_equal_freq(self):
        df = pd.DataFrame({
            'col': [1, 2, 3, 80, 81, 82, 98, 99, 100]
        })
        got = discretize(df, 'col', 'equal-frequency', 3)['col'].to_list()
        expected = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.assertListEqual(expected, got)