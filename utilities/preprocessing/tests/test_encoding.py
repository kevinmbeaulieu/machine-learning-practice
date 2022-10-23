import pandas as pd
import unittest

from utilities.preprocessing import encoding

class TestEncoding(unittest.TestCase):
    def test_one_hot_encode(self):
        df = pd.DataFrame({
            'col': ['a', 'b', 'c', 'a', 'b', 'c']
        })
        got = encoding.one_hot_encode(df, 'col')
        expected = pd.DataFrame({
            'col_b': [0, 1, 0, 0, 1, 0],
            'col_c': [0, 0, 1, 0, 0, 1],
        }, dtype='uint8')
        pd.testing.assert_frame_equal(expected, got)

    def test_encode_ordinal(self):
        df = pd.DataFrame({
            'col': ['a', 'b', 'c', 'a', 'b', 'c']
        })
        got = encoding.encode_ordinal(df, 'col', ['a', 'b', 'c'])
        expected = pd.DataFrame({
            'col': [1, 2, 3, 1, 2, 3]
        })
        pd.testing.assert_frame_equal(expected, got)
