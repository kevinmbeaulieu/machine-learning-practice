import pandas as pd
import unittest

from utilities.preprocessing.dataset import Dataset
from utilities.preprocessing.imputation import impute_missing_values

class TestImputation(unittest.TestCase):
    def test_impute_missing_values(self):
        df = pd.DataFrame({
            'col_a': ['blue', '?', '?', 'blue', 'red', 'green'],
            'col_b': ['first', 'second', '?', '?', 'second', 'third'],
            'col_c': [1, 2, '?', 4, '?', 6],
        })
        dataset = Dataset(
            name='test',
            task='regression',
            file_path='fake.csv',
            col_names=['col_a', 'col_b', 'col_c'],
            nominal_cols=['col_a'],
            ordinal_cols={'col_b': ['first', 'second', 'third']},
            missing_value_symbol='?',
        )
        got = impute_missing_values(df, dataset)
        expected = pd.DataFrame({
            'col_a': ['blue', 'blue', 'blue', 'blue', 'red', 'green'],
            'col_b': ['first', 'second', 'second', 'second', 'second', 'third'],
            'col_c': [1, 2, 3.25, 4, 3.25, 6],
        })
        pd.testing.assert_frame_equal(expected, got)