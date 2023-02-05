import numpy as np
import pandas as pd
import random
import unittest

from utilities.crossvalidation import split_for_cross_validation
from utilities.preprocessing.dataset import Dataset

class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)

        self.df = pd.DataFrame({
            'col_a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'col_b': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'col_c': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
            'class': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue', 'green', 'red'],
        })
        self.dataset = Dataset(
            name='test',
            task='classification',
            file_path='fake.csv',
            col_names=['col_a', 'col_b', 'col_c', 'class']
        )

    def test_k_split_for_cross_validation(self):
        got = split_for_cross_validation(self.df, 'k', 5, self.dataset)

        expected = [
            pd.DataFrame({
                'col_a': [1, 9],
                'col_b': [101, 109],
                'col_c': [1001, 1009],
                'class': ['red', 'green'],
            }),
            pd.DataFrame({
                'col_a': [10, 4],
                'col_b': [110, 104],
                'col_c': [1010, 1004],
                'class': ['red', 'red'],
            }),
            pd.DataFrame({
                'col_a': [6],
                'col_b': [106],
                'col_c': [1006],
                'class': ['green'],
            }),
            pd.DataFrame({
                'col_a': [7],
                'col_b': [107],
                'col_c': [1007],
                'class': ['red'],
            }),
            pd.DataFrame({
                'col_a': [8, 3],
                'col_b': [108, 103],
                'col_c': [1008, 1003],
                'class': ['blue', 'green'],
            }),
            pd.DataFrame({
                'col_a': [5, 2],
                'col_b': [105, 102],
                'col_c': [1005, 1002],
                'class': ['blue', 'blue'],
            }),
        ]

        self.verify_dataframe_list_equal(expected, got)

    def test_kx2_split_for_cross_validation(self):
        got = split_for_cross_validation(self.df, 'kx2', 3, self.dataset)

        expected = [
            pd.DataFrame({
                'col_a': [1, 9],
                'col_b': [101, 109],
                'col_c': [1001, 1009],
                'class': ['red', 'green'],
            }),
            pd.DataFrame({
                'col_a': [10, 4, 2, 3, 6],
                'col_b': [110, 104, 102, 103, 106],
                'col_c': [1010, 1004, 1002, 1003, 1006],
                'class': ['red', 'red', 'blue', 'green', 'green'],
            }),
            pd.DataFrame({
                'col_a': [7, 5, 8],
                'col_b': [107, 105, 108],
                'col_c': [1007, 1005, 1008],
                'class': ['red', 'blue', 'blue'],
            }),
            pd.DataFrame({
                'col_a': [10, 7, 8, 2, 3, 6],
                'col_b': [110, 107, 108, 102, 103, 106],
                'col_c': [1010, 1007, 1008, 1002, 1003, 1006],
                'class': ['red', 'red', 'blue', 'blue', 'green', 'green'],
            }),
            pd.DataFrame({
                'col_a': [4, 5],
                'col_b': [104, 105],
                'col_c': [1004, 1005],
                'class': ['red', 'blue'],
            }),
            pd.DataFrame({
                'col_a': [7, 10, 3],
                'col_b': [107, 110, 103],
                'col_c': [1007, 1010, 1003],
                'class': ['red', 'red', 'green'],
            }),
            pd.DataFrame({
                'col_a': [4, 8, 5, 2, 6],
                'col_b': [104, 108, 105, 102, 106],
                'col_c': [1004, 1008, 1005, 1002, 1006],
                'class': ['red', 'blue', 'blue', 'blue', 'green'],
            }),
        ]

        self.verify_dataframe_list_equal(expected, got)

    def verify_dataframe_list_equal(self, left: list[pd.DataFrame], right: list[pd.DataFrame]):
        left = list(map(lambda df: df.reset_index(drop=True), left))
        right = list(map(lambda df: df.reset_index(drop=True), right))

        self.assertEqual(len(left), len(right))

        for i in range(len(left)):
            pd.testing.assert_frame_equal(left[i], right[i])

