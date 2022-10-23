import pandas as pd
import unittest

from utilities.model import NullModel
from utilities.preprocessing.dataset import Dataset

class TestNullModel(unittest.TestCase):
    def test_predict_classification(self):
        model = NullModel()
        df_train = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'class': ['red', 'green', 'red', 'blue', 'red', 'green'],
        })
        df_test = pd.DataFrame({
            'id': [7, 8, 9, 10],
        })
        dataset = Dataset(
            name='test',
            task='classification',
            file_path='fake.csv',
            col_names=['id', 'class'],
        )
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series(['red', 'red', 'red', 'red'])

        pd.testing.assert_series_equal(expected, got)

    def test_predict_regression(self):
        model = NullModel()
        df_train = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'output': [1, 2, 3, 4, 5, 6],
        })
        df_test = pd.DataFrame({
            'id': [7, 8, 9, 10],
        })
        dataset = Dataset(
            name='test',
            task='regression',
            file_path='fake.csv',
            col_names=['id', 'output'],
        )
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([3.5, 3.5, 3.5, 3.5])

        pd.testing.assert_series_equal(expected, got)
