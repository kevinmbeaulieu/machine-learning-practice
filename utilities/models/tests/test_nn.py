import pandas as pd
import unittest

from utilities.models.nn import NeuralNetworkModel, InputLayer, DenseLayer
from utilities.preprocessing.dataset import Dataset

class TestNeuralNetwork(unittest.TestCase):
    def test_predict_classification(self):
        df_train = pd.DataFrame({
            'size': [1, 2, 4, 2, 3, 4, 6, 6, 8],
            'shape': [2, 3, 2, 7, 6, 6, 3, 5, 3],
            'class': ['red', 'red', 'red', 'green', 'green', 'green', 'blue', 'blue', 'blue'],
        })
        df_test = pd.DataFrame({
            'size': [1, 2, 4, 5, 7, 8],
            'shape': [7, 1, 1, 7, 2, 5],
        })
        dataset = Dataset(
            name='test',
            task='classification',
            file_path='fake.csv',
            col_names=['size', 'shape', 'class'],
        )

        model = NeuralNetworkModel(num_processes=1)
        model.layers = [
            InputLayer(2),
            DenseLayer(3, activation='softmax'),
        ]
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series(['green', 'red', 'red', 'green', 'blue', 'blue'])

        pd.testing.assert_series_equal(expected, got)

    def test_predict_regression(self):
        df_train = pd.DataFrame({
            'size': [1, 2, 4, 2, 3, 4, 6, 6, 8],
            'shape': [2, 3, 2, 7, 6, 6, 3, 5, 3],
            'output': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        })
        df_test = pd.DataFrame({
            'size': [1, 2, 4, 5, 7, 8],
            'shape': [7, 1, 1, 7, 2, 5],
        })
        dataset = Dataset(
            name='test',
            task='regression',
            file_path='fake.csv',
            col_names=['size', 'shape', 'output'],
        )

        model = NeuralNetworkModel()
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([5.0, 2.0, 2.0, 5.0, 8.0, 8.0])
        pd.testing.assert_series_equal(expected, got, atol=0.001)

