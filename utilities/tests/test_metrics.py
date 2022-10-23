import numpy as np
import pandas as pd
import unittest

from utilities.metrics import compute_metrics

class TestMetrics(unittest.TestCase):
    def verify_metric(self, actual: np.ndarray, predicted: np.ndarray, metric: str):
        got = compute_metrics(actual, predicted, [metric], positive_class=1, negative_class=0)[0]
        expected = compute_metrics(actual, predicted, [metric], positive_class=1, negative_class=0, use_sklearn=True)[0]
        self.assertEqual(expected, got)

class TestRegressionMetrics(TestMetrics):
    def setUp(self):
        self.df = pd.read_csv('utilities/tests/fixtures/housing.csv')

    def test_compute_mse(self):
        actual_prices = self.df[['SalePrice']].to_numpy()
        predicted_prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10,1)

        self.verify_metric(actual_prices, predicted_prices, 'mse')

    def test_compute_mae(self):
        actual_prices = self.df[['SalePrice']].to_numpy()
        predicted_prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10,1)

        self.verify_metric(actual_prices, predicted_prices, 'mae')

    def test_compute_r2(self):
        actual_prices = self.df[['SalePrice']].to_numpy()
        predicted_prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10,1)

        self.verify_metric(actual_prices, predicted_prices, 'r2')

    def test_compute_pearson(self):
        actual_prices = self.df[['SalePrice']].to_numpy()
        predicted_prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(10,1)

        self.verify_metric(actual_prices, predicted_prices, 'pearson')

class TestClassificationMetrics(TestMetrics):
    def setUp(self):
        self.df = pd.read_csv('utilities/tests/fixtures/iris.csv')

    def test_compute_accuracy(self):
        actual_species = self.df[['IsSetosa']].to_numpy()
        predicted_species = np.array([1, 0, 1, 0, 0]).reshape(5,1)

        self.verify_metric(actual_species, predicted_species, 'acc')

    def test_compute_precision(self):
        actual_species = self.df[['IsSetosa']].to_numpy()
        predicted_species = np.array([1, 0, 1, 0, 0]).reshape(5,1)

        self.verify_metric(actual_species, predicted_species, 'precision')

    def test_compute_recall(self):
        actual_species = self.df[['IsSetosa']].to_numpy()
        predicted_species = np.array([1, 0, 1, 0, 0]).reshape(5,1)

        self.verify_metric(actual_species, predicted_species, 'recall')

    def test_compute_f1(self):
        actual_species = self.df[['IsSetosa']].to_numpy()
        predicted_species = np.array([1, 0, 1, 0, 0]).reshape(5,1)

        self.verify_metric(actual_species, predicted_species, 'f1')

if __name__ == '__main__':
    unittest.main()