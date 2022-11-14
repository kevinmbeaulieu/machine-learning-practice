from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import unittest

from utilities.models.tree import DecisionTreeModel, RandomForestModel
from utilities.preprocessing.dataset import Dataset

class TestDecisionTree(unittest.TestCase):
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

        model = DecisionTreeModel(pruning_strategy=None)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series(['green', 'red', 'red', 'green', 'blue', 'blue'])

        pd.testing.assert_series_equal(expected, got)

    def test_predict_classification_pre_prune(self):
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

        model = DecisionTreeModel(pruning_strategy='pre-prune')
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series(['green', 'red', 'red', 'green', 'blue', 'blue'])

        pd.testing.assert_series_equal(expected, got)

    def test_predict_classification_post_prune(self):
        df_train = pd.DataFrame({
            'size': [1, 4, 2, 4, 6, 8],
            'shape': [2, 2, 7, 6, 3, 3],
            'class': ['red', 'red', 'green', 'green', 'blue', 'blue'],
        })
        df_val = pd.DataFrame({
            'size': [2, 3, 6],
            'shape': [3, 6, 5],
            'class': ['red', 'green', 'blue'],
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

        model = DecisionTreeModel(pruning_strategy='post-prune', post_pruning_set=df_val)
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

        model = DecisionTreeModel(
            pruning_strategy=None,
            leaf_size = 0.34,
        )
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([5.0, 2.0, 2.0, 5.0, 8.0, 8.0])
        pd.testing.assert_series_equal(expected, got, atol=0.001)

class TestRandomForestModel(unittest.TestCase):
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

        np.random.seed(1234)

        model = RandomForestModel(num_trees=10, pruning_strategy=None)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # Third test example has equal votes for red and green, so either prediction is valid.
        expected1 = pd.Series(['green', 'red', 'red', 'green', 'blue', 'blue'])
        expected2 = pd.Series(['green', 'red', 'green', 'green', 'blue', 'blue'])
        try:
            pd.testing.assert_series_equal(expected1, got)
        except AssertionError:
            pd.testing.assert_series_equal(expected2, got)

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

        np.random.seed(1234)

        model = RandomForestModel(num_trees=10, pruning_strategy=None, leaf_size = 0.34)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([2.6, 2.5, 4.1, 5.45, 6.3, 6.85])
        pd.testing.assert_series_equal(expected, got, atol=0.001)
