from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import unittest

from utilities.model import NullModel, KNNModel, EditedKNNModel, CondensedKNNModel
from utilities.preprocessing.dataset import Dataset

class TestNullModel(unittest.TestCase):
    def test_predict_classification(self):
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

        model = NullModel()
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series(['red', 'red', 'red', 'red'])

        pd.testing.assert_series_equal(expected, got)

    def test_predict_regression(self):
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

        model = NullModel()
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([3.5, 3.5, 3.5, 3.5])

        pd.testing.assert_series_equal(expected, got)

class TestKNNModel(unittest.TestCase):
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

        model = KNNModel(k=3)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # use sklearn to get expected values
        sklearn_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute', p=2)
        sklearn_model.fit(df_train[['size', 'shape']], df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test))

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

        model = KNNModel(k=3)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        expected = pd.Series([4.7389, 1.8983, 3.7597, 6.2985, 6.9056, 8.0952])
        pd.testing.assert_series_equal(expected, got, atol=0.001)

class TestEditedKNNModel(unittest.TestCase):
    def test_edit_training_set(self):
        df_train = pd.DataFrame({
            'size': [1, 2, 2, 3, 3, 5, 2, 4, 4, 5, 5, 6],
            'shape': [6, 1, 2, 1, 4, 2, 7, 5, 8, 7, 8, 3],
            'class': ['red', 'red', 'red', 'red', 'red', 'red', 'green', 'green', 'green', 'green', 'green', 'green'],
        })
        df_val = pd.DataFrame({
            'size': [1, 4, 4, 7],
            'shape': [3, 1, 9, 7],
            'class': ['red', 'red', 'green', 'green'],
        })
        dataset = Dataset(
            name='test',
            task='classification',
            file_path='fake.csv',
            col_names=['size', 'shape', 'class'],
        )

        model = EditedKNNModel(k=3, df_val=df_val)
        model.train(df_train, dataset)
        got = model.df_train
        expected = pd.DataFrame({
            'size': [1, 3, 5, 2, 4, 6],
            'shape': [6, 4, 2, 7, 5, 3],
            'class': ['red', 'red', 'red', 'green', 'green', 'green'],
        })
        pd.testing.assert_frame_equal(expected, got)


class TestCondensedKNNModel(unittest.TestCase):
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

        model = CondensedKNNModel(k=3)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # use sklearn to get expected values
        sklearn_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute', p=2)
        sklearn_model.fit(df_train[['size', 'shape']], df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test))

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

         model = CondensedKNNModel(k=3, ε=0.1)
         model.train(df_train, dataset)
         got = model.predict(df_test).reset_index(drop=True)

         # Use sklearn to get the expected values
         sklearn_model = KNeighborsRegressor(n_neighbors=3, algorithm='brute', p=2)
         sklearn_model.fit(df_train[['size', 'shape']], df_train['output'])
         expected = pd.Series(sklearn_model.predict(df_test))

         pd.testing.assert_series_equal(expected, got)
