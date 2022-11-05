from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import pandas as pd
import unittest

from utilities.model import NullModel, KNNClassifierModel, KNNRegressionModel, EditedKNNClassifierModel, EditedKNNRegressionModel, CondensedKNNClassifierModel, CondensedKNNRegressionModel
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

class TestKNNClassifierModel(unittest.TestCase):
    def test_predict(self):
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

        model = KNNClassifierModel(k=3)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # use sklearn to get expected values
        sklearn_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute', p=2)
        sklearn_model.fit(df_train[['size', 'shape']], df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test))

        pd.testing.assert_series_equal(expected, got)

class TestKNNRegressionModel(unittest.TestCase):
   def test_predict(self):
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

       model = KNNRegressionModel(k=3)
       model.train(df_train, dataset)
       got = model.predict(df_test).reset_index(drop=True)

       # Use sklearn to get the expected values
       sklearn_model = KNeighborsRegressor(n_neighbors=3, algorithm='brute', p=2)
       sklearn_model.fit(df_train[['size', 'shape']], df_train['output'])
       expected = pd.Series(sklearn_model.predict(df_test))

       pd.testing.assert_series_equal(expected, got)

class TestEditedKNNClassifierModel(unittest.TestCase):
    def test_predict(self):
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

        model = EditedKNNClassifierModel(k=3, df_val=df_val)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # use sklearn to get expected values
        sklearn_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute', p=2)
        sklearn_model.fit(df_train[['size', 'shape']], df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test))

        pd.testing.assert_series_equal(expected, got)

class TestEditedKNNRegressionModel(unittest.TestCase):
    def test_predict(self):
         df_train = pd.DataFrame({
              'size': [1, 4, 2, 4, 6, 8],
              'shape': [2, 2, 7, 6, 3, 3],
              'output': [1, 3, 4, 6, 7, 9],
         })
         df_val = pd.DataFrame({
              'size': [2, 3, 6],
              'shape': [3, 6, 5],
              'output': [2, 5, 8],
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

         model = EditedKNNRegressionModel(k=3, ε=0.1, df_val=df_val)
         model.train(df_train, dataset)
         got = model.predict(df_test).reset_index(drop=True)

         # Use sklearn to get the expected values
         sklearn_model = KNeighborsRegressor(n_neighbors=3, algorithm='brute', p=2)
         sklearn_model.fit(df_train[['size', 'shape']], df_train['output'])
         expected = pd.Series(sklearn_model.predict(df_test))

         pd.testing.assert_series_equal(expected, got)

class TestCondensedKNNClassifierModel(unittest.TestCase):
    def test_predict(self):
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

        model = CondensedKNNClassifierModel(k=3)
        model.train(df_train, dataset)
        got = model.predict(df_test).reset_index(drop=True)

        # use sklearn to get expected values
        sklearn_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute', p=2)
        sklearn_model.fit(df_train[['size', 'shape']], df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test))

        pd.testing.assert_series_equal(expected, got)

class TestCondensedKNNRegressionModel(unittest.TestCase):
    def test_predict(self):
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

         model = CondensedKNNRegressionModel(k=3, ε=0.1)
         model.train(df_train, dataset)
         got = model.predict(df_test).reset_index(drop=True)

         # Use sklearn to get the expected values
         sklearn_model = KNeighborsRegressor(n_neighbors=3, algorithm='brute', p=2)
         sklearn_model.fit(df_train[['size', 'shape']], df_train['output'])
         expected = pd.Series(sklearn_model.predict(df_test))

         pd.testing.assert_series_equal(expected, got)
