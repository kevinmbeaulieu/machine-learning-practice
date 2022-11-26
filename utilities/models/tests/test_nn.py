from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import unittest

from utilities import crossvalidation
from utilities.models.nn import NeuralNetworkModel, InputLayer, DenseLayer, DropoutLayer
from utilities.preprocessing import featurescaling, encoding
from utilities.preprocessing.dataset import Dataset

class TestNeuralNetwork(unittest.TestCase):
    def test_predict_classification_iris(self):
        dataset = Dataset(
            name='iris',
            task='classification',
            file_path='utilities/models/tests/fixtures/Iris.csv',
            col_names=['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
            header=0,
            ignore_cols=['id'],
            standardize_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            metrics=['acc'],
        )
        self._verify_classification_dataset(dataset)

    def test_predict_classification_fruit(self):
        dataset = Dataset(
            name='fruit',
            task='classification',
            file_path='utilities/models/tests/fixtures/fruit_data_with_colours.csv',
            col_names=['fruit_label', 'class', 'fruit_subtype', 'mass', 'width', 'height', 'color_score'],
            header=0,
            ignore_cols=['fruit_label', 'fruit_subtype'],
            standardize_cols=['mass', 'width', 'height', 'color_score'],
            metrics=['acc'],
        )
        self._verify_classification_dataset(dataset)

    def _verify_classification_dataset(self, dataset: Dataset):
        df = dataset.load_data()
        df_train, df_test = crossvalidation.split(df, frac=[0.8, 0.2], stratify_by='class')
        df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)
        X_test = df_test.drop('class', axis=1)
        y_test = df_test['class']

        model = NeuralNetworkModel(
            batch_size=df_train.shape[0],
            learning_rate=0.01,
            num_epochs=5000,
            verbose=True
        )
        model.layers = [
            InputLayer(X_test.shape[1]),
            DenseLayer(500, activation='relu'),
            DenseLayer(100, activation='relu'),
            DenseLayer(df_train['class'].unique().shape[0], activation='softmax'),
        ]
        model.train(df_train, dataset)
        got = model.predict(X_test)

        pd.testing.assert_series_equal(y_test, got, check_index=False)

    def test_predict_classification(self):
        dataset = Dataset(
            name='test',
            task='classification',
            file_path='fake.csv',
            col_names=['size', 'shape', 'class'],
            standardize_cols=['size', 'shape'],
            metrics=['acc'],
        )

        df_train = pd.DataFrame({
            'size': [1, 2, 2, 3, 6, 6],
            'shape': [2, 3, 7, 6, 3, 5],
            'class': ['red', 'red', 'green', 'green', 'blue', 'blue'],
        })
        df_test = pd.DataFrame({
            'size': [1, 2, 4, 5, 7, 8],
            'shape': [7, 1, 1, 7, 2, 5],
            'class': ['green', 'red', 'red', 'green', 'blue', 'blue'],
        })
        df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)
        X_test = df_test.drop('class', axis=1)
        y_test = df_test['class']
        y_test.name = None

        model = NeuralNetworkModel(batch_size=6, learning_rate=0.001, num_epochs=500, verbose=True)
        model.layers = [
            InputLayer(2),
            DenseLayer(100, activation='relu'),
            DenseLayer(50, activation='relu'),
            DenseLayer(3, activation='softmax'),
        ]
        model.train(df_train, dataset)
        got = model.predict(X_test)

        pd.testing.assert_series_equal(y_test, got, check_index=False)

    def test_predict_regression_fish(self):
        dataset = Dataset(
            name='fish',
            task='regression',
            file_path='utilities/models/tests/fixtures/Fish.csv',
            col_names=['species', 'output', 'length1', 'length2', 'length3', 'height', 'width'],
            header=0,
            standardize_cols=['length1', 'length2', 'length3', 'height', 'width'],
            nominal_cols=['species'],
            metrics=['mse'],
        )
        self._verify_regression_dataset(dataset)

    def _verify_regression_dataset(self, dataset: Dataset):
        df = dataset.load_data()
        df = encoding.encode_categorical_data(df, dataset)
        df_train, df_test = crossvalidation.split(df, frac=[0.8, 0.2])
        df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)
        X_test = df_test.drop('output', axis=1)
        y_test = df_test['output']

        model = NeuralNetworkModel(
            batch_size=df_train.shape[0],
            learning_rate=0.1,
            num_epochs=5000,
            verbose=True
        )
        model.layers = [
            InputLayer(X_test.shape[1]),
            DenseLayer(500, activation='relu'),
            DenseLayer(100, activation='relu'),
            DenseLayer(1, activation='linear'),
        ]
        model.train(df_train, dataset)
        got = model.predict(X_test)

        pd.testing.assert_series_equal(y_test, got, atol=0.01, check_index=False)


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
            metrics=['mse'],
        )

        model = NeuralNetworkModel(batch_size=6, learning_rate=0.001, num_epochs=500, verbose=True)
        model.layers = [
            InputLayer(2),
            DenseLayer(100, activation='relu'),
            DenseLayer(50, activation='relu'),
            DenseLayer(1, activation='linear'),
        ]
        model.train(df_train, dataset)
        got = model.predict(df_test)

        expected = pd.Series([5.0, 2.0, 2.0, 5.0, 8.0, 8.0])
        pd.testing.assert_series_equal(expected, got, atol=0.001)

