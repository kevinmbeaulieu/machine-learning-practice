from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import unittest

from utilities import crossvalidation
from utilities.models.nn import NeuralNetworkModel, InputLayer, DenseLayer, DropoutLayer
from utilities.preprocessing import featurescaling
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
        df = dataset.load_data()
        df_train, df_test = crossvalidation.split(df, frac=[0.8, 0.2], stratify_by='class')
        df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)
        print(df_train)
        print(df_test)
        X_test = df_test.drop('class', axis=1)
        y_test = df_test['class'].reset_index(drop=True)

        model = NeuralNetworkModel(batch_size=df_train.shape[0], learning_rate=0.001, num_epochs=1000)
        model.layers = [
            InputLayer(4),
            DenseLayer(100, activation='tanh'),
            DropoutLayer(0.8),
            DenseLayer(50, activation='tanh'),
            DenseLayer(3, activation='softmax'),
        ]
        model.train(df_train, dataset)
        got = model.predict(X_test).reset_index(drop=True)

        pd.testing.assert_series_equal(y_test, got)

    def test_predict_classification_fruit(self):
#         np.random.seed(1)

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
        df = dataset.load_data()
        df_train, df_test = crossvalidation.split(df, frac=[0.8, 0.2], stratify_by='class')
        df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)

        model = NeuralNetworkModel(batch_size=df_train.shape[0], learning_rate=0.001, num_epochs=100)
        model.layers = [
            InputLayer(4),
            DenseLayer(150, activation='sigmoid'),
            DropoutLayer(0.8),
            DenseLayer(150, activation='sigmoid'),
            DenseLayer(4, activation='softmax'),
        ]
        model.train(df_train, dataset)
        got = model.predict(df_test.drop('class', axis=1)).reset_index(drop=True)

#         expected = df_test['class'].reset_index(drop=True)
        sklearn_model = MLPClassifier(
            (100, 100),
#             activation='tanh',
#             solver='sgd',
#             alpha=0.001,
#             batch_size=df_train.shape[0],
            max_iter=1000,
#             random_state=1
        ).fit(df_train.drop('class', axis=1), df_train['class'])
        expected = pd.Series(sklearn_model.predict(df_test.drop('class', axis=1)))

#         pd.testing.assert_series_equal(df_test['class'].reset_index(drop=True), expected)
        pd.testing.assert_series_equal(expected, got)

    def test_predict_classification(self):
        df_train = pd.DataFrame({
            'size': [1, 2, 2, 3, 6, 6],
            'shape': [2, 3, 7, 6, 3, 5],
            'class': ['red', 'red', 'green', 'green', 'blue', 'blue'],
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

        np.random.seed(1)

        model = NeuralNetworkModel(batch_size=6, learning_rate=0.0001, num_epochs=500)
        model.layers = [
            InputLayer(2),
#             DenseLayer(3, activation='sigmoid'),
            DenseLayer(3, activation='sigmoid'),
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

