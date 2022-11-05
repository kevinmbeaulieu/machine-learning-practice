import math
import numpy as np
import statistics
import pandas as pd

from utilities.metrics import compute_metrics
from utilities.preprocessing.dataset import Dataset

class Model:
    """
    Abstract class defining a common interface for machine learning models.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        """
        Must be overridden by subclass to train the model.

        :param df: pd.DataFrame, Training set
        :param dataset: Dataset, Metadata about the dataset being used for training
        """
        raise Exception("Model subclass expected to override train function, but didn't")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Must be overridden in subclass to predict class label/regression output for the test set.

        :param df: pd.DataFrame, Test set for which to predict values

        :return pd.Series
            For classification tasks, predicted class labels for the test set
            For regression tasks, predicted output values for the test set
        """
        raise Exception("Model subclass expected to override predict function, but didn't")

class NullModel(Model):
    """
    Null Model for testing machine learning pipeline.

    For classification tasks, uses the training set's plurality (most common) class label as prediction.
    For regression tasks, uses the average value of the training set's output attribute as prediction.
    """

    def __init__(self):
        self.predict_value = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if dataset.task == 'classification':
            self.predict_value = df['class'].mode()[0]
        elif dataset.task == 'regression':
            self.predict_value = df['output'].mean()
        else:
            raise Exception("Failed to train null model for dataset with unrecognized task {}".format(dataset.task))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.predict_value is None:
            raise Exception("Failed to predict values with null model before model was trained")

        return pd.Series([self.predict_value]).repeat(df.shape[0])


class _KNNModel(Model):
    """
    Abstract parent class for K-Nearest Neighbors models.
    """
    def __init__(self, k: int):
        """
        :param k: int, Number of nearest neighbors to use for prediction
        """

        self.k = k
        self.df_train = None
        self.dataset = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        self.df_train = df
        self.dataset = dataset

    def _distance(self, df_left: pd.DataFrame, left: pd.Series, df_right: pd.DataFrame, right: pd.Series) -> float:
        """
        Calculates the distance between two data points.

        :param df_left: pd.DataFrame, Data frame containing the left data point
        :param left: pd.Series, Data point
        :param df_right: pd.DataFrame, Data frame containing the right data point
        :param right: pd.Series, Data point

        :return float, Distance between the two data points
        """

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        left = left.drop(output_col, errors='ignore')
        right = right.drop(output_col, errors='ignore')

        if left.shape != right.shape:
            raise Exception("Failed to calculate distance between vectors with different shapes {} != {}.\n\n{}\n\n{}".format(left.shape, right.shape, left, right))

        distance = 0
        for col in df_left.columns:
            if col == output_col:
                continue
            if col in self.dataset.nominal_cols:
                distance += self._vdm_distance_sq(df_left, left[col], df_right, right[col])
            else:
                distance += self._euclidean_distance_sq(left[col], right[col])
        return distance ** 0.5

    def _euclidean_distance_sq(self, left: float, right: float) -> float:
        return (left - right) ** 2

    def _vdm_distance_sq(self, df_left: pd.DataFrame, left: any, df_right: pd.DataFrame, right: any) -> float:
        """
        Calculates the square of the VDM distance between two nominal values.

        :param df_left: pd.DataFrame, Data frame containing the left value
        :param left: any, Left value
        :param df_right: pd.DataFrame, Data frame containing the right value
        :param right: any, Right value

        :return float, squared VDM distance between the two nominal values
        """
        distance = 0
        vdm_exp = 2

        # Number of rows in df_left where value of k'th attribute is left
        C_left = df_left.loc[df_left.iloc[:, self.k] == left].shape[0]
        # Number of rows in df_right where value of k'th attribute is right
        C_right = df_right.loc[df_right.iloc[:, self.k] == right].shape[0]

        for c in self.classes:
            # Number of rows in df_left where value of k'th attribute is left and class label is c
            C_left_a = df_left.loc[(df_left['class'] == c) & (df_left.iloc[:, self.k] == left)].shape[0]
            # Number of rows in df_right where value of k'th attribute is right and class label is c
            C_right_a = df_right.loc[(df_right['class'] == c) & (df_right.iloc[:, self.k] == right)].shape[0]

            distance += abs(C_left_a/C_left - C_right_a/C_right) ** vdm_exp

        return distance


class KNNClassifierModel(_KNNModel):
    """
    K-Nearest Neighbors Classifier Model.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if dataset.task != 'classification':
            raise Exception("Failed to predict values with KNN classifier model for dataset with unrecognized task {}".format(dataset.task))

        super().train(df, dataset)

        self.classes = self.df_train['class'].unique()

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.df_train is None or self.dataset is None:
            raise Exception("Failed to predict values with KNN classifier model before model was trained")

        y_pred = pd.Series(dtype='float64', index=range(df.shape[0]))

        for row in range(df.shape[0]):
            distances: list[tuple[float, str]] = []
            for train_row in range(self.df_train.shape[0]):
                x = df.iloc[row]
                x_train = self.df_train.iloc[train_row]
                distance = self._distance(df, x, self.df_train, x_train)
                distances.append((distance, x_train['class']))
            if len(distances) < self.k:
                raise Exception("Only found {} < {} neighbors for row {}. Training set had shape {}".format(len(distances), self.k, row, self.df_train.shape))
            y_pred.iloc[row] = self._get_knn_class(distances)

        return y_pred

    def _get_knn_class(self, distances: list[tuple[float, str]]) -> str:
        """
        Finds the most common class label among the k nearest neighbors.

        :param distances: list[tuple[float, str]], List of (distance, class label) tuples

        :return str, Most common class label among the k nearest neighbors
        """
        distances.sort(key=lambda x: x[0])
        classes = [distances[i][1] for i in range(self.k)]
        return max(classes, key=classes.count)


class KNNRegressionModel(_KNNModel):
    """
    K-Nearest Neighbors Regression Model.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if dataset.task != 'regression':
            raise Exception("Failed to predict values with KNN regression model for dataset with unrecognized task {}".format(dataset.task))

        super().train(df, dataset)

        self.γ = 1 / self.df_train['output'].std()

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.df_train is None or self.dataset is None:
            raise Exception("Failed to predict values with KNN regression model before model was trained")

        return pd.Series([self._get_knn_output(df, row) for row in range(df.shape[0])])

    def _get_knn_output(self, df: pd.DataFrame, row: int) -> float:
        """
        Calculates the output for a single row in the test set using the KNN smoother, where the KNN smoother is defined as
        ghat(x) = sum_t (K(x, x_t) * y_t) / sum_t (K(x, x_t))
        where x_t = t'th training example, y_t = t'th training example's output, and K(x, x_t) = exp[-γ ||x - x_t||_2].
        """

        numerator = np.sum(
            [self._gaussian_kernel(df, row, train_row) * self.df_train.iloc[train_row]['output'] for train_row in range(self.df_train.shape[0])]
        )
        denominator = np.sum(
            [self._gaussian_kernel(df, row, train_row) for train_row in range(self.df_train.shape[0])]
        )
        return numerator / denominator

    def _gaussian_kernel(self, df: pd.DataFrame, row: int, row_train: int) -> float:
        """
        Compute K(x, x_q) = exp[-γ ||x - x_q||_2] for use in the kernel smoother.
        """
        l2_norm = self._distance(df, df.iloc[row], self.df_train, self.df_train.iloc[row_train])
        return math.exp(-self.γ * l2_norm)


class EditedKNNClassifierModel(KNNClassifierModel):
    """
    Edited K-Nearest Neighbors Classifier Model.
    """
    def __init__(self, k: int, df_val: pd.DataFrame, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param df_val: pd.DataFrame, Validation set
        :param max_iterations: int, Maximum number of iterations to run the edited KNN algorithm
        """
        super().__init__(k)
        self.df_val = df_val
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._edit_training_set()

    def _edit_training_set(self):
        """
        Edited KNN algorithm:
            1. Consider each data point individually.
            2. For each data point, use its single nearest neighbor to make a prediction.
            3. If the prediction is correct, mark the data point for deletion.
            4. Stop editing once performance on the validation set starts to degrade.
        """

        iterations = 0
        prev_val_accuracy = 0
        prev_df_train = self.df_train.copy()
        while iterations < self.max_iterations:
            validation_model = KNNClassifierModel(1)
            validation_model.train(self.df_train, self.dataset)

            # Remove all rows in the training set that are correctly classified by
            # their single nearest neighbor.
            self.df_train['prediction'] = validation_model.predict(self.df_train)
            incorrectly_classified = self.df_train.loc[
                self.df_train['prediction'] != self.df_train['class']
            ]
            if incorrectly_classified.shape[0] == 0:
                # If this iteration would remove all remaining rows of the training set,
                # revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            self.df_train = incorrectly_classified.drop('prediction', axis=1)

            # Stop editing if performance on the validation set starts to degrade.
            self.df_val['predicted'] = self.predict(self.df_val)
            val_accuracy = compute_metrics(
                actual=self.df_val['class'].to_numpy(),
                predicted=self.df_val['predicted'].to_numpy(),
                metrics=['acc'],
            )[0]
            if val_accuracy < prev_val_accuracy:
                # If accuracy has decreased, revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            iterations += 1

class EditedKNNRegressionModel(KNNRegressionModel):
    """
    Edited K-Nearest Neighbors Regression Model.
    """
    def __init__(self, k: int, df_val: pd.DataFrame, ϵ: float, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param df_val: pd.DataFrame, Validation set
        :param ϵ: float, Tolerance for determining if a data point is correctly predicted
        :param max_iterations: int, Maximum number of iterations to run the edited KNN algorithm
        """
        super().__init__(k)
        self.df_val = df_val
        self.ϵ = ϵ
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._edit_training_set()

    def _edit_training_set(self):
        """
        Edited KNN algorithm:
            1. Consider each data point individually.
            2. For each data point, use its single nearest neighbor to make a prediction.
            3. If the prediction is correct, mark the data point for deletion.
            4. Stop editing once performance on the validation set starts to degrade.
        """

        iterations = 0
        prev_val_mse = np.inf
        prev_df_train = self.df_train.copy()
        while iterations < self.max_iterations:
            validation_model = KNNRegressionModel(1)
            validation_model.train(self.df_train, self.dataset)

            # Remove all rows in the training set that are correctly classified by
            # their single nearest neighbor.
            self.df_train['prediction'] = validation_model.predict(self.df_train)
            incorrectly_classified = self.df_train.loc[
                np.abs(self.df_train['prediction'] - self.df_train['output']) > self.ϵ
            ]
            if incorrectly_classified.shape[0] == 0:
                # If this iteration would remove all remaining rows of the training set,
                # revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            self.df_train = incorrectly_classified.drop('prediction', axis=1)

            # Stop editing if performance on the validation set starts to degrade.
            self.df_val['predicted'] = self.predict(self.df_val)
            val_mse = compute_metrics(
                actual=self.df_val['output'].to_numpy(),
                predicted=self.df_val['predicted'].to_numpy(),
                metrics=['mse'],
            )[0]
            self.df_val.drop('predicted', axis=1, inplace=True)
            if val_mse > prev_val_mse:
                # If MSE has increased, revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            iterations += 1

class CondensedKNNClassifierModel(KNNClassifierModel):
    """
    Condensed K-Nearest Neighbors Classifier Model.
    """
    def __init__(self, k: int, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param max_iterations: int, Maximum number of iterations to run the condensed KNN algorithm
        """
        super().__init__(k)
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._condense_training_set()

    def _condense_training_set(self):
        """
        Condensed KNN algorithm:
            1. Add the first data point from the training set into the condensed set.
            2. Consider the remaining data points in the training set individually.
            3. For each data point, attempt to predict its value using the condensed set via 1-nn.
            4. If the prediction is incorrect, add the data point to the condensed set. Otherwise, move on to the next data point.
            5. Make multiple passes through the data in the training set that has not been added until the condensed set stops changing.
        """

        iterations = 0

        condensed_set = self.df_train.iloc[[0]]
        self.df_train.drop(0, inplace=True)
        while iterations < self.max_iterations:
            condensed_model = KNNClassifierModel(1)
            condensed_model.train(condensed_set, self.dataset)
            self.df_train['prediction'] = condensed_model.predict(self.df_train)
            incorrectly_classified = self.df_train.loc[
                self.df_train['prediction'] != self.df_train['class']
            ].drop('prediction', axis=1)
            prev_condensed_set_size = condensed_set.shape[0]
            condensed_set = pd.concat([condensed_set, incorrectly_classified])
            condensed_set.drop_duplicates(inplace=True)
            if condensed_set.shape[0] == prev_condensed_set_size:
                # If the condensed set has stopped changing, stop condensing.
                self.df_train = condensed_set
                return

            self.df_train = self.df_train.loc[
                self.df_train['prediction'] == self.df_train['class']
            ].drop('prediction', axis=1)
        self.df_train = condensed_set

class CondensedKNNRegressionModel(KNNRegressionModel):
    """
    Condensed K-Nearest Neighbors Regression Model.
    """
    def __init__(self, k: int, ϵ: float, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param ϵ: float, Tolerance for determining if a data point is correctly predicted
        :param max_iterations: int, Maximum number of iterations to run the condensed KNN algorithm
        """
        super().__init__(k)
        self.ϵ = ϵ
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._condense_training_set()

    def _condense_training_set(self):
        """
        Condensed KNN algorithm:
            1. Add the first data point from the training set into the condensed set.
            2. Consider the remaining data points in the training set individually.
            3. For each data point, attempt to predict its value using the condensed set via 1-nn.
            4. If the prediction is incorrect, add the data point to the condensed set. Otherwise, move on to the next data point.
            5. Make multiple passes through the data in the training set that has not been added until the condensed set stops changing.
        """

        iterations = 0

        condensed_set = self.df_train.iloc[[0]]
        self.df_train.drop(0, inplace=True)
        while iterations < self.max_iterations:
            condensed_model = KNNRegressionModel(1)
            condensed_model.train(condensed_set, self.dataset)
            self.df_train['prediction'] = condensed_model.predict(self.df_train)
            incorrectly_classified = self.df_train.loc[
                np.abs(self.df_train['prediction'] - self.df_train['output']) > self.ϵ
            ].drop('prediction', axis=1)
            prev_condensed_set_size = condensed_set.shape[0]
            condensed_set = pd.concat([condensed_set, incorrectly_classified])
            condensed_set.drop_duplicates(inplace=True)
            if condensed_set.shape[0] == prev_condensed_set_size:
                # If the condensed set has stopped changing, stop condensing.
                self.df_train = condensed_set
                return

            self.df_train = self.df_train.loc[
                np.abs(self.df_train['prediction'] - self.df_train['output']) <= self.ϵ
            ].drop('prediction', axis=1)
        self.df_train = condensed_set

