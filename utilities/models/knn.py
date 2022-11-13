import numpy as np
import pandas as pd

from .model import Model
from utilities.preprocessing.dataset import Dataset
from utilities import metrics

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
        left = left.copy().drop(output_col, errors='ignore')
        right = right.copy().drop(output_col, errors='ignore')

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

class KNNModel(_KNNModel):
    """
    K-Nearest Neighbors Model.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        if dataset.task == 'classification':
            self.classes = self.df_train['class'].unique()
        elif dataset.task == 'regression':
            self.γ = 1 / self.df_train['output'].std()
        else:
            raise Exception("Failed to train KNN model for dataset with unrecognized task {}".format(dataset.task))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.df_train is None:
            raise Exception("Failed to predict values with KNN model before model was trained")

        if self.dataset.task == 'classification':
            return df.apply(lambda row: self._predict_class(df, row), axis=1)
        elif self.dataset.task == 'regression':
            return df.apply(lambda row: self._predict_output(df, row), axis=1)
        else:
            raise Exception("Failed to predict values with KNN model for dataset with unrecognized task {}".format(self.dataset.task))

    def _predict_class(self, df: pd.DataFrame, x: pd.Series) -> str:
        """
        Predict class based on mode of k nearest neighbors.
        """
        distances = [self._distance(df, x, self.df_train, x_train) for _, x_train in self.df_train.iterrows()]
        k_nearest_neighbors = self.df_train.iloc[np.argsort(distances)[:self.k]]
        return k_nearest_neighbors['class'].mode()[0]

    def _predict_output(self, df: pd.DataFrame, x: pd.Series) -> float:
        """
        Predict regression output using KNN smoother, as defined by:
        ghat(x) = sum_t (K(x, x_t) * y_t) / sum_t (K(x, x_t))
        where
            x_t = t'th training example
            y_t = t'th training example's output
            K(x, x_t) = exp[-γ ||x - x_t||_2]
            ||a - b||_2 = Euclidean distance between a and b
        """
        distances = [self._distance(df, x, self.df_train, x_train) for _, x_train in self.df_train.iterrows()]
        k_nearest_distances = np.sort(distances)[:self.k]
        k_nearest_neighbors = self.df_train.iloc[np.argsort(distances)[:self.k]]
        numerator = np.sum(
            [np.exp(-self.γ * d) * y for d, y in zip(k_nearest_distances, k_nearest_neighbors['output'])]
        )
        denominator = np.sum([np.exp(-self.γ * d) for d in k_nearest_distances])
        return numerator / denominator


class EditedKNNModel(KNNModel):
    """
    Edited K-Nearest Neighbors Model.
    """

    def __init__(self, k: int, df_val: pd.DataFrame, ε: float = 0, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param df_val: pd.DataFrame, Validation set
        :param ε: float, Maximum allowed error rate on validation set
        :param max_iterations: int, Maximum number of iterations to run the edited KNN algorithm
        """
        super().__init__(k)
        self.df_val = df_val
        self.ε = ε
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._edit_training_set()
        self.df_train.reset_index(drop=True, inplace=True)

    def _edit_training_set(self):
        """
        Edited KNN algorithm:
            1. Consider each data point individually.
            2. For each data point, use its single nearest neighbor to make a prediction.
            3. If the prediction is correct, mark the data point for deletion.
            4. Stop editing once performance on the validation set starts to degrade.
        """

        iterations = 0
        prev_val_accuracy = 0 # For classification
        prev_val_mse = np.inf # For regression
        prev_df_train = self.df_train.copy()
        while iterations < self.max_iterations:
            # (1–2) Predict each training example based on its single nearest neighbor
            # (*don't forget to exclude itself when looking for its nearest neighbor 😄*)
            self.df_train['prediction'] = self.df_train.apply(lambda row: self._predict_1nn(row), axis=1)

            # (3) Remove all rows in the training set that are correctly classified by
            # their single nearest neighbor.
            if self.dataset.task == 'classification':
                incorrectly_classified = self.df_train.loc[
                    self.df_train['prediction'] != self.df_train['class']
                ]
            elif self.dataset.task == 'regression':
                incorrectly_classified = self.df_train.loc[
                    np.abs(self.df_train['prediction'] - self.df_train['output']) > self.ε
                ]
            else:
                raise Exception("Failed to edit training set for dataset with unrecognized task {}".format(self.dataset.task))

            if incorrectly_classified.shape[0] == 0:
                # If this iteration would remove all remaining rows of the training set,
                # revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            self.df_train = incorrectly_classified.drop('prediction', axis=1)

            # (4) Stop editing if performance on the validation set starts to degrade.
            self.df_val['prediction'] = self.predict(self.df_val)
            if self.dataset.task == 'classification':
                val_accuracy = metrics.compute_metrics(
                    actual=self.df_val['class'].to_numpy(),
                    predicted=self.df_val['prediction'].to_numpy(),
                    metrics=['acc'],
                )[0]
                if val_accuracy < prev_val_accuracy:
                    # If accuracy has decreased, revert to the previous training set and stop editing.
                    self.df_train = prev_df_train
                    return
            elif self.dataset.task == 'regression':
                val_mse = metrics.compute_metrics(
                    actual=self.df_val['output'].to_numpy(),
                    predicted=self.df_val['prediction'].to_numpy(),
                    metrics=['mse'],
                )[0]
                if val_mse > prev_val_mse:
                    # If MSE has increased, revert to the previous training set and stop editing.
                    self.df_train = prev_df_train
                    return
            else:
                raise Exception("Failed to edit training set for dataset with unrecognized task {}".format(self.dataset.task))

            self.df_val.drop('prediction', axis=1, inplace=True)
            prev_df_train = self.df_train.copy()
            iterations += 1

    def _predict_1nn(self, x: pd.Series) -> any:
        """
        Predict class or output based on single nearest neighbor.

        :param x: pd.Series, Example to predict
        :return str|float, Predicted class or output
        """
        index = x.name
        df_x = self.df_train[self.df_train.index == index]
        df_train = self.df_train.drop(index).drop('prediction', axis=1, errors='ignore')

        validation_model = KNNModel(1)
        validation_model.train(df_train, self.dataset)
        return validation_model.predict(df_x).iloc[0]

class CondensedKNNModel(KNNModel):
    """
    Condensed K-Nearest Neighbors Model.
    """

    def __init__(self, k: int, ε: float = 0, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param ε: float, Tolerance for determining if a data point is correctly predicted for regression
        :param max_iterations: int, Maximum number of iterations to run the condensed KNN algorithm
        """
        super().__init__(k)
        self.ε = ε
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._condense_training_set()
        self.df_train.reset_index(drop=True, inplace=True)

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

        # (1) Add first training example to condensed set
        condensed_set = self.df_train.iloc[[0]]
        self.df_train.drop(0, inplace=True)

        # (5) Make multiple passes through the data in the training set that has not been added until the condensed set stops changing.
        prev_condensed_set_size = condensed_set.shape[0]
        while iterations < self.max_iterations and self.df_train.shape[0] > 0:
            # (2) Consider the remaining data points in the training set individually.
            for i, x in self.df_train.iterrows():
                prediction = self._predict_1nn(x, condensed_set)
                is_correct = False
                if self.dataset.task == 'classification':
                    is_correct = prediction == x['class']
                elif self.dataset.task == 'regression':
                    is_correct = np.abs(prediction - x['output']) <= self.ε
                else:
                    raise Exception("Failed to condense training set for dataset with unrecognized task {}".format(self.dataset.task))

                if not is_correct:
                    # (4) If the prediction is incorrect, add the data point to the condensed set.
                    condensed_set = pd.concat([condensed_set, self.df_train.loc[[i]]])

            if condensed_set.shape[0] == prev_condensed_set_size:
                # (5) If the condensed set stops changing, stop condensing.
                break

            self.df_train = self.df_train[~self.df_train.index.isin(condensed_set.index)]
            prev_condensed_set_size = condensed_set.shape[0]
            iterations += 1
        self.df_train = condensed_set.copy()

    def _predict_1nn(self, x: pd.Series, condensed_set: pd.DataFrame) -> any:
        """
        Predict class or output based on single nearest neighbor in condensed set.

        :param x: pd.Series, Example to predict
        :param condensed_set: pd.DataFrame, Condensed training set

        :return str|float, Predicted class or output
        """
        index = x.name
        df_x = self.df_train[self.df_train.index == index]
        validation_model = KNNModel(1)
        validation_model.train(condensed_set, self.dataset)
        return validation_model.predict(df_x).iloc[0]

