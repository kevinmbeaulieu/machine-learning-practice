import pandas as pd

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


class KNNClassifierModel(Model):
    """
    K-Nearest Neighbors Classifier Model.
    """

    def __init__(self, k: int):
        self.k = k
        self.df_train = None
        self.dataset = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        self.df_train = df
        self.classes = df['class'].unique()
        self.dataset = dataset

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.df_train is None or self.dataset is None:
            raise Exception("Failed to predict values with KNN classifier model before model was trained")

        if self.dataset.task != 'classification':
            raise Exception("Failed to predict values with KNN classifier model for dataset with unrecognized task {}".format(self.dataset.task))

        y_pred = pd.Series(dtype='float64', index=range(df.shape[0]))
        
        for row in df.shape[0]:
            distances: list[tuple[float, str]] = []
            for train_row in self.df_train.shape[0]:
                x = df.iloc[row]
                x_train = self.df_train.iloc[train_row]
                distance = self._distance(x, x_train)
                distances.append((distance, x_train['class']))
            y_pred.iloc[row] = self._get_knn_class(distances)
        
        return y_pred

    def _distance(self, left: pd.Series, right: pd.Series) -> float:
        """
        Calculates the distance between two data points.

        :param left: pd.Series, Data point
        :param right: pd.Series, Data point

        :return float, Distance between the two data points
        """
        if left.shape != right.shape:
            raise Exception("Failed to calculate distance between data points with different shapes")

        distance = 0
        for col in left.size:
            if col in self.dataset.nominal_cols:
                # TODO: Implement nominal distance calculation
                pass
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

    def _get_knn_class(self, distances: list[tuple[float, str]]) -> str:
        """
        Finds the most common class label among the k nearest neighbors.

        :param distances: list[tuple[float, str]], List of (distance, class label) tuples

        :return str, Most common class label among the k nearest neighbors
        """
        distances.sort(key=lambda x: x[0])
        classes = [distances[i][1] for i in range(self.k)]
        return max(classes, key=classes.count)
