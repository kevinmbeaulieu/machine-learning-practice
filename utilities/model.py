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

