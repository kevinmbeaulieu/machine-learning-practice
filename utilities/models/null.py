import pandas as pd

from utilities.preprocessing.dataset import Dataset
from .model import Model

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
