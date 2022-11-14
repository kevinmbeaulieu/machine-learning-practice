from multiprocessing import Pool
import numpy as np
import os
import pandas as pd

from utilities.preprocessing.dataset import Dataset
from .model import Model

class NeuralNetworkModel(Model):
    """
    Neural Network Model.
    """

    def __init__(self, num_processes: int = os.cpu_count()):
        self.num_processes = num_processes
        self.layers = []

    def train(self, df: pd.DataFrame, dataset: Dataset):
        self.df = df
        self.classes = df['class'].unique()

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=df.index, dtype='object')
        for _, row in df.iterrows():
            result[row.name] = self._predict_row_class(row)
        return result

    def _predict_row_class(self, row: pd.Series) -> str:
        values = row.values
        for layer in self.layers:
            values = layer.forward(values)
        return self.classes[values.argmax()]

class Layer:
    """
    Abstract class for a layer in a neural network.
    """

    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise Exception("Must be implemented by subclass")

class DenseLayer(Layer):
    """
    Dense layer in a neural network.
    """

    def __init__(
        self,
        units: int,
        input_size: int,
        activation: str = 'linear',
        num_processes: int = os.cpu_count()
    ):
        self.nodes = [Node(input_size=input_size, activation=activation) for _ in range(units)]
        self.num_processes = num_processes

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pool = Pool(self.num_processes)
        outputs = pool.starmap(Node.forward, [(node, inputs) for node in self.nodes])
        pool.close()
        pool.join()
        return np.array(outputs)

class Node:
    """
    A node in a neural network.
    """

    def __init__(self, input_size: int, activation: str):
        """
        :param activation: Activation function for this node ('sigmoid', 'relu', 'tanh',
            'softmax', 'linear'; default: 'linear')
        """
        self.activation = activation
        self.weights = np.random.randn(input_size)

    def forward(self, inputs: np.ndarray) -> float:
        """
        :param inputs, np.ndarray: Inputs to this node
        :return: Output of this node.
        """
        if inputs.shape != self.weights.shape:
            raise ValueError('Input shape does not match weight shape, {} != {}'.format(inputs.shape, self.weights.shape))

        value = np.dot(inputs, self.weights)
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif activation == 'relu':
            return max(0, value)
        elif activation == 'tanh':
            return np.tanh(value)
        elif activation == 'softmax':
            return np.exp(value) / np.sum(np.exp(value))
        elif activation == 'linear':
            return value

