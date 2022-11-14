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
        if not isinstance(self.layers[0], InputLayer):
            raise ValueError('First layer must be an input layer.')

        self.classes = df['class'].unique()

        # TODO: Train via backpropagation.

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=df.index, dtype='object')
        for _, row in df.iterrows():
            result[row.name] = self._predict_row_class(row)
        return result

    def _predict_row_class(self, row: pd.Series) -> str:
        values = row.values
        for layer in self.layers:
            values = layer.forward(values, num_processes=self.num_processes)
        return self.classes[values.argmax()]

class Layer:
    """
    Abstract class for a layer in a neural network.
    """

    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray, num_processes: int) -> np.ndarray:
        """
        Forward propagate inputs through the layer in parallel.

        :param inputs: Inputs to forward propagate.
        :param num_processes: Number of processes to use.
        """
        raise Exception("Must be implemented by subclass")

class InputLayer(Layer):
    """
    Input layer in a neural network.
    """

    def __init__(self, input_size: int):
        self.input_size = input_size

    def forward(self, inputs: np.ndarray, num_processes: int) -> np.ndarray:
        if inputs.shape != (self.input_size,):
            raise Exception("Input size does not match")

        return inputs

class DenseLayer(Layer):
    """
    Dense layer in a neural network.
    """

    def __init__(
        self,
        units: int,
        activation: str = 'linear',
    ):
        """
        :param units, int: Number of nodes in this layer, or number of features if this is
            the input layer
        :param activation, str: Activation function for each node in this layer
            ('sigmoid', 'relu', 'tanh', 'softmax', 'linear'; default: 'linear')
        """
        self.units = units
        self.activation = activation
        self.input_size = None
        self.nodes = None

    def forward(self, inputs: np.ndarray, num_processes: int) -> np.ndarray:
        if self.input_size is None:
            self.input_size = inputs.shape[0]

        if self.nodes is None:
            # Lazily construct nodes since input size is not known at initialization.
            self.nodes = [
                _Node(input_size=self.input_size, activation=self.activation) for _ in range(self.units)
            ]
        elif self.nodes[0].input_size != self.input_size:
            raise Exception("Input size does not match, {} != {}".format(self.nodes[0].input_size, self.input_size))

        pool = Pool(num_processes)
        outputs = pool.starmap(_Node.forward, [(node, inputs) for node in self.nodes])
        pool.close()
        pool.join()
        return np.array(outputs)

class _Node:
    """
    A node in a neural network.
    """

    def __init__(self, input_size: int, activation: str = 'linear'):
        """
        :param input_size, int: Number of inputs to this node (0 for nodes in input layer)
        :param activation: Activation function for this node ('sigmoid', 'relu', 'tanh',
            'softmax', 'linear'; default: 'linear')
        """
        self.activation = activation
        self.input_size = input_size
        self.weights = np.random.randn(input_size)

    def forward(self, inputs: np.ndarray) -> float:
        """
        :param inputs, np.ndarray: Inputs to this node
        :return: Output of this node.
        """
        value = np.dot(inputs, self.weights)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-value))
        elif self.activation == 'relu':
            return max(0, value)
        elif self.activation == 'tanh':
            return np.tanh(value)
        elif self.activation == 'softmax':
            return np.exp(value) / np.sum(np.exp(value))
        elif self.activation == 'linear':
            return value
