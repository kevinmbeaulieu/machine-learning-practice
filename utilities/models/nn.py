import numpy as np
import pandas as pd

from utilities.preprocessing.dataset import Dataset
from .model import Model

class NeuralNetworkModel(Model):
    """
    Neural Network Model.
    """

    def __init__(self, batch_size: int = 32, num_epochs: int = 100, learning_rate: float = 0.01):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.layers = []

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if not isinstance(self.layers[0], InputLayer):
            raise ValueError('First layer must be an input layer.')

        self.dataset = dataset
        self.classes = df['class'].unique()

        for _ in range(self.num_epochs):
            batch_error = 0
            for batch in dataset.get_batches(df, self.batch_size):
                batch_error += self._train_batch(batch)

    def _train_batch(self, batch: pd.DataFrame):
        error = 0
        for _, row in batch.iterrows():
            error += self._train_row(row)
        return error

    def _train_row(self, row: pd.Series) -> float:
        output = row.values
        for layer in self.layers:
            output = layer.forward(outputs)

        expected = row['class'] if self.dataset.task == 'classification' else row['output']
        error = output - expected

        for layer in reversed(self.layers):
            error = layer.backward(error)

        return error

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
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to forward propagate.
        :return: Outputs of the layer.
        """
        raise Exception("Must be implemented by subclass")

    def backward(self, error: np.ndarray) -> np.ndarray:
        """
        Back propagate error through the layer.

        :param error: Error to back propagate.
        :return: Error of the layer.
        """
        raise Exception("Must be implemented by subclass")

class InputLayer(Layer):
    """
    Input layer in a neural network.
    """

    def __init__(self, input_size: int):
        self.input_size = input_size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != (self.input_size,):
            raise Exception("Input size does not match")

        return inputs

    def backward(self, error: np.ndarray) -> np.ndarray:
        return error

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
        self.weights = None
        self.bias = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.input_size is None:
            self.input_size = inputs.shape[0]

        if self.nodes is None:
            # Lazily construct nodes since input size is not known at initialization.
            self._initialize_weights_and_bias()
        elif self.nodes[0].input_size != self.input_size:
            raise Exception("Input size does not match, {} != {}".format(self.nodes[0].input_size, self.input_size))

        z = np.dot(self.weights, inputs) + self.bias
        return self._activate(z)

    def _initialize_weights_and_bias(self):
        if self.activation == 'relu':
            # He-et-al uniform initialization.
            scale = 2.0 / max(1.0, self.input_size)
            limit = np.sqrt(3.0 * scale)
            self.weights = np.random.uniform(-limit, limit, (self.units, self.input_size))
        elif self.activation == 'tanh' or self.activation == 'sigmoid':
            # Xavier uniform initialization.
            scale = 1.0 / max(1.0, (self.input_size + self.units) / 2.0)
            limit = np.sqrt(3.0 * scale)
            self.weights = np.random.uniform(-limit, limit, (self.units, self.input_size))
        else:
            self.weights = np.random.randn(self.units, self.input_size)

        self.bias = np.zeros((self.units, 1))

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            return np.exp(z) / np.sum(np.exp(z))
        elif self.activation == 'linear':
            return z

    def backward(self, error: np.ndarray) -> np.ndarray:
        error = np.dot(self.weights.T, error)
        # TODO: Update weights and bias.
        return error

class DropoutLayer(Layer):
    """
    Dropout layer in a neural network.
    """

    def __init__(self, rate: float):
        """
        :param rate, float: Fraction of nodes to drop out (0.0 - 1.0)
        """
        self.rate = rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        dropout = lambda x: x if np.random.rand() > self.rate else 0
        return dropout(inputs)

    def backward(self, error: np.ndarray) -> np.ndarray:
        return error

