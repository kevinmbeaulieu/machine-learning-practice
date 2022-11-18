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

        print("Training for {} epochs".format(self.num_epochs))
        for epoch in range(self.num_epochs):
            print("Starting epoch {}...".format(epoch))
            batches = self._get_batches(df)
            print("  Got {} batches".format(len(batches)))
            for batch_index, batch in enumerate(self._get_batches(df)):
                print("  Training batch {}".format(batch_index))
                self._train_batch(batch)

    def _get_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        num_batches = np.ceil(df.shape[0] / self.batch_size).astype(int)
        print("Creating {} batches".format(num_batches))
        return [df.sample(self.batch_size) for _ in range(num_batches)]

    def _train_batch(self, batch: pd.DataFrame):
        Δw = np.zeros((len(self.layers,)))
        Δb = np.zeros((len(self.layers,)))
        for _, row in batch.iterrows():
            Δw_t, Δb_t = self._train_row(row)
            Δw += Δw_t
            Δb += Δb_t

        print("Δw:", Δw)
        print("Δb:", Δb)

    def _train_row(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """
        Train row via forward-propagation followed by back-propagation.

        :param row: Training example
        :return: Tuple of changes to weights (first element) and biases for each layer (second element)
        """

        # Notation:
        #   row = input row t (= x^t = z_0^t)
        #
        #   f_i(x) = activation function of layer i
        #   f_i'(x) = derivative of activation function of layer i
        #
        #   z_i^t = output of layer i for x^t
        #         = f_i(w_i^t • z_{i-1}^t + b_i^t)
        #   y^t = output of last layer for x^t
        #       = z_{L-1}^t
        #
        #   e^t = error of last layer for x^t
        #       = y\hat^t - y^t
        #   e_i^t = error of layer i for x^t for i < L-1
        #         = e_{i+1}^t • w_{i+1}^t
        #
        #   η = learning rate
        #   Δw_i^t = change in weights of layer i for x^t
        #          = η • e_i^t • f_i'(z_i^t) • z_{i-1}^t
        #   Δw_i = change in weights of layer i for this batch
        #        = sum(Δw_i^t)
        #   Δb_i^t = change in biases of layer i for x^t
        #          = η • e_i^t • f_i'(z_i^t)
        #   Δb_i = change in biases of layer i for this batch
        #        = sum(Δb_i^t)

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        z = []
        for layer in self.layers:
            layer_inputs = (z[-1] if z else row.drop(output_col).values).astype('float')
            z.append(layer.forward(layer_inputs))

        # TODO: Encode output column for use with softmax before running through NN model.
        y = row[output_col]
        e = [z[-1] - y]
        for layer_index in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]
            next_layer_error = e[0]
            e.insert(0, np.dot(next_layer_error, next_layer.weights.T))

        Δw = []
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            Δw.append(η * e[i] * layer._activate_derivative(z[i]) * z[i - 1])

        Δb = []
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            Δb.append(η * e[i] + layer._activate_derivative(z[i]))

        return Δw, Δb

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(index=df.index)
        for _, row in df.iterrows():
            result[row.name] = self._predict_row_class(row)
        return result

    def _predict_row_class(self, row: pd.Series) -> str:
        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        values = row.drop(output_col).values
        for i, layer in enumerate(self.layers):
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

class InputLayer(Layer):
    """
    Input layer in a neural network.
    """

    def __init__(self, input_size: int):
        self.input_size = input_size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != (self.input_size,):
            raise Exception("Input size does not match ({} ≠ {})", inputs.shape, (self.input_size,))

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
        self.weights = None
        self.bias = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to forward propagate.
        :return: Outputs of the layer (shape: (units,)).
        """
        if self.input_size is None:
            self.input_size = inputs.shape[0]

        if self.weights is None:
            # Lazily construct weights to avoid having to specify input size on initialization.
            self._initialize_weights_and_bias()

        return self._activate(np.dot(self.weights, inputs) + self.bias)

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

        self.bias = np.zeros((self.units,))

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

    def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return np.exp(z) / (1 + np.exp(z)) ** 2
        elif self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'tanh':
            return np.sech(z) ** 2
        elif self.activation == 'softmax':
            s = self._activate(z).reshape(-1, 1)
            return np.diagflat(s) - np.dot(s, s.T)
        elif self.activation == 'linear':
            return np.ones(z.shape)

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
