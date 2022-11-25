import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities.metrics import compute_metrics
from utilities.preprocessing.dataset import Dataset
from .model import Model

class NeuralNetworkModel(Model):
    """
    Neural Network Model.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = False
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.layers = []
        self.verbose = verbose

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if not isinstance(self.layers[0], InputLayer):
            raise ValueError('First layer must be an input layer.')

        self.dataset = dataset
        if self.dataset.task == 'classification':
            self.classes = df['class'].unique()

        if self.verbose:
            print("Training for {} epochs".format(self.num_epochs))
        metrics = []
        for epoch in range(self.num_epochs):
            if self.verbose:
                print("Starting epoch {}...".format(epoch))
            batches = self._get_batches(df)
            for batch_index, batch in enumerate(self._get_batches(df)):
                if self.verbose:
                    print("  Training batch {}".format(batch_index))
                self._train_batch(batch)

            if self.verbose:
                actual = df['class']
                expected = self.predict(df.drop('class', axis=1))
                epoch_metrics = compute_metrics(
                    actual.to_numpy(),
                    expected.to_numpy(),
                    metrics=dataset.metrics,
                    use_sklearn=False
                )
                metrics.append(epoch_metrics)
                print("  Epoch metrics:")
                for i, metric in enumerate(dataset.metrics):
                    print(f"    {metric}: {epoch_metrics[i]}")

        for i, metric in enumerate(dataset.metrics):
            plt.title(metric)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.plot([metric[i] for metric in metrics])
            plt.show()

    def _get_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        num_batches = np.ceil(df.shape[0] / self.batch_size).astype(int)
        return [df.sample(self.batch_size) for _ in range(num_batches)]

    def _train_batch(self, batch: pd.DataFrame):
        Δw, Δb = [], []
        for r in range(batch.shape[0]):
            row = batch.iloc[r]
            Δw_t, Δb_t = self._backpropagate(row)
            Δw = [Δw[i] + Δw_t[i] for i in range(len(Δw))] if Δw else Δw_t
            Δb = [Δb[i] + Δb_t[i] for i in range(len(Δb))] if Δb else Δb_t

        Δw = [Δw[i] / batch.shape[0] for i in range(len(Δw))]
        Δb = [Δb[i] / batch.shape[0] for i in range(len(Δb))]

        assert len(Δw) == len(self.layers) - 1
        assert len(Δb) == len(self.layers) - 1
        for i, layer in enumerate(self.layers[1:]):
            layer.update_weights(Δw[i], Δb[i])

    def _backpropagate(self, row: pd.Series) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Train row via forward-propagation followed by back-propagation.

        :param row: Training example
        :return: Change in weights/biases for each layer
        """

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        x_t = row.drop(output_col).values
        if self.dataset.task == 'classification':
            y_t = np.array([1 if c == row[output_col] else 0 for c in self.classes])
        else:
            y_t = row[output_col]

        a_t = [] # shape: (num_layers, num_neurons_in_layer)
        z_t = [] # shape: (num_layers, num_neurons_in_layer)
        for k, layer in enumerate(self.layers):
            z_t_k = (a_t[-1] if a_t else x_t).astype('float64')
            a_t_k = layer.forward(z_t_k)
            z_t.append(z_t_k)
            a_t.append(a_t_k)

        e_t_k = self._compute_loss_partial(y_t, a_t[-1])
        Δw_t, Δb_t = [], []
        for k in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[k]
            if k < len(self.layers) - 1:
                e_t_k = e_t_k.reshape(1, -1) @ self.layers[k + 1].weights
            f_prime = layer.activation_fn_derivative(a_t[k - 1])
            delta = -self.learning_rate * e_t_k.reshape(-1, 1) @ f_prime.reshape(1, -1)
            Δw_t_k = delta * z_t[k]
            Δb_t_k = delta.mean(axis=1)
            Δw_t.insert(0, Δw_t_k)
            Δb_t.insert(0, Δb_t_k)

        return Δw_t, Δb_t

    def _compute_loss_partial(self, y_t: np.ndarray, a_t: np.ndarray) -> float:
        """
        Compute partial derivative of loss function with respect to input of layer
        (i.e., δ_t = ∂loss/∂a_t).

        For classification, loss is the cross-entropy loss.
        For regression, loss is the mean squared error.

        :param y_t: Expected output (shape: (1, num_classes))
        :param a_t: Actual output (shape: (1, num_classes))
        :return: Partial derivative of loss function with respect to input of layer (shape: (1, num_classes))
        """

        if self.dataset.task == 'classification':
            # ∂E/∂z for E = -y * log(a)
            return a_t - y_t
        else:
            # ∂E/∂z for E = (y - a)^2 / 2
            return a_t - y_t

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_dtype = 'str' if self.dataset.task == 'classification' else 'float64'
        result = pd.Series(index=df.index, dtype=result_dtype)
        for _, row in df.iterrows():
            values = row.values
            for i, layer in enumerate(self.layers):
                values = layer.forward(values)

            if self.dataset.task == 'classification':
                result[row.name] = self.classes[values.argmax()]
            else:
                result[row.name] = values[0]
        return result

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

    def update_weights(self, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights of the layer.

        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """
        pass

class InputLayer(Layer):
    """
    Input layer in a neural network.
    """

    def __init__(self, input_size: int):
        self.input_size = input_size

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != (self.input_size,):
            raise Exception("Input size does not match ({} ≠ {})".format(inputs.shape, (self.input_size,)))

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
        self.weights = None
        self.bias = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to forward propagate.
        :return: Outputs of the layer (shape: (units,)).
        """

        if self.weights is None:
            # Lazily construct weights to avoid having to specify input size on initialization.
            self._initialize_weights(inputs.shape[0])
        elif inputs.shape != (self.weights.shape[1],):
            raise Exception("Input size does not match ({} ≠ {})".format(inputs.shape, (self.weights.shape[1],)))

        return self._activate(np.dot(self.weights, inputs))

    def _initialize_weights(self, input_size: int):
        """
        Initialize weights and bias.

        :param input_size: Number of inputs to this layer.
        """

        if self.activation == 'relu':
            # He-et-al uniform initialization.
            scale = 2.0 / max(1.0, input_size)
            limit = np.sqrt(3.0 * scale)
            self.weights = np.random.uniform(-limit, limit, (self.units, input_size))

        elif self.activation == 'tanh' or self.activation == 'sigmoid':
            # Xavier uniform initialization.
            scale = 1.0 / max(1.0, (input_size + self.units) / 2.0)
            limit = np.sqrt(3.0 * scale)
            self.weights = np.random.uniform(-limit, limit, (self.units, input_size))

        else:
            self.weights = np.random.randn(self.units, input_size)

        self.bias = np.zeros(self.units)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """
        Apply activation function to inputs.

        :param z: Inputs to activation function (shape: (units,)).
        :return: Outputs of activation function (shape: (units,)).
        """

        if self.activation == 'sigmoid':
            # Avoid overflow.
            sigmoid = lambda x: (np.exp(x) / (1 + np.exp(x))) if x < 0 else (1 / (1 + np.exp(-x)))
            return np.array([sigmoid(x) for x in z])
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'softmax':
            z -= np.max(z) # Avoid overflow.
            return np.exp(z) / np.sum(np.exp(z))
        elif self.activation == 'linear':
            return z

    def activation_fn_derivative(self, a: np.ndarray) -> np.ndarray:
        """
        Compute the partial derivative of the activation function with respect to its inputs
        for each node in this layer.

        Note: The parameter a is the output of the activation function, not the input. This
        simplifies the implementation since the derivative of all the activation functions can
        be represented in terms of the output of the activation function.

        :param a: Outputs of activation function (shape: (units,)).
        :return: Partial derivatives of activation function with respect to its inputs (shape: (units,)).
        """

        if self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'relu':
            return np.where(a > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - a ** 2
        elif self.activation == 'softmax':
            return a * (1 - a)
        elif self.activation == 'linear':
            return np.ones(a.shape)

    def update_weights(self, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights of the layer.

        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """

        self.weights += Δw
        self.bias += Δb

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
        # For each row in input, replace it with zero with probability self.rate.
        return np.where(np.random.rand(*inputs.shape) < self.rate, 0, inputs)
