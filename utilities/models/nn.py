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
        self.df_validation = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if not isinstance(self.layers[0], InputLayer):
            raise ValueError('First layer must be an input layer.')

        self.dataset = dataset
        if self.dataset.task == 'classification':
            self.classes = df['class'].unique()

        if self.verbose:
            print("Training for {} epochs".format(self.num_epochs))

        metrics_train = []
        metrics_validation = []
        for epoch in range(self.num_epochs):
            if self.verbose:
                print("Starting epoch {}...".format(epoch))

            for batch_index, batch in enumerate(self._get_batches(df)):
                if self.verbose:
                    print("  Training batch {}".format(batch_index))
                self._train_batch(batch)

            if self.verbose:
                metrics_train.append(self._compute_epoch_metrics(df))
                if self.df_validation is not None:
                    metrics_validation.append(self._compute_epoch_metrics(self.df_validation))

        if metrics_train:
            self._plot_epoch_metrics(metrics_train, metrics_validation)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_dtype = 'str' if self.dataset.task == 'classification' else np.double
        result = pd.Series(index=df.index, dtype=result_dtype)
        for _, row in df.iterrows():
            x = row.values.reshape(-1, 1)
            a_back = x
            a = None
            for layer in self.layers:
                a = layer.forward(a_back)
                a_back = a

            if self.dataset.task == 'classification':
                result[row.name] = self.classes[a.argmax()]
            else:
                result[row.name] = a[0]
        return result

    def _get_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Split the dataset into batches.

        :param df: Training Dataset
        :return: List of batches
        """

        num_batches = np.ceil(df.shape[0] / self.batch_size).astype(int)
        return [df.sample(self.batch_size) for _ in range(num_batches)]

    def _train_batch(self, batch: pd.DataFrame):
        """
        Train the model on a batch of data.

        :param batch: Batch of data
        """

        Δw, Δb = [], []
        for r in range(batch.shape[0]):
            row = batch.iloc[r]
            Δw_t, Δb_t = self._train_row(row)
            Δw = [Δw[i] + Δw_t[i] for i in range(len(Δw))] if Δw else Δw_t
            Δb = [Δb[i] + Δb_t[i] for i in range(len(Δb))] if Δb else Δb_t

        # Scale down since we just summed over all rows in the batch
        Δw = [Δw[i] / batch.shape[0] for i in range(len(Δw))]
        Δb = [Δb[i] / batch.shape[0] for i in range(len(Δb))]

        assert len(Δw) == len(self.layers) - 1
        assert len(Δb) == len(self.layers) - 1
        for i, layer in enumerate(self.layers[1:]):
            layer.update_weights(Δw[i], Δb[i])

    def _train_row(self, row: pd.Series) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Train row via forward-propagation followed by back-propagation.

        :param row: Training example
        :return: Change in weights/biases for each layer
        """

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        x = row.drop(output_col).values.reshape(-1, 1)
        if self.dataset.task == 'classification':
            y = np.array([1 if c == row[output_col] else 0 for c in self.classes]).reshape(-1, 1)
        else:
            y = row[output_col]

        a = [] # a[k] = output of layer k
        for k, layer in enumerate(self.layers):
            a_back = (x if k == 0 else a[k - 1]).astype(np.double)
            a.append(layer.forward(a_back))

        da_forward = self._compute_loss_partial(y, a[-1])
        Δw, Δb = [], []
        for k in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[k]
            Δw_k, Δb_k, da_k = layer.backward(
                a=a[k],
                da_forward=da_forward,
                a_back=a[k - 1],
            )
            Δw.insert(0, self.learning_rate * Δw_k)
            Δb.insert(0, self.learning_rate * Δb_k)
            da_forward = da_k

        return Δw, Δb

    def _compute_loss_partial(self, y: np.ndarray, a: np.ndarray) -> np.double:
        """
        Compute partial derivative of loss function with respect to input of layer
        (i.e., δ = ∂loss/∂a).

        For classification, loss is the cross-entropy loss.
        For regression, loss is the mean squared error.

        :param y: Expected output (shape: (1, num_classes))
        :param a: Actual output (shape: (1, num_classes))
        :return: Partial derivative of loss function with respect to input of layer (shape: (1, num_classes))
        """

        # TODO: Is it true that the partial derivative of the loss function happens to be
        # the same for both categorical cross-entropy and mean squared error?
        if self.dataset.task == 'classification':
            # ∂E/∂z for E = -y * log(a)
            return a - y
        else:
            # ∂E/∂z for E = (y - a)^2 / 2
            return a - y

    def _compute_epoch_metrics(self, df: pd.DataFrame) -> list[float]:
        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        actual = df[output_col]
        expected = self.predict(df.drop(output_col, axis=1))
        epoch_metrics = compute_metrics(
            actual.to_numpy(),
            expected.to_numpy(),
            metrics=self.dataset.metrics,
            use_sklearn=True
        )
        print("  Epoch metrics:")
        for i, metric in enumerate(self.dataset.metrics):
            print(f"    {metric}: {epoch_metrics[i]}")
        return epoch_metrics

    def _plot_epoch_metrics(self, values_train: list[list[float]], values_validation: list[list[float]] = None):
        for i, metric in enumerate(self.dataset.metrics):
            plt.title(metric)
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            if self.dataset.task == 'regression':
                plt.yscale('log')
            map_metric_values = lambda values: [v[i] for v in values]
            plt.plot(map_metric_values(values_train), label='train')
            plt.plot(map_metric_values(values_validation), label='validation')
            plt.legend()
            plt.show()

class Layer:
    """
    Abstract class for a layer in a neural network.
    """

    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to forward propagate (e.g., outputs of previous layer).
        :return: Outputs of the layer.
        """
        raise Exception("Must be implemented by subclass")

    def backward(self, a: np.ndarray, da_forward: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagate error through the layer.

        :param a: Outputs of this layer (shape: (1, num_neurons_in_layer))
        :param da_forward: Error of next layer (shape: (1, num_neurons_in_next_layer))
        :param a_back: Outputs of previous layer (shape: (1, num_neurons_in_layer-1))
        :return: Change in weights, change in biases, and error for this layer
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
        if inputs.shape != (self.input_size, 1):
            raise Exception("Input size does not match ({} ≠ {})".format(inputs.shape, (self.input_size, 1)))

        return inputs

    def backward(self, a: np.ndarray, da_forward: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return None, None, None

class DenseLayer(Layer):
    """
    Dense layer in a neural network.
    """

    def __init__(
        self,
        units: int,
        activation: str = 'linear',
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        :param units, int: Number of nodes in this layer, or number of features if this is
            the input layer
        :param activation, str: Activation function for each node in this layer
            ('sigmoid', 'relu', 'tanh', 'softmax', 'linear'; default: 'linear')
        :param rng, np.random.Generator: Random number generator
        """

        self.units = units
        self.activation = activation
        self.rng = rng
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
        elif inputs.shape != (self.weights.shape[1], 1):
            raise Exception("Input size does not match ({} ≠ {})".format(inputs.shape, (self.weights.shape[1], 1)))

        z = self.weights.dot(inputs) + self.bias
        return self._activate(z)

    def backward(self, a: np.ndarray, da_forward: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagate error through the layer.

        :param a: Outputs of this layer (shape: (1, num_neurons_in_layer))
        :param da_forward: Error of next layer (shape: (1, num_neurons_in_next_layer))
        :param a_back: Outputs of previous layer (shape: (1, num_neurons_in_layer-1))
        :return: Change in weights, change in biases, and error for this layer
        """

        da_dz = self._activation_fn_derivative(a)
        dz = (da_forward * da_dz)
        Δw = -dz @ a_back.T / self.units
        Δb = -dz.mean(axis=1, keepdims=True)

        da = self.weights.T.dot(dz)
        return Δw, Δb, da

    def update_weights(self, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights of the layer.

        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """

        self.weights += Δw
        self.bias += Δb

    def _initialize_weights(self, input_size: int):
        """
        Initialize weights and bias.

        :param input_size: Number of inputs to this layer.
        """

        if self.activation == 'relu':
            # He-et-al uniform initialization.
            scale = 2.0 / max(1.0, input_size)
            limit = np.sqrt(3.0 * scale)
            self.weights = self.rng.uniform(-limit, limit, (self.units, input_size))

        elif self.activation == 'tanh' or self.activation == 'sigmoid':
            # Xavier uniform initialization.
            scale = 1.0 / max(1.0, (input_size + self.units) / 2.0)
            limit = np.sqrt(3.0 * scale)
            self.weights = self.rng.uniform(-limit, limit, (self.units, input_size))

        else:
            self.weights = self.rng.standard_normal((self.units, input_size))

        self.bias = np.zeros((self.units, 1))

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

    def _activation_fn_derivative(self, a: np.ndarray) -> np.ndarray:
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
