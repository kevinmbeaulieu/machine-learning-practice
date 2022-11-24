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
        self.classes = df['class'].unique()

        if self.verbose:
            print("Training for {} epochs".format(self.num_epochs))
        for epoch in range(self.num_epochs):
            if self.verbose:
                print("Starting epoch {}...".format(epoch))
            batches = self._get_batches(df)
            if self.verbose:
                print("  Got {} batches".format(len(batches)))
            for batch_index, batch in enumerate(self._get_batches(df)):
                if self.verbose:
                    print("  Training batch {}".format(batch_index))
                self._train_batch(batch, batch_index)

            if self.verbose:
                actual = df['class']
                expected = self.predict(df.drop('class', axis=1))
                train_acc = compute_metrics(
                    actual.to_numpy(),
                    expected.to_numpy(),
                    metrics=dataset.metrics,
                    use_sklearn=False
                )
                print("  Train accuracy: {}".format(train_acc))

    def _get_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        num_batches = np.ceil(df.shape[0] / self.batch_size).astype(int)
        if self.verbose:
            print("Creating {} batches".format(num_batches))
        return [df.sample(self.batch_size) for _ in range(num_batches)]

    def _train_batch(self, batch: pd.DataFrame):
        dW = []
        for t, row in batch.iterrows():
            dW_t = self._backpropagate(row)
            if dW:
                dW = [dw + self.learning_rate * dw_t for dw, dw_t in zip(dW, dW_t)]
            else:
                dW = [self.learning_rate * dw_t for dw_t in dW_t]

        for i, layer in enumerate(self.layers[1:]):
            layer.update_weights(dW[i])

    def _backpropagate(self, row: pd.Series) -> list[np.ndarray]:
        """
        Train row via forward-propagation followed by back-propagation.

        :param row: Training example
        :return: Change in weights
        """

        # Notation:
        #   row = input row t (= x^t = s_0^t = a_0^t)
        #
        #   L = number of layers
        #   N_i = number of neurons in layer i
        #
        #   w_i^t = weights of layer i for x^t ((ℝ^N_i, ℝ^N_{i-1}))
        #   b_i^t = bias of layer i for x^t (ℝ)
        #
        #   f_i(x) = activation function of layer i (ℝ -> ℝ)
        #   f_i'(x) = derivative of activation function of layer i (ℝ -> ℝ)
        #
        #   s_i^t = w_i^t • a_{i-1}^t + b_i^t (ℝ^N_i)
        #   a_i^t = output of layer i for x^t (ℝ^N_i)
        #         = f_i(s_i^t)
        #   y^t = correct output of last layer for x^t (ℝ^N_{L-1})
        #
        #   e^t = error of last layer for x^t (ℝ^N_{L-1})
        #       = a_i^t - y^t
        #   e_i^t = error of layer i for x^t (ℝ^N_{i+1})
        #         = e_{i+1}^t • w_{i+1}^t
        #
        #   η = learning rate (ℝ)
        #   Δw_i^t = change in weights of layer i for x^t (ℝ^N_i, ℝ^N_{i-1})
        #          = η * a_{i-1}^t ⨂ e_i^t
        #   Δw_i = change in weights of layer i for this batch (ℝ^N_i, ℝ^N_{i-1})
        #        = Σ_t(Δw_i^t)
        #   Δb_i^t = change in biases of layer i for x^t (ℝ^N_i)
        #          = η * e_i^t
        #   Δb_i = change in biases of layer i for this batch (ℝ^N_i)
        #        = Σ_t(Δb_i^t)

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        x_t = row.drop(output_col).values
        a_t = []
        for layer in self.layers:
            layer_inputs = (a_t[-1] if a_t else x_t).astype('float')
            a_t.append(layer.forward(layer_inputs))

        correct_class = row[output_col]
        y_t = pd.Series([1 if c == correct_class else 0 for c in self.classes], index=self.classes)
        e_t = [a_t[-1] - y_t]
        print(f"e_{foo}: {(e_t[0] ** 2).sum()}")
        for k in range(len(self.layers) - 2, -1, -1):
            layer_k = self.layers[k]
            layer_kplus1 = self.layers[k + 1]
            e_kplus1_t = e_t[0]
            w_kplus1_t = layer_kplus1.weights
            e_k_t = np.dot(e_kplus1_t, w_kplus1_t)
            e_t.insert(0, e_k_t)

        Δw_t = []
        for k in range(1, len(self.layers) - 1):
            layer = self.layers[k]
            s_kmin1_t = layer._activation_fn_derivative(a_t[k - 1])
            Δw_k_t = np.outer(s_kmin1_t, e_t[k]).T
            Δw_t.append(Δw_k_t)
        Δw_Lmin1_t = np.outer(a_t[-2], e_t[-1]).T
        Δw_t.append(Δw_Lmin1_t)

        return Δw_t

    def predict(self, df: pd.DataFrame) -> pd.Series:
        result_dtype = 'str' if self.dataset.task == 'classification' else 'float'
        result = pd.Series(index=df.index, dtype=result_dtype)
        for _, row in df.iterrows():
            result[row.name] = self._predict_row_class(row)
        return result

    def _predict_row_class(self, row: pd.Series) -> any:
        output_col = 'class' if self.dataset.task == 'classification' else 'output'

        values = row.values
        for i, layer in enumerate(self.layers):
            values = layer.forward(values)

        if self.dataset.task == 'classification':
            return self.classes[values.argmax()]
        else:
            return values

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

    def update_weights(self, Δw: np.ndarray):
        """
        Update the weights of the layer.

        :param Δw: Change in weights.
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
#         self.bias = None

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

#         return self._activate(np.dot(self.weights, inputs) + self.bias)
        return self._activate(np.dot(self.weights, inputs))

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

#         self.bias = np.zeros((self.units,))

    def _activate(self, z: np.ndarray) -> np.ndarray:
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

    def _activation_fn_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation == 'sigmoid':
            return np.exp(z) / (1 + np.exp(z)) ** 2
        elif self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'tanh':
            return np.sech(z) ** 2
        elif self.activation == 'softmax':
            # Jacobian of softmax is a diagonal matrix with the softmax values on the diagonal.
            # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            return np.diag(z) - np.outer(z, z)
        elif self.activation == 'linear':
            return np.ones(z.shape)

    def update_weights(self, Δw: np.ndarray):
        self.weights += Δw

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
