from typing import Tuple
import numpy as np

class Layer:
    """
    Abstract class for a layer in a neural network.
    """

    def __init__(self):
        self.id = id(self)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to forward propagate (e.g., outputs of previous layer).
        :return: Outputs of the layer.
        """
        raise NotImplementedError

    def backward(self, a: np.ndarray, da: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagate error through the layer.

        :param a: Outputs of this layer (shape: (1, num_neurons_in_layer))
        :param da: Error of this layer's output (shape: (1, num_neurons_in_layer))
        :param a_back: Outputs of previous layer (shape: (1, num_neurons_in_layer-1))
        :return: Change in weights/bias for this layer, error of previous layer's output
        """
        raise NotImplementedError

    def update_weights(self, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights of the layer. Default behavior is to do nothing.

        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """
        pass

class Input(Layer):
    """
    Input layer in a neural network.
    """

    def __init__(self, input_size: int):
        self.input_size = input_size
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.shape != (self.input_size, 1):
            raise ValueError("Input size does not match ({} ≠ {})".format(inputs.shape, (self.input_size, 1)))

        return inputs

class Dense(Layer):
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
        super().__init__()

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
            raise ValueError("Input size does not match ({} ≠ {})".format(inputs.shape, (self.weights.shape[1], 1)))

        z = self.weights.dot(inputs) + self.bias
        return self._activate(z)

    def backward(self, a: np.ndarray, da: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagate error through the layer.

        :param a: Outputs of this layer (shape: (1, num_neurons_in_layer))
        :param da: Error of this layer's output (shape: (1, num_neurons_in_layer))
        :param a_back: Outputs of previous layer (shape: (1, num_neurons_in_layer-1))
        :return: Change in weights/biases for this layer and error for previous layer
        """

        da_dz = self._activation_fn_derivative(a)
        dz = da * da_dz
        Δw = -dz @ a_back.T / self.units
        Δb = -dz.mean(axis=1, keepdims=True)

        da_back = self.weights.T.dot(dz)
        return Δw, Δb, da_back

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

class Convolutional(Layer):
    """
    Convolutional layer in a neural network.
    """

    def __init__(
        self,
        kernel_size: tuple[int, int],
        stride: int = 1,
        activation: str = 'linear',
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        :param kernel_size, tuple[int, int]: Size of the convolutional kernel (height, width).
        :param stride, int: Stride of the convolutional kernel (default: 1).
        :param activation, str: Activation function for each node in this layer
            ('sigmoid', 'relu', 'tanh', 'linear'; default: 'linear')
        :param rng, np.random.Generator: Random number generator
        """

        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.rng = rng
        self.kernel = None
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward propagate inputs through the layer.

        :param inputs: Inputs to the layer (shape: (height, width)).
        :return: Outputs of the layer (shape: (height, width)).
        """

        if self.kernel is None:
            # Lazily construct kernel to avoid having to specify input size on initialization.
            self._initialize_kernel(inputs.shape)
        elif inputs.shape != (self.kernel.shape):
            raise ValueError("Input size does not match ({} ≠ {})".format(inputs.shape, (self.kernel.shape)))

        return self._activate(z)

    def backward(
        self,
        a: np.ndarray,
        da: np.ndarray,
        a_back: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def update_weights(self, Δw: np.ndarray, Δb: np.ndarray):
        raise NotImplementedError

    def _initialize_kernel(self, input_size: Tuple[int, ...]):
        if self.activation == 'relu':
            # He-et-al normal initialization.
            variance = 2.0 / max(1.0, input_size)
            self.kernel = self.rng.normal(0, variance, self.kernel_size)

        elif self.activation == 'tanh' or self.activation == 'sigmoid':
            # Xavier normal initialization.
            variance = 1.0 / max(1.0, (input_size + np.prod(self.kernel_size)) / 2.0)
            self.kernel = self.rng.normal(0, variance, self.kernel_size)

        else:
            self.kernel = self.rng.standard_normal(self.kernel_size)

class Dropout(Layer):
    """
    Dropout layer in a neural network.
    """

    def __init__(
        self,
        rate: float,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        :param rate, float: Fraction of nodes to drop out (0.0 - 1.0)
        :param rng, np.random.Generator: Random number generator
        """

        self.rate = rate
        self.rng = rng
        self.training = True
        self.weights = None # Mask of nodes to drop (shape: (input_size, 1))
        super().__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.training:
            # During inference, do not drop out any nodes
            return inputs

        if self.weights is None:
            # Lazily construct weights to avoid having to specify input size on initialization.
            self._reset_weights(inputs.shape)
        elif inputs.shape != self.weights.shape:
            raise ValueError("Input size does not match ({} ≠ {})".format(inputs.shape, self.weights.shape))

        # Since the weights are applied to only a subset of the inputs, we must scale the
        # remaining inputs by the inverse of the dropout rate to maintain the same expected value.
        effective_drop_rate = round(inputs.shape[0] * self.rate) / inputs.shape[0]
        scale = 1 / (1 - effective_drop_rate)
        return inputs * self.weights * scale

    def backward(self, a: np.ndarray, da: np.ndarray, a_back: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backpropagate error through the layer.

        :param a: Outputs of this layer (shape: (1, num_neurons_in_layer))
        :param da: Error of this layer's output (shape: (1, num_neurons_in_layer))
        :param a_back: Outputs of previous layer (shape: (1, num_neurons_in_layer-1))
        :return: Change in weights, change in biases, and error for this layer
        """

        da = self.weights * da
        return 0, 0, da

    def start_batch(self):
        """
        Select a new set of dropped nodes for the next batch.

        Note: This should be called at the beginning of each batch.
        """

        if self.weights is None:
            return

        if not self.training:
            raise RuntimeError("Cannot start batch when not training")

        self._reset_weights(self.weights.shape)

    def _reset_weights(self, input_shape: tuple[int]):
        self.weights = self.rng.binomial(1, 1 - self.rate, input_shape)
