import numpy as np

from . import layers

class Optimizer:
    """
    Base class for optimizers.
    """

    def __init__(self):
        pass

    def start_batch(self):
        """
        Start a new iteration.
        """

        pass

    def update(self, layer: layers.Layer, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights and biases of a layer.

        :param layer: Layer to update.
        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """

        layer.update_weights(Δw, Δb)

class MiniBatchGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    def update(self, layer: layers.Layer, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights and biases of a layer according to a fixed learning rate.

        :param layer: Layer to update.
        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """

        super().update(layer, self.learning_rate * Δw, self.learning_rate * Δb)

class Adam(Optimizer):
    """
    Adam optimizer.

    This optimizer is based on the paper "Adam: A Method for Stochastic Optimization" by
    Diederik P. Kingma and Jimmy Ba (https://arxiv.org/abs/1412.6980).
    """

    def __init__(self, learning_rate: float = 0.0001, β1: float = 0.9, β2: float = 0.999, ε: float = 1e-8):
        """
        :param learning_rate: Learning rate η.
        :param β1: Exponential decay rate for the first moment estimates.
        :param β2: Exponential decay rate for the second moment estimates.
        :param ε: Small constant to avoid division by zero.
        """

        self.learning_rate = learning_rate
        self.β1 = β1
        self.β2 = β2
        self.ε = ε
        self.s_i = {} # First moment estimates for weights/biases of each layer
        self.r_i = {} # Second moment estimates for weights/biases of each layer
        self.t = 0 # Time step

    def start_batch(self):
        """
        Start a new iteration.
        """

        self.t += 1

    def update(self, layer: layers.Layer, Δw: np.ndarray, Δb: np.ndarray):
        """
        Update the weights and biases of a layer according to the Adam optimization algorithm.

        :param layer: Layer to update.
        :param Δw: Change in weights.
        :param Δb: Change in biases.
        """

        s_i = self.s_i.get(layer, (0, 0))
        r_i = self.r_i.get(layer, (0, 0))
        Δw, s_i_w, r_i_w = self._update(Δw, s_i[0], r_i[0])
        Δb, s_i_b, r_i_b = self._update(Δb, s_i[1], r_i[1])

        self.s_i[layer.id] = (s_i_w, s_i_b)
        self.r_i[layer.id] = (r_i_w, r_i_b)

        super().update(layer, Δw, Δb)

    def _update(self, Δ: np.ndarray, s_i: np.double, r_i: np.double) -> tuple[np.ndarray, np.double, np.double]:
        """
        Update the weights or biases of a layer according to the Adam optimization algorithm.

        :param Δ_: Change in weights or biases.
        :param s_i: Exponential decay rate for the first moment estimates.
        :param r_i: Exponential decay rate for the second moment estimates.
        """

        s_i_new = self.β1 * s_i + (1 - self.β1) * Δ
        r_i_new = self.β2 * r_i + (1 - self.β2) * Δ ** 2

        s_i_hat = s_i_new / (1 - self.β1 ** self.t)
        r_i_hat = r_i_new / (1 - self.β2 ** self.t)

        Δ_new = self.learning_rate * s_i_hat / (np.sqrt(r_i_hat) + self.ε)

        return Δ_new, s_i_new, r_i_new
