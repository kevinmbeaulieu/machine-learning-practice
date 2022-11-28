import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import layers, optimizers
from utilities.metrics import compute_metrics
from utilities.models.model import Model
from utilities.preprocessing.dataset import Dataset

class NeuralNetworkModel(Model):
    """
    Neural Network Model.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_epochs: int = 100,
        optimizer: optimizers.Optimizer = optimizers.MiniBatchGradientDescent(),
        verbose: bool = False
    ):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.layers = []
        self.verbose = verbose
        self.df_validation = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if not isinstance(self.layers[0], layers.Input):
            raise ValueError('First layer must be an input layer.')

        self.dataset = dataset
        if self.dataset.task == 'classification':
            self.classes = df['class'].unique()

        if self.verbose:
            print("Training for {} epochs".format(self.num_epochs))

        self._enable_dropout_training()
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
                self._disable_dropout_training()
                metrics_train.append(self._compute_epoch_metrics(df))
                if self.df_validation is not None:
                    metrics_validation.append(self._compute_epoch_metrics(self.df_validation))
                self._enable_dropout_training()

        self._disable_dropout_training()

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

        self.optimizer.start_batch()
        for layer in filter(lambda l: isinstance(l, layers.Dropout), self.layers):
            layer.start_batch()

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
            self.optimizer.update(layer, Δw[i], Δb[i])

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

        da_k = self._compute_loss_partial(y, a[-1])
        Δw, Δb = [], []
        for k in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[k]
            Δw_k, Δb_k, da_kmin1 = layer.backward(
                a=a[k],
                da=da_k,
                a_back=a[k - 1],
            )
            Δw.insert(0, Δw_k)
            Δb.insert(0, Δb_k)
            da_k = da_kmin1

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

    def _enable_dropout_training(self):
        for layer in filter(lambda l: isinstance(l, layers.Dropout), self.layers):
            layer.training = True

    def _disable_dropout_training(self):
        for layer in filter(lambda l: isinstance(l, layers.Dropout), self.layers):
            layer.training = False
