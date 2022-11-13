import math
import numpy as np
import statistics
import pandas as pd

from .metrics import compute_metrics
from .preprocessing.dataset import Dataset

class Model:
    """
    Abstract class defining a common interface for machine learning models.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        """
        Must be overridden by subclass to train the model.

        :param df: pd.DataFrame, Training set
        :param dataset: Dataset, Metadata about the dataset being used for training
        """
        raise Exception("Model subclass expected to override train function, but didn't")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Must be overridden in subclass to predict class label/regression output for the test set.

        :param df: pd.DataFrame, Test set for which to predict values

        :return pd.Series
            For classification tasks, predicted class labels for the test set
            For regression tasks, predicted output values for the test set
        """
        raise Exception("Model subclass expected to override predict function, but didn't")

class NullModel(Model):
    """
    Null Model for testing machine learning pipeline.

    For classification tasks, uses the training set's plurality (most common) class label as prediction.
    For regression tasks, uses the average value of the training set's output attribute as prediction.
    """

    def __init__(self):
        self.predict_value = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        if dataset.task == 'classification':
            self.predict_value = df['class'].mode()[0]
        elif dataset.task == 'regression':
            self.predict_value = df['output'].mean()
        else:
            raise Exception("Failed to train null model for dataset with unrecognized task {}".format(dataset.task))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.predict_value is None:
            raise Exception("Failed to predict values with null model before model was trained")

        return pd.Series([self.predict_value]).repeat(df.shape[0])


class _KNNModel(Model):
    """
    Abstract parent class for K-Nearest Neighbors models.
    """
    def __init__(self, k: int):
        """
        :param k: int, Number of nearest neighbors to use for prediction
        """

        self.k = k
        self.df_train = None
        self.dataset = None

    def train(self, df: pd.DataFrame, dataset: Dataset):
        self.df_train = df
        self.dataset = dataset

    def _distance(self, df_left: pd.DataFrame, left: pd.Series, df_right: pd.DataFrame, right: pd.Series) -> float:
        """
        Calculates the distance between two data points.

        :param df_left: pd.DataFrame, Data frame containing the left data point
        :param left: pd.Series, Data point
        :param df_right: pd.DataFrame, Data frame containing the right data point
        :param right: pd.Series, Data point

        :return float, Distance between the two data points
        """

        output_col = 'class' if self.dataset.task == 'classification' else 'output'
        left = left.copy().drop(output_col, errors='ignore')
        right = right.copy().drop(output_col, errors='ignore')

        if left.shape != right.shape:
            raise Exception("Failed to calculate distance between vectors with different shapes {} != {}.\n\n{}\n\n{}".format(left.shape, right.shape, left, right))

        distance = 0
        for col in df_left.columns:
            if col == output_col:
                continue
            if col in self.dataset.nominal_cols:
                distance += self._vdm_distance_sq(df_left, left[col], df_right, right[col])
            else:
                distance += self._euclidean_distance_sq(left[col], right[col])
        return distance ** 0.5

    def _euclidean_distance_sq(self, left: float, right: float) -> float:
        return (left - right) ** 2

    def _vdm_distance_sq(self, df_left: pd.DataFrame, left: any, df_right: pd.DataFrame, right: any) -> float:
        """
        Calculates the square of the VDM distance between two nominal values.

        :param df_left: pd.DataFrame, Data frame containing the left value
        :param left: any, Left value
        :param df_right: pd.DataFrame, Data frame containing the right value
        :param right: any, Right value

        :return float, squared VDM distance between the two nominal values
        """
        distance = 0
        vdm_exp = 2

        # Number of rows in df_left where value of k'th attribute is left
        C_left = df_left.loc[df_left.iloc[:, self.k] == left].shape[0]
        # Number of rows in df_right where value of k'th attribute is right
        C_right = df_right.loc[df_right.iloc[:, self.k] == right].shape[0]

        for c in self.classes:
            # Number of rows in df_left where value of k'th attribute is left and class label is c
            C_left_a = df_left.loc[(df_left['class'] == c) & (df_left.iloc[:, self.k] == left)].shape[0]
            # Number of rows in df_right where value of k'th attribute is right and class label is c
            C_right_a = df_right.loc[(df_right['class'] == c) & (df_right.iloc[:, self.k] == right)].shape[0]

            distance += abs(C_left_a/C_left - C_right_a/C_right) ** vdm_exp

        return distance

class KNNModel(_KNNModel):
    """
    K-Nearest Neighbors Model.
    """

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        if dataset.task == 'classification':
            self.classes = self.df_train['class'].unique()
        elif dataset.task == 'regression':
            self.γ = 1 / self.df_train['output'].std()
        else:
            raise Exception("Failed to train KNN model for dataset with unrecognized task {}".format(dataset.task))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.df_train is None:
            raise Exception("Failed to predict values with KNN model before model was trained")

        if self.dataset.task == 'classification':
            return df.apply(lambda row: self._predict_class(df, row), axis=1)
        elif self.dataset.task == 'regression':
            return df.apply(lambda row: self._predict_output(df, row), axis=1)
        else:
            raise Exception("Failed to predict values with KNN model for dataset with unrecognized task {}".format(self.dataset.task))

    def _predict_class(self, df: pd.DataFrame, x: pd.Series) -> str:
        """
        Predict class based on mode of k nearest neighbors.
        """
        distances = [self._distance(df, x, self.df_train, x_train) for _, x_train in self.df_train.iterrows()]
        k_nearest_neighbors = self.df_train.iloc[np.argsort(distances)[:self.k]]
        return k_nearest_neighbors['class'].mode()[0]

    def _predict_output(self, df: pd.DataFrame, x: pd.Series) -> float:
        """
        Predict regression output using KNN smoother, as defined by:
        ghat(x) = sum_t (K(x, x_t) * y_t) / sum_t (K(x, x_t))
        where
            x_t = t'th training example
            y_t = t'th training example's output
            K(x, x_t) = exp[-γ ||x - x_t||_2]
            ||a - b||_2 = Euclidean distance between a and b
        """
        distances = [self._distance(df, x, self.df_train, x_train) for _, x_train in self.df_train.iterrows()]
        k_nearest_distances = np.sort(distances)[:self.k]
        k_nearest_neighbors = self.df_train.iloc[np.argsort(distances)[:self.k]]
        numerator = np.sum(
            [np.exp(-self.γ * d) * y for d, y in zip(k_nearest_distances, k_nearest_neighbors['output'])]
        )
        denominator = np.sum([np.exp(-self.γ * d) for d in k_nearest_distances])
        return numerator / denominator


class EditedKNNModel(KNNModel):
    """
    Edited K-Nearest Neighbors Model.
    """
    def __init__(self, k: int, df_val: pd.DataFrame, ε: float = 0, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param df_val: pd.DataFrame, Validation set
        :param ε: float, Maximum allowed error rate on validation set
        :param max_iterations: int, Maximum number of iterations to run the edited KNN algorithm
        """
        super().__init__(k)
        self.df_val = df_val
        self.ε = ε
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._edit_training_set()
        self.df_train.reset_index(drop=True, inplace=True)

    def _edit_training_set(self):
        """
        Edited KNN algorithm:
            1. Consider each data point individually.
            2. For each data point, use its single nearest neighbor to make a prediction.
            3. If the prediction is correct, mark the data point for deletion.
            4. Stop editing once performance on the validation set starts to degrade.
        """

        iterations = 0
        prev_val_accuracy = 0 # For classification
        prev_val_mse = np.inf # For regression
        prev_df_train = self.df_train.copy()
        while iterations < self.max_iterations:
            # (1–2) Predict each training example based on its single nearest neighbor
            # (*don't forget to exclude itself when looking for its nearest neighbor 😄*)
            self.df_train['prediction'] = self.df_train.apply(lambda row: self._predict_1nn(row), axis=1)

            # (3) Remove all rows in the training set that are correctly classified by
            # their single nearest neighbor.
            if self.dataset.task == 'classification':
                incorrectly_classified = self.df_train.loc[
                    self.df_train['prediction'] != self.df_train['class']
                ]
            elif self.dataset.task == 'regression':
                incorrectly_classified = self.df_train.loc[
                    np.abs(self.df_train['prediction'] - self.df_train['output']) > self.ε
                ]
            else:
                raise Exception("Failed to edit training set for dataset with unrecognized task {}".format(self.dataset.task))

            if incorrectly_classified.shape[0] == 0:
                # If this iteration would remove all remaining rows of the training set,
                # revert to the previous training set and stop editing.
                self.df_train = prev_df_train
                return

            self.df_train = incorrectly_classified.drop('prediction', axis=1)

            # (4) Stop editing if performance on the validation set starts to degrade.
            self.df_val['prediction'] = self.predict(self.df_val)
            if self.dataset.task == 'classification':
                val_accuracy = compute_metrics(
                    actual=self.df_val['class'].to_numpy(),
                    predicted=self.df_val['prediction'].to_numpy(),
                    metrics=['acc'],
                )[0]
                if val_accuracy < prev_val_accuracy:
                    # If accuracy has decreased, revert to the previous training set and stop editing.
                    self.df_train = prev_df_train
                    return
            elif self.dataset.task == 'regression':
                val_mse = compute_metrics(
                    actual=self.df_val['output'].to_numpy(),
                    predicted=self.df_val['prediction'].to_numpy(),
                    metrics=['mse'],
                )[0]
                if val_mse > prev_val_mse:
                    # If MSE has increased, revert to the previous training set and stop editing.
                    self.df_train = prev_df_train
                    return
            else:
                raise Exception("Failed to edit training set for dataset with unrecognized task {}".format(self.dataset.task))

            self.df_val.drop('prediction', axis=1, inplace=True)
            prev_df_train = self.df_train.copy()
            iterations += 1

    def _predict_1nn(self, x: pd.Series) -> any:
        """
        Predict class or output based on single nearest neighbor.

        :param x: pd.Series, Example to predict
        :return str|float, Predicted class or output
        """
        index = x.name
        df_x = self.df_train[self.df_train.index == index]
        df_train = self.df_train.drop(index).drop('prediction', axis=1, errors='ignore')

        validation_model = KNNModel(1)
        validation_model.train(df_train, self.dataset)
        return validation_model.predict(df_x).iloc[0]

class CondensedKNNModel(KNNModel):
    """
    Condensed K-Nearest Neighbors Model.
    """
    def __init__(self, k: int, ε: float = 0, max_iterations: int = 5):
        """
        :param k: int, Number of nearest neighbors to consider
        :param ε: float, Tolerance for determining if a data point is correctly predicted for regression
        :param max_iterations: int, Maximum number of iterations to run the condensed KNN algorithm
        """
        super().__init__(k)
        self.ε = ε
        self.max_iterations = max_iterations

    def train(self, df: pd.DataFrame, dataset: Dataset):
        super().train(df, dataset)

        self._condense_training_set()
        self.df_train.reset_index(drop=True, inplace=True)

    def _condense_training_set(self):
        """
        Condensed KNN algorithm:
            1. Add the first data point from the training set into the condensed set.
            2. Consider the remaining data points in the training set individually.
            3. For each data point, attempt to predict its value using the condensed set via 1-nn.
            4. If the prediction is incorrect, add the data point to the condensed set. Otherwise, move on to the next data point.
            5. Make multiple passes through the data in the training set that has not been added until the condensed set stops changing.
        """

        iterations = 0

        # (1) Add first training example to condensed set
        condensed_set = self.df_train.iloc[[0]]
        self.df_train.drop(0, inplace=True)

        # (5) Make multiple passes through the data in the training set that has not been added until the condensed set stops changing.
        prev_condensed_set_size = condensed_set.shape[0]
        while iterations < self.max_iterations and self.df_train.shape[0] > 0:
            # (2) Consider the remaining data points in the training set individually.
            for i, x in self.df_train.iterrows():
                prediction = self._predict_1nn(x, condensed_set)
                is_correct = False
                if self.dataset.task == 'classification':
                    is_correct = prediction == x['class']
                elif self.dataset.task == 'regression':
                    is_correct = np.abs(prediction - x['output']) <= self.ε
                else:
                    raise Exception("Failed to condense training set for dataset with unrecognized task {}".format(self.dataset.task))

                if not is_correct:
                    # (4) If the prediction is incorrect, add the data point to the condensed set.
                    condensed_set = pd.concat([condensed_set, self.df_train.loc[[i]]])

            if condensed_set.shape[0] == prev_condensed_set_size:
                # (5) If the condensed set stops changing, stop condensing.
                break

            self.df_train = self.df_train[~self.df_train.index.isin(condensed_set.index)]
            prev_condensed_set_size = condensed_set.shape[0]
            iterations += 1
        self.df_train = condensed_set.copy()

    def _predict_1nn(self, x: pd.Series, condensed_set: pd.DataFrame) -> any:
        """
        Predict class or output based on single nearest neighbor in condensed set.

        :param x: pd.Series, Example to predict
        :param condensed_set: pd.DataFrame, Condensed training set

        :return str|float, Predicted class or output
        """
        index = x.name
        df_x = self.df_train[self.df_train.index == index]
        validation_model = KNNModel(1)
        validation_model.train(condensed_set, self.dataset)
        return validation_model.predict(df_x).iloc[0]

class DecisionTreeModel(Model):
    """
    Decision Tree Model.
    """
    def __init__(
        self,
        pruning_strategy = None,
        leaf_size: float = 0,
        post_pruning_set: pd.DataFrame = None,
        ε: float = 0.001
    ):
        """
        :param pruning_strategy: str|None, Pruning strategy to use (None, 'pre-prune', 'post-prune')
        :param leaf_size: float, For pre-pruning or regression, minimum number of examples in a
            leaf node as a percentage of the total number of training examples
        :param post_pruning_set: pd.DataFrame, For post-pruning, validation set to use
        :param ε: float, Tolerance used for determining whether a node is pure
        """
        self.pruning_strategy = pruning_strategy
        self.leaf_size = leaf_size
        self.post_pruning_set = post_pruning_set
        self.ε = ε

    def train(self, df: pd.DataFrame, dataset: Dataset):
        self.dataset = dataset
        self.df_train = df

        self.root = self._build_node(self.df_train, self._select_attribute(self.df_train))

        if self.pruning_strategy == 'post-prune':
            self._post_prune_subtree(self.root)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda x: self.root.predict(x), axis=1)

    def __str__(self):
        return str(self.root)

    def _select_attribute(self, examples: pd.DataFrame) -> str:
        """
        Select best attribute to split on.

        :param examples: pd.DataFrame, Examples to split
        :return str|None, Best attribute to split on, or None if entropy is already below threshold.
        """

        entropy = self._entropy(examples)
        if entropy <= self.ε:
            return None

        gains = {}
        for attribute in self.dataset.col_names:
            if attribute in self.dataset.ignore_cols or attribute in ['class', 'output']:
                continue

            gains[attribute] = self._information_gain(entropy, attribute, examples)
        result = max(gains, key=gains.get)
        return result

    def _information_gain(self, entropy: float, attribute: str, examples: pd.DataFrame) -> float:
        """
        Calculate information gain from splitting on an attribute.

        Gain(S, A) = Entropy(S) - Sum( |Sv| / |S| * Entropy(Sv) )
        S = examples
        A = attribute
        Sv = examples with value v for attribute A

        :param entropy: float, Entropy of examples (Entropy(S))
        :param attribute: str, Attribute to split on (A)
        :param examples: pd.DataFrame, Examples to split (S)

        :return float, Information gain
        """
        weighted_child_entropies = []

        if attribute in self.dataset.nominal_cols:
            attribute_values = examples[attribute].unique()
            for v in attribute_values:
                val_subset = examples[examples[attribute] == v]
                weighted_entropy = len(val_subset) / len(examples) * self._entropy(val_subset)
                weighted_child_entropies.append(weighted_entropy)
        else:
            threshold = self._find_best_threshold(attribute, examples)
            if threshold is None:
                return entropy

            left_subset = examples[examples[attribute] <= threshold]
            right_subset = examples[examples[attribute] > threshold]

            weighted_child_entropies = [
                len(left_subset) / len(examples) * self._entropy(left_subset),
                len(right_subset) / len(examples) * self._entropy(right_subset)
            ]

        return entropy - sum(weighted_child_entropies)

    def _entropy(self, df: pd.DataFrame) -> float:
        """
        Calculate entropy of examples.

        :param df: pd.DataFrame, Examples to calculate entropy of

        :return float, Entropy of examples
        """
        if self.dataset.task == 'classification':
            class_counts = df['class'].value_counts()
            entropy_term = lambda Ni: -(Ni / df.shape[0]) * np.log2(Ni / df.shape[0])
            entropy = class_counts.apply(lambda Ni: -(Ni / df.shape[0]) * np.log2(Ni / df.shape[0])).sum()
            return entropy
        elif self.dataset.task == 'regression':
            return np.var(df['output'])
        else:
            raise Exception("Failed to calculate entropy for dataset with unrecognized task {}".format(self.dataset.task))

    def _find_best_threshold(self, attribute: str, examples: pd.DataFrame) -> float:
        """
        Find the best threshold to split on for this attribute.

        :param attribute: str, Attribute to split on (must be numeric)
        :param examples: pd.DataFrame, Examples to split

        :return float|None, Best threshold
        """
        sorted_values = examples[attribute].sort_values().drop_duplicates()
        if sorted_values.shape[0] <= 1:
            return None

        candidate_thresholds = []
        for i in range(sorted_values.shape[0] - 1):
            midpoint = (sorted_values.iloc[i] + sorted_values.iloc[i + 1]) / 2
            candidate_thresholds.append(midpoint)

        candidate_entropies = []
        for threshold in candidate_thresholds:
            left_subset = examples[examples[attribute] <= threshold]
            right_subset = examples[examples[attribute] > threshold]

            left_entropy = self._entropy(left_subset)
            right_entropy = self._entropy(right_subset)

            left_weight = left_subset.shape[0] / examples.shape[0]
            right_weight = 1 - left_weight

            total_entropy = left_weight * left_entropy + right_weight * right_entropy
            candidate_entropies.append(total_entropy)

        return candidate_thresholds[np.argmin(candidate_entropies)]

    def _build_node(self, examples: pd.DataFrame, attribute: str=None):
        """
        Recursively build subtree rooted at a new node which splits the examples on
        the specified attribute. If no attribute is specified, a leaf node is returned.

        :param examples: pd.DataFrame, Examples to split
        :param attribute: str|None, Attribute to split on

        :return Node, Root of subtree
        """
        if attribute is None or ((self.pruning_strategy == 'pre-prune' or self.dataset.task == 'regression') and examples.shape[0] < self.leaf_size * self.df_train.shape[0]):
            return self.LeafNode(examples, self.dataset)

        if attribute in self.dataset.nominal_cols:
            node = NominalNode(attribute, examples, self.dataset)

            attribute_values = examples[attribute].unique()
            node.children = {}
            for v in attribute_values:
                val_subset = examples[examples[attribute] == v]
                node.children[v] = self._build_node(val_subset, self._select_attribute(val_subset))
                node.children[v].parent = node

            return node
        else:
            threshold = self._find_best_threshold(attribute, examples)
            if threshold is None:
                return self.LeafNode(examples, self.dataset)

            left_subset = examples[examples[attribute] <= threshold]
            right_subset = examples[examples[attribute] > threshold]

            node = self.NumericalNode(attribute, threshold, examples, self.dataset)
            node.left_child = self._build_node(left_subset, self._select_attribute(left_subset))
            node.left_child.parent = node
            node.right_child = self._build_node(right_subset, self._select_attribute(right_subset))
            node.right_child.parent = node

            return node

    def _post_prune_subtree(self, node):
        if isinstance(node, self.LeafNode):
            return

        pruned = self._get_pruned_subtree(node)
        if pruned is not None:
            self._replace_node(node, pruned)
            return

        if isinstance(node, self.NominalNode):
            for child in node.children.values():
                self._post_prune_subtree(child)
        elif isinstance(node, self.NumericalNode):
            self._post_prune_subtree(node.left_child)
            self._post_prune_subtree(node.right_child)

    def _replace_node(self, node, replacement):
        if node.parent is None:
            self.root = replacement
        elif isinstance(node.parent, self.NominalNode):
            node_parent_key = [k for k, v in node.parent.children.items() if v == node][0]
            node.parent.children[node_parent_key] = replacement
        elif isinstance(node.parent, self.NumericalNode):
            if node.parent.left_child == node:
                node.parent.left_child = replacement
            else:
                node.parent.right_child = replacement

    def _get_pruned_subtree(self, node):
        """
        Recursively post-prune tree using validation set.

        :param node: Node, Root of subtree

        :return Node|None, Leaf node to replace subtree, or None if subtree should be kept
        """
        if isinstance(node, self.LeafNode):
            return node

        leaf_alternative = self.LeafNode(node.examples, self.dataset)
        if self.dataset.task == 'classification':
            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: node.predict(row), axis=1)
            val_accuracy = compute_metrics(
                actual=self.post_pruning_set['class'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['acc'],
            )[0]

            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: leaf_alternative.predict(row), axis=1)
            pruned_accuracy = compute_metrics(
                actual=self.post_pruning_set['class'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['acc'],
            )[0]

            return leaf_alternative if pruned_accuracy >= val_accuracy else node

        elif self.dataset.task == 'regression':
            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: node.predict(row), axis=1)
            val_mse = compute_metrics(
                actual=self.post_pruning_set['output'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['mse'],
            )[0]

            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: leaf_alternative.predict(row), axis=1)
            pruned_mse = compute_metrics(
                actual=self.post_pruning_set['output'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['mse'],
            )[0]

            return leaf_alternative if pruned_mse <= val_mse else node

    class Node:
        """
        Abstract node class for decision tree.
        """
        def __init__(self, examples: pd.DataFrame, dataset: Dataset):
            """
            :param examples: pd.DataFrame, Training examples reaching this node
            :param dataset: Dataset, Dataset these examples belong to
            """
            self.examples = examples
            self.dataset = dataset
            self.parent = None

        def predict(self, x: pd.Series) -> any:
            """
            Predict class or output based on subtree rooted at this node.

            :param x: pd.Series, Example to predict

            :return str|float, Predicted class or output
            """
            raise Exception("Must be implemented by subclass")

    class NominalNode(Node):
        """
        Node splitting on a nominal attribute in a decision tree.
        """
        def __init__(self, attribute: str, examples: pd.DataFrame, dataset: Dataset):
            """
            :param attribute: str, Attribute to split on
            :param examples: pd.DataFrame, Training examples reaching this node
            :param dataset: Dataset, Dataset these examples belong to
            """
            self.attribute = attribute
            self.children = {}

            super().__init__(examples, dataset)

        def predict(self, x: pd.Series) -> any:
            return self.children[x[self.attribute]].predict(x)

        def __str__(self):
            description = "NominalNode({})".format(self.attribute)
            for v, child in self.children.items():
                child_description = str(child).replace("\n", "\n\t")
                description += "\n\t{}: {}".format(v, child_description)
            return description

    class NumericalNode(Node):
        """
        Node splitting on a numerical attribute in a decision tree.
        """
        def __init__(self, attribute: str, threshold: float, examples: pd.DataFrame, dataset: Dataset):
            """
            :param attribute: str, Attribute to split on
            :param threshold: float, Threshold to split on
            :param examples: pd.DataFrame, Training examples reaching this node
            :param dataset: Dataset, Dataset these examples belong to
            """
            self.attribute = attribute
            self.threshold = threshold
            self.left_child = None
            self.right_child = None

            super().__init__(examples, dataset)

        def predict(self, x: pd.Series) -> any:
            child = self.left_child if x[self.attribute] <= self.threshold else self.right_child
            return child.predict(x)

        def __str__(self):
            description = "NumericalNode({}<={})".format(self.attribute, self.threshold)
            if self.left_child is not None:
                description += "\n\tLeft: {}".format(str(self.left_child).replace("\n", "\n\t"))
            if self.right_child is not None:
                description += "\n\tRight: {}".format(str(self.right_child).replace("\n", "\n\t"))
            return description

    class LeafNode(Node):
        """
        Leaf node in a decision tree.
        """
        def __init__(self, examples: pd.DataFrame, dataset: Dataset):
            """
            :param examples: pd.DataFrame, Examples in the leaf node
            :param dataset: Dataset, Dataset the examples belong to
            """
            super().__init__(examples, dataset)

            if self.dataset.task == 'classification':
                # Predict the most common class amongst examples
                self.predict_value = self.examples['class'].value_counts().index[0]
            elif self.dataset.task == 'regression':
                # Predict the average output amongst examples
                self.predict_value = self.examples['output'].mean()

        def predict(self, x: pd.Series) -> any:
            return self.predict_value

        def __str__(self):
            description = "LeafNode({})".format(self.predict_value)
            output_col = 'class' if self.dataset.task == 'classification' else 'output'
            description += "\n\tExamples: {}".format(self.examples[output_col].values)
            return description

