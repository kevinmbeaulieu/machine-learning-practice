import numpy as np
import pandas as pd

from .model import Model
from utilities.preprocessing.dataset import Dataset
from utilities import metrics

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
            val_accuracy = metrics.compute_metrics(
                actual=self.post_pruning_set['class'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['acc'],
            )[0]

            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: leaf_alternative.predict(row), axis=1)
            pruned_accuracy = metrics.compute_metrics(
                actual=self.post_pruning_set['class'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['acc'],
            )[0]

            return leaf_alternative if pruned_accuracy >= val_accuracy else node

        elif self.dataset.task == 'regression':
            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: node.predict(row), axis=1)
            val_mse = metrics.compute_metrics(
                actual=self.post_pruning_set['output'].to_numpy(),
                predicted=self.post_pruning_set['prediction'].to_numpy(),
                metrics=['mse'],
            )[0]

            self.post_pruning_set['prediction'] = self.post_pruning_set.apply(lambda row: leaf_alternative.predict(row), axis=1)
            pruned_mse = metrics.compute_metrics(
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


