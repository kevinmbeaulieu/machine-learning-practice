import math
import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
from sklearn import feature_selection as sklearn_feature_selection

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    metrics: list[str],
    positive_class: str=None,
    negative_class: str=None,
    use_sklearn: bool=False,
) -> list:
    """
    Computes the specified evluation metric(s) based on the provided ground truth and predicted values.

    :param actual: np.ndarray, Ground truth values
    :param predicted: np.ndarray, Predicted values
    :param metrics: list[str], List of evaluation metrics to compute. Supported metrics are:
        Classification tasks:
            'acc': Accuracy
            'precision': Precision
            'recall': Recall
            'f1': F1 score
        Regression tasks:
            'mse': Mean squared error
            'mae': Mean absolute error
            'r2': R^2 coefficient of determination
            'pearson': Pearson correlation coefficient
    :param positive_class: str|None, Label used for the positive class of a binary classification task.
    :param negative_class: str|None, Label used for the negative class of a binary classification task.
    :param use_sklearn: bool, If True, use scikit-learn to compute metrics instead of manual calculations.

    :return list, List of metric values in the same order as the input list of metric names.
    """
    assert actual.size == predicted.size
    # actual = actual.reset_index(drop=True)
    # predicted = predicted.reset_index(drop=True)

    if use_sklearn:
        return list(
            map(
                lambda metric: _sklearn_compute_metric(actual, predicted, metric, positive_class),
                metrics
            )
        )

    confusion_matrix = None
    if len(set(metrics) & set(['precision', 'recall', 'f1'])) > 0:
        confusion_matrix = _compute_confusion_matrix(actual, predicted, positive_class, negative_class)
    return list(
        map(
            lambda metric: _compute_metric(actual, predicted, metric, confusion_matrix), 
            metrics
        )
    )


def _compute_metric(actual: np.ndarray, predicted: np.ndarray, metric: str, confusion_matrix: np.ndarray=None) -> float:
    if metric == 'acc':
        return _compute_accuracy(actual, predicted)
    elif metric == 'precision':
        return _compute_precision(confusion_matrix)
    elif metric == 'recall':
        return _compute_recall(confusion_matrix)
    elif metric == 'f1':
        return _compute_f1(confusion_matrix)
    elif metric == 'mse':
        return _compute_mse(actual, predicted)
    elif metric == 'mae':
        return _compute_mae(actual, predicted)
    elif metric == 'r2':
        return _compute_r2(actual, predicted)
    elif metric == 'pearson':
        return _compute_pearson(actual, predicted)
    else:
        raise Exception("Failed to compute unrecognized evaluation metric {}".format(metric))

def _compute_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    n_correct = 0
    for a, p in zip(actual, predicted):
        n_correct += 1 if a == p else 0
    return n_correct / predicted.size

def _compute_confusion_matrix(actual: np.ndarray, predicted: np.ndarray, positive_class: str, negative_class: str) -> np.array:
    result = np.empty((2, 2))
    n_true_positives, n_false_positives, n_true_negatives, n_false_negatives = 0, 0, 0, 0
    for i in range(predicted.size):
        if predicted[i] == positive_class and actual[i] == positive_class:
            n_true_positives += 1
        elif predicted[i] == positive_class:
            n_false_positives += 1
        elif actual[i] == negative_class:
            n_true_negatives += 1
        else:
            n_false_negatives += 1
    result[0, 0] = n_true_positives
    result[0, 1] = n_false_negatives
    result[1, 0] = n_false_positives
    result[1, 1] = n_true_negatives
    return result

def _compute_precision(confusion_matrix: np.ndarray) -> float:
    n_pred_positive = confusion_matrix[:, 0].sum()
    
    if n_pred_positive == 0:
        return math.nan
    return confusion_matrix[0, 0] / n_pred_positive

def _compute_recall(confusion_matrix: np.ndarray) -> float:
    n_true_positive = confusion_matrix[0, 0]
    n_actual_positive = confusion_matrix[0, :].sum()
    
    if n_actual_positive == 0:
        return math.nan
    return n_true_positive / n_actual_positive

def _compute_f1(confusion_matrix: np.ndarray) -> float:
    # Formula source: https://en.wikipedia.org/wiki/F-score#Definition
    precision = _compute_precision(confusion_matrix)
    recall = _compute_recall(confusion_matrix)

    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        return math.nan
    return 2 * precision * recall / (precision + recall)

def _compute_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return ((predicted - actual) ** 2).sum() / predicted.size

def _compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.abs(predicted - actual).sum() / predicted.size

def _compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    # Formula source: https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
    ss_residual = ((actual - predicted) ** 2).sum()
    mean_actual = actual.mean()
    ss_total = ((actual - mean_actual) ** 2).sum()
    
    if ss_total == 0:
        return math.nan
    return 1 - (ss_residual / ss_total)

def _compute_pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.corrcoef(actual, predicted)


def _sklearn_compute_metric(actual, predicted, metric: str, positive_class: str=None) -> float:
    if metric == 'acc':
        return _sklearn_compute_accuracy(actual, predicted)
    elif metric == 'precision':
        return _sklearn_compute_precision(actual, predicted, positive_class)
    elif metric == 'recall':
        return _sklearn_compute_recall(actual, predicted, positive_class)
    elif metric == 'f1':
        return _sklearn_compute_f1(actual, predicted, positive_class)
    elif metric == 'mse':
        return _sklearn_compute_mse(actual, predicted)
    elif metric == 'mae':
        return _sklearn_compute_mae(actual, predicted)
    elif metric == 'r2':
        return _sklearn_compute_r2(actual, predicted)
    elif metric == 'pearson':
        return _sklearn_compute_pearson(actual, predicted)
    else:
        raise Exception("Failed to compute unrecognized evaluation metric {}".format(metric))

def _sklearn_compute_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sklearn_metrics.accuracy_score(actual.reshape(-1, 1), predicted.reshape(-1, 1))

def _sklearn_compute_precision(actual: np.ndarray, predicted: np.ndarray, positive_class: str) -> float:
    return sklearn_metrics.precision_score(actual, predicted, pos_label=positive_class)

def _sklearn_compute_recall(actual: np.ndarray, predicted: np.ndarray, positive_class: str) -> float:
    return sklearn_metrics.recall_score(actual, predicted, pos_label=positive_class)

def _sklearn_compute_f1(actual: np.ndarray, predicted: np.ndarray, positive_class: str) -> float:
    return sklearn_metrics.f1_score(actual, predicted, pos_label=positive_class)

def _sklearn_compute_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sklearn_metrics.mean_squared_error(actual, predicted)

def _sklearn_compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sklearn_metrics.mean_absolute_error(actual, predicted)

def _sklearn_compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sklearn_metrics.r2_score(actual, predicted)

def _sklearn_compute_pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    return sklearn_feature_selection.r_regression(actual, predicted)