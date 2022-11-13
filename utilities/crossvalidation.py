from collections import defaultdict
from scipy import stats
from typing import Callable
import pandas as pd
import random

from .models.model import Model
from .preprocessing.dataset import Dataset
from .preprocessing import featurescaling

def split_for_cross_validation(
    df: pd.DataFrame,
    strategy: str,
    k: int,
    dataset: Dataset,
    include_validation: bool=True
) -> list[pd.DataFrame]:
    """
    Partition the DataFrame for cross-validation.

    :param df: pd.DataFrame, Dataset to partition
    :param strategy: str ('k'|'kx2'), Cross validation strategy to use
        'k' for k-fold cross validation
        'kx2' for (k x 2)-fold cross validation
    :param k: int, k to use for k-fold/(k x 2)-fold cross validation
    :param dataset: Dataset, Metadata about the dataset being processed
    
    :return list[pd.DataFrame]
        k-fold cross validation: First element is a validation set (20% of original
            dataset). Remaining elements are k equally sized partitions of the
            remaining 80% of the original dataset.
        (k x 2)-fold cross validation: First element is a validation set (20% of
            original dataset). Remaining elements are pairs of equally sized subsets
            of the remaining 80% of the original dataset (e.g., list[1] and list[2]
            make up one pair, list[3] and list[4] make up another pair, and so on
            up to list[2k - 1] and list[2k]).
    """

    stratify_by = 'class' if dataset.task == 'classification' else None
    if include_validation:
        df_train_test, df_val = _split(df, frac=[0.8, 0.2], stratify_by=stratify_by)
        result = [df_val]
    else:
        df_train_test = df
        result = []

    if strategy == 'k':
        result.extend(_split(df_train_test, k=k, stratify_by=stratify_by))
    elif strategy == 'kx2':
        for _ in range(k):
            result.extend(_split(df_train_test, k=2, stratify_by=stratify_by))
    else:
        raise Exception("Failed to cross validate with unrecognized strategy {}".format(strategy))

    return result

def cross_validate(
    kx2_fold_dfs: dict[str, list[pd.DataFrame]], 
    datasets: dict[str, Dataset], 
    model_factory: Callable[[], Model],
    verbose: bool=True
):
    """
    Run 5x2-fold cross validation on each of a collection of data frames, using a specified model type.
    Evaluation metrics are printed to stdout.

    :param kx2_fold_dfs: dict[str, list[pd.DataFrame]], Dictionary mapping dataset names to
        the output of split_for_cross_validation. Any preprocessing of data should have already
        been performed before calling this function.
    :param datasets: dict[str, Dataset], Dictionary mapping dataset names to metadata about the
        corresponding dataset.
    :param model_factory: Callable[[], Model], Factory method to initialize a model to train/test.

    :return dict[str, dict[str, float]], Dictionary mapping dataset name to a dictionary of evaluation metrics
    """

    result_metrics = defaultdict(lambda: {})

    for name in kx2_fold_dfs.keys():
        if verbose:
            print("Starting cross validation for {} dataset...".format(name))

        dataset = datasets[name]
        dfs = kx2_fold_dfs[name].copy()

        metrics = defaultdict(lambda: [])
        for i in range(int(len(dfs) / 2)):
            if verbose:
                print("  Performing iteration {} of cross-validation...".format(i))

            df1 = dfs[2 * i]
            df2 = dfs[2 * i + 1]

            train_test_pairs = [(df1, df2), (df2, df1)]
            for j in range(len(train_test_pairs)):
                if verbose:
                    print("    Performing iteration {}x{}...".format(i, j))
                df_train, df_test = train_test_pairs[j]

                if verbose:
                    print("      Standardizing data...")
                df_train, df_test = featurescaling.standardize_attributes(df_train, df_test, dataset)

                if verbose:
                    print("      Normalizing data...")
                df_train, df_test = featurescaling.normalize_attributes(df_train, df_test, dataset)

                model = model_factory()
                
                if verbose:
                    print("      Training model...")
                model.train(df_train, dataset)
                
                if verbose:
                    print("      Predicting test data...")
                predictions = model.predict(df_test)

                if dataset.task == 'classification':
                    actual = df_test['class']
                elif dataset.task == 'regression':
                    actual = df_test['output']

                if verbose:
                    print("      Computing evaluation metrics...")
                metric_values = compute_metrics(actual, predictions, dataset.metrics, dataset.positive_class, dataset.negative_class)
                for metric_name, metric_value in zip(dataset.metrics, metric_values):
                    metrics[metric_name].append(metric_value)
        
        print("Results for {} dataset".format(name))
        for metric_name in metrics.keys():
            metrics[metric_name] = stats.tmean(metrics[metric_name])
            print("    {}: {}".format(metric_name, metrics[metric_name]))
        
        result_metrics[name] = metrics
    
    return result_metrics

def _split(df: pd.DataFrame, k: int=None, frac: list[float]=None, stratify_by: str=None) -> list[pd.DataFrame]:
    """
    Split the DataFrame into k partitions. Each row in the DataFrame will exist in 
    exactly one of the partitions.

    :param df: pd.DataFrame, Dataset to partition
    :param k: int|None, Create k partitions of equal size. If provided, frac must be None.
    :param frac: list[float]|None, List specifying the proportion of elements that should
        be placed in each partition. Elements must sum to 1.0. If provided, k must be None.
    :param stratify_by: str|None, For classification tasks, a column label (e.g., 'class')
        may be provided to ensure that each of the k partitions has a relatively equal
        distribution of instances in each class. If None, the partitions will be created
        randomly, without any attempt at stratifying (default: None).

    :return tuple, A tuple of k DataFrames
    """

    if k is None and frac is None:
        raise Exception("One of k or frac must be provided to split, neither provided")
    elif k is not None and frac is not None:
        raise Exception("Only one of k or frac may be provided to split, both provided")

    if k is not None:
        frac = [1 / k] * k

    n_partitions = len(frac)
    partitions = [pd.DataFrame() for _ in range(n_partitions)]
    n_per_partition = [int(df.shape[0] * frac[i]) for i in range(n_partitions)]

    if stratify_by is None:
        _random_split(df, n_per_partition, partitions)
    else:
        _stratified_split(df, stratify_by, n_per_partition, partitions)

    return partitions

def _random_split(df: pd.DataFrame, n_per_partition: list[int], partitions: list[pd.DataFrame]):
    remaining = df.copy()
    n_partitions = len(n_per_partition)

    while remaining.shape[0] > 0:
        # Pick a random partition out of the partitions that have not yet been filled up.
        # (If all partitions have ≥n_per_partition[i] elements, pick any random partition for
        # each remaining element).
        unfilled_partition_indices = list(filter(
            lambda i: partitions[i].shape[0] < n_per_partition[i], 
            list(range(n_partitions))
        ))
        if len(unfilled_partition_indices) > 0:
            add_to_index = random.choice(unfilled_partition_indices)
        else:
            add_to_index = random.randint(0, n_partitions - 1)

        # Add a random row to the chosen partition and remove it from set of remaining rows.
        row = remaining.sample(1)
        partitions[add_to_index] = pd.concat(
            [partitions[add_to_index], row],
            axis=0,
            copy=False
        )
        remaining.drop(index=row.index, inplace=True)

def _stratified_split(df: pd.DataFrame, stratify_by: str, frac: list[float], partitions: list[pd.DataFrame]):
    n_partitions = len(frac)
    df_for_each_class = [df[df[stratify_by] == c] for c in df[stratify_by].unique()]

    # (partitions_for_each_class[i][j] = j'th partition of class i, for 0≤i<n_classes, 0≤j<k)
    partitions_for_each_class = []
    for class_df in df_for_each_class:
        partitions_for_current_class = _split(class_df, frac=frac)
        partitions_for_each_class.append(partitions_for_current_class)
    
    for class_index in range(len(partitions_for_each_class)):
        for partition_index in range(n_partitions):
            partitions[partition_index] = pd.concat(
                [partitions[partition_index], partitions_for_each_class[class_index][partition_index]],
                axis=0,
                copy=False,
            )
