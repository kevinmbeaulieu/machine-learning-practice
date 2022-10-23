import pandas as pd

from utilities.preprocessing.dataset import Dataset

def standardize(df_train: pd.DataFrame, df_test: pd.DataFrame, col_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform z-score standardization for an attribute in the training and test datasets.

    Note: The mean and standard deviation of the training set is used for standardizing both the training and test sets.

    :param df_train: pd.DataFrame, Training dataset.
    :param df_test: pd.DataFrame, Test dataset.
    :param col_name: str, Column label of attribute to standardize (must have the same name in df_train and df_test).

    :return tuple[pd.DataFrame, pd.DataFrame]:
        0: A new DataFrame containing the training set with the specified attribute standardized.
        1: A new DataFrame containing the test set with the specified attributed standardized.
    """

    train_mean = df_train[col_name].mean()
    train_stddev = df_train[col_name].std()

    result_train = df_train.copy()
    result_test = df_test.copy()

    result_train[col_name] = (result_train[col_name] - train_mean) / train_stddev
    result_test[col_name] = (result_test[col_name] - train_mean) / train_stddev

    return (result_train, result_test)

def standardize_attributes(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform z-score standardization for all specified attributes in the dataset.

    Note: The mean and standard deviation of the training set is used for standardizing both the training and test sets.

    :param df_train: pd.DataFrame, Training dataset.
    :param df_test: pd.DataFrame, Test dataset.
    :param dataset: Dataset, Metadata about the dataset being processed

    :return tuple[pd.DataFrame, pd.DataFrame]:
        0: A new DataFrame containing the training set with the specified attribute standardized.
        1: A new DataFrame containing the test set with the specified attributed standardized.
    """
    result_train, result_test = df_train, df_test
    for col_name in dataset.standardize_cols:
        result_train, result_test = standardize(result_train, result_test, col_name)
    return result_train, result_test

def min_max_scale(df_train: pd.DataFrame, df_test: pd.DataFrame, col_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform min-max scaling for all specified attributes in the dataset.

    Note: The min and max of the training set is used for normalizing both the training and test sets.

    :param df_train: pd.DataFrame, Training dataset.
    :param df_test: pd.DataFrame, Test dataset.
    :param col_name: str, Column label of attribute to normalize (must have the same name in df_train and df_test).

    :return tuple[pd.DataFrame, pd.DataFrame]:
        0: A new DataFrame containing the training set with the specified attribute standardized.
        1: A new DataFrame containing the test set with the specified attributed standardized.
    """

    train_min = df_train[col_name].min()
    train_max = df_train[col_name].max()

    result_train, result_test = df_train.copy(), df_test.copy()

    result_train[col_name] = (result_train[col_name] - train_min) / (train_max - train_min)
    result_test[col_name] = (result_test[col_name] - train_min) / (train_max - train_min)

    return (result_train, result_test)

def normalize_attributes(df_train: pd.DataFrame, df_test: pd.DataFrame, dataset: Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform min-max scaling for all specified attributes in the dataset.

    Note: The min and max of the training set is used for normalizing both the training and test sets.

    :param df_train: pd.DataFrame, Training dataset.
    :param df_test: pd.DataFrame, Test dataset.
    :param dataset: Dataset, Metadata about the dataset being processed

    :return tuple[pd.DataFrame, pd.DataFrame]:
        0: A new DataFrame containing the training set with the specified attribute standardized.
        1: A new DataFrame containing the test set with the specified attributed standardized.
    """
    result_train, result_test = df_train, df_test
    for col_name in dataset.normalize_cols:
        result_train, result_test = min_max_scale(result_train, result_test, col_name)
    return result_train, result_test
