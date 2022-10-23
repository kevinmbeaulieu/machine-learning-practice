import numpy as np
import pandas as pd
import sklearn

from utilities.preprocessing.dataset import Dataset

def encode_categorical_data(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """
    Return a new Pandas DataFrame with any categorical attributes encoded as numerical attributes.
    Ordinal data will be encoded as integers 1...n.
    Nominal data will be one-hot-encoded, dropping the first column.

    :param df: pd.DataFrame, Pandas DataFrame to process
    :param dataset: Dataset, Metadata about the dataset being processed

    :returns pd.DataFrame, A new DataFrame in which all categorical attributes have been
        replaced by numerical attributes.
    """

    result = df.copy()

    for col in dataset.nominal_cols:
        result = one_hot_encode(result, col)

    for col_name, values in dataset.ordinal_cols.items():
        encode_ordinal(result, col_name, values)

    return result

def one_hot_encode(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    dummy_cols = pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)
    df.drop(col_name, axis=1, inplace=True)
    return pd.concat([df, dummy_cols], axis=1, copy=False)

def encode_ordinal(df: pd.DataFrame, col_name: str, values: list[any]) -> pd.DataFrame:
    df = df.copy()
    for i in range(len(values)):
        df[col_name].replace(values[i], i + 1, inplace=True)
    df[col_name] = df[col_name].astype('int64')
    return df
