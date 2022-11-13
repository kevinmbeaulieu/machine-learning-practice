import pandas as pd

from .dataset import Dataset

def impute_missing_values(df: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    """
    Fill in any missing values in a dataset.

    Missing values for categorical attributes will be replaced with the mode.
    Missing values for all other attributes will be replaced with the mean.

    :param df: pd.DataFrame, Pandas DataFrame for which to impute missing values
    :param dataset: Dataset, Metadata about the dataset being processed

    :return pd.DataFrame, A new DataFrame with all missing values filled in
    """

    result = df.copy()
    if dataset.missing_value_symbol is None:
        return result

    for col_name in result.columns:
        if (dataset.task == 'classification' and col_name == 'class') or (dataset.task == 'regression' and col_name == 'output'):
            continue
        elif col_name in dataset.nominal_cols or col_name in dataset.ordinal_cols:
            try:
                mode = df[df[col_name] != dataset.missing_value_symbol][col_name].mode()
                # If multiple classes tied for most frequent, arbitrarily use the first one.
                replace_with = mode[0]
            except (ValueError, TypeError) as e:
                print("Failed to compute mode for categorical attribute {} in {} dataset: {}".format(col_name, dataset.name, e))
                continue
        else:
            try:
                replace_with = df[df[col_name] != dataset.missing_value_symbol][col_name].mean()
            except (ValueError, TypeError) as e:
                print("Failed to compute mean for numerical attribute {} in {} dataset: {}".format(col_name, dataset.name, e))
                continue

        result[col_name].replace(dataset.missing_value_symbol, replace_with, inplace=True)
    
    return result
    
