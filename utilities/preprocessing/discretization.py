import pandas as pd

def discretize(df: pd.DataFrame, col_name: str, strategy: str, n_bins: int) -> pd.DataFrame:
    """
    Partition a real-valued attribute into a series of discretized buckets.

    :param df: pd.DataFrame, Pandas DataFrame containing the data to discretize.
    :param col_name: str, Column label for the attribute to discretize.
    :param strategy: str ('equal-width'|'equal-frequency'), Strategy for defining the buckets.
    :param n_bins: int, Number of buckets to create.

    :return pd.DataFrame, A new DataFrame in which the specified attribute has been replaced 
        with the corresponding discretized version.
    """

    if strategy == 'equal-width':
        new_col = _discretize_equal_width(df[col_name], n_bins)
    elif strategy == 'equal-frequency':
        new_col = _discretize_equal_freq(df[col_name], n_bins)
    else:
        raise Exception("Unrecognized discretization strategy {}".format(strategy))
    
    result = df.copy()
    result[col_name] = new_col
    return result

def _discretize_equal_width(col: pd.Series, n_bins: int) -> pd.Series:
    min_value = col.min()
    max_value = col.max()
    bin_size = (max_value - min_value) / n_bins
    result = ((col - min_value) // bin_size).astype('int')
    result = result + 1 # Index discretized values from 1...n_bins, rather than 0...(n_bins-1)
    result = result.clip(lower=1, upper=n_bins) # Ensure max value doesn't get put into an (n_bins+1)'th bin
    return result

def _discretize_equal_freq(col: pd.Series, n_bins: int) -> pd.Series:
    result = ((col.rank() - 1) // n_bins).astype('int') + 1
    result = result.clip(lower=1, upper=n_bins) # Ensure max value doesn't get put into an (n_bins+1)'th bin
    return result
