import pandas as pd

from .dataset import Dataset
from .imputation import impute_missing_values
import encoding

def preprocess_data(
    dfs: dict[str, pd.DataFrame],
    datasets: dict[str, Dataset],
    should_impute_missing_values: bool=True,
    should_one_hot_encode_nominal_attrs: bool=True,
    should_num_encode_ordinal_attrs: bool=True
) -> dict[str, pd.DataFrame]:
    """
    Preprocess data.

    :param dfs: dict[str, pd.DataFrame], Dictionary mapping dataset names to corresponding data frames
    :param datsets: dict[str, Dataset], Dictionary mapping dataset names to corresponding metadata
    :param should_impute_missing_values: bool, Boolean indicating whether missing values should be 
        imputed (default: True)
    :param should_one_hot_encode_nominal_attrs: bool, Boolean indicating whether nominal attributes should 
        be one hot encoded (default: True)
    :param should_num_encode_ordinal_attrs: bool, Boolean indicating whether ordinal attribiutes should be
        encoded as numbers (default: True)

    :returns dict[str, pd.DataFrame], Dictionary mapping dataset names to data frames with the result
        of the preprocessing steps
    """
    return {
        name: _preprocess_dataset(
            dfs[name], 
            datasets[name],
            should_impute_missing_values=should_impute_missing_values, 
            should_one_hot_encode_nominal_attrs=should_one_hot_encode_nominal_attrs, 
            should_num_encode_ordinal_attrs=should_num_encode_ordinal_attrs
        ) for name in dfs.keys()
    }

def _preprocess_dataset(
    df: pd.DataFrame,
    dataset: Dataset,
    should_impute_missing_values: bool=True,
    should_one_hot_encode_nominal_attrs: bool=True,
    should_num_encode_ordinal_attrs: bool=True
) -> pd.DataFrame:
    df = df.copy()

    if should_impute_missing_values:
        df = impute_missing_values(df, dataset)

    if should_one_hot_encode_nominal_attrs:
        for col in dataset.nominal_cols:
            encoding.one_hot_encode(df, col)
        dataset.nominal_cols = []
    
    if should_num_encode_ordinal_attrs:
        for col, values in dataset.ordinal_cols.items():
            df = encoding.encode_ordinal(df, col, values)
        dataset.ordinal_cols = []

    return df
