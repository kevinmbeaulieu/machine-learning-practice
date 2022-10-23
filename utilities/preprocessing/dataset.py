import pandas as pd

class Dataset:
    """
    Data structure for representing a dataset, containing any dataset-specific
    information that may be needed later in the pipeline.
    """

    def __init__(
        self,
        name: str,
        task: str,
        file_path: str,
        col_names: list[str],
        header: int = None,
        ignore_cols: list[str] = [],
        nominal_cols: list[str] = [],
        ordinal_cols: dict[str, list[any]] = {},
        standardize_cols: list[str] = [],
        normalize_cols: list[str] = [],
        missing_value_symbol: str = None,
        positive_class: str = None,
        negative_class: str = None,
        metrics: list[str] = [],
    ):
        """
        Define a new dataset with the specific properties.

        :param name: str, A human-readable name for the dataset.
        :param task: str ('classification'|'regression'), String indicating whether
            classification or regression should be used for this dataset.
        :param file_path: str, Path to CSV file containing the data.
        :param col_names: list[str], Labels for each column/attribute. If the CSV
            already contains a header row, the labels passed in here will override
            those labels.
            For classification data, the class attribute *must* be named 'class' here.
            For regression data, the output value *must* be named 'output' here.
        :param header: int|None, Index of header row in CSV, or None if no header
            is present (default: None).
        :param ignore_cols: list[str], Labels for any columns present in the CSV
            that should be ignored (e.g., ID fields) (default: []).
        :param nominal_cols: list[str], Labels for columns which should be treated
            as nominal categorical variables (default: []).
        :param ordinal_cols: dict[str, list[any]], Dictionary mapping the label for each
            ordinal categorical variable to the ordered list of values it can take on
            (default: {}).
        :param standardize_cols: list[str], Labels for columns which should be standardized
            according to z-score standardization.
        :param normalize_cols: list[str], Labels for columns which should be normalized
            using min-max scaling.
        :param missing_value: str, Symbol used to represent missing values in the CSV,
            or None if the dataset is known to contain no missing values (default: None).
        :param positive_class: str, Label used to represent the positive class for
            a binary classification task, or None if the dataset does not represent a binary
            classification task.
        :param negative_class: str, Label used to represent the negative class for
            a binary classification task, or None if the dataset does not represent a binary
            classification task.
        :param metrics: list[str], List of evaluation metrics to compute.
        """
        self.name = name
        self.task = task
        self.file_path = file_path
        self.col_names = col_names
        self.header = header
        self.ignore_cols = ignore_cols
        self.nominal_cols = nominal_cols
        self.ordinal_cols = ordinal_cols
        self.standardize_cols = standardize_cols
        self.normalize_cols = normalize_cols
        self.missing_value_symbol = missing_value_symbol
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.metrics = metrics

    def load_data(self) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.

        :param dataset: Dataset, The dataset to load.

        :return pd.DataFrame, A new DataFrame with the dataset
        """

        df = pd.read_csv(self.file_path, names=self.col_names, header=self.header)
        if len(self.ignore_cols) > 0:
            df.drop(self.ignore_cols, axis=1, inplace=True)
        return df

def get_default_datasets() -> dict[str, Dataset]:
    """
    Get the default set of six datasets used for most assignments.

    :returns dict[str, Dataset], Dictionary mapping dataset names to metadata about the dataset
    """
    data_computer_hardware = Dataset(
        name='Computer Hardware',
        task='regression',
        file_path='../Datasets/Regression/Computer Hardware/machine.data',
        col_names=['vendor name', 'model name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'output', 'ERP'],
        ignore_cols=['ERP'],
        nominal_cols=['vendor name', 'model name'],
        standardize_cols=['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX'],
        metrics=['mse', 'mae', 'r2', 'pearson'],
    )

    data_forest_fires = Dataset(
        name='Forest Fires',
        task='regression',
        file_path='../Datasets/Regression/Forest Fires/forestfires.csv',
        col_names=['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'output'],
        header=0,
        nominal_cols=['month', 'day'],
        standardize_cols=['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain'],
        metrics=['mse', 'mae', 'r2', 'pearson'],
    )

    data_abalone = Dataset(
        name='Abalone',
        task='regression',
        file_path='../Datasets/Regression/Abalone/abalone.data',
        col_names=['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'output'],
        nominal_cols=['sex'],
        standardize_cols=['length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight'],
        metrics=['mse', 'mae', 'r2', 'pearson'],
    )

    data_congressional_vote = Dataset(
        name='Congressional Votes',
        task='classification',
        file_path='../Datasets/Classification/Congressional Vote/house-votes-84.data',
        col_names=['class', 'handicapped infants', 'water project cost sharing', 'adoption of the budget resolution', 'physician fee freeze', 'El Salvador aid', 'religious groups in schools', 'anti-satellite test ban', 'aid to Nicaraguan contras', 'MX missile', 'immigration', 'synfuels corporation cutback', 'education spending', 'superfund right to sue', 'crime', 'duty-free exports', 'export administration act South Africa'],
        nominal_cols=['handicapped infants', 'water project cost sharing', 'adoption of the budget resolution', 'physician fee freeze', 'El Salvador aid', 'religious groups in schools', 'anti-satellite test ban', 'aid to Nicaraguan contras', 'MX missile', 'immigration', 'synfuels corporation cutback', 'education spending', 'superfund right to sue', 'crime', 'duty-free exports', 'export administration act South Africa'],
        metrics=['acc'],
    )

    data_car_evaluation = Dataset(
        name='Car Evaluation',
        task='classification',
        file_path='../Datasets/Classification/Car Evaluation/car.data',
        col_names=['buying price', 'maintenance price', 'doors', 'persons', 'luggage boot size', 'safety', 'class'],
        ordinal_cols={
            'buying price': ['low', 'med', 'high', 'vhigh'],
            'maintenance price': ['low', 'med', 'high', 'vhigh'],
            'doors': [2, 3, 4, '5more'],
            'persons': [2, 4, 'more'],
            'luggage boot size': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']
        },
        metrics=['acc'],
    )

    data_breast_cancer = Dataset(
        name='Breast Cancer',
        file_path='../Datasets/Classification/Breast Cancer/breast-cancer-wisconsin.data',
        task='classification',
        col_names=['sample code number', 'clump thickness', 'cell size uniformity', 'cell shape uniformity', 'marginal adhesion', 'single epithelial cell size', 'bare nuclei', 'bland chromatin', 'normal nucleoli', 'mitoses', 'class'],
        ignore_cols=['sample code number'],
        missing_value_symbol='?',
        positive_class='4',
        negative_class='2',
        metrics=['acc', 'precision', 'recall', 'f1'],
    )

    datasets = [data_computer_hardware, data_forest_fires, data_abalone, data_congressional_vote, data_car_evaluation, data_breast_cancer]
    return {dataset.name: dataset for dataset in datasets}

def load_datasets(datasets: dict[str, Dataset]) -> dict[str, pd.DataFrame]:
    return {name: dataset.load_data() for name, dataset in datasets.items()}
