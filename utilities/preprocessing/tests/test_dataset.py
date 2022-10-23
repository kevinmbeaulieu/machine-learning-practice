import unittest

from utilities.preprocessing.dataset import Dataset

class TestDataset(unittest.TestCase):
    def test_load_data(self):
        dataset = Dataset(
            name='test dataset',
            task='regression',
            file_path='utilities/preprocessing/tests/fixtures/dataset1.csv',
            col_names=['Col A', 'Col B', 'Col C', 'output'],
            ignore_cols=['Col B'],
            nominal_cols=['Col A'],
            ordinal_cols={
                'Col C': ['first', 'second', 'third']
            },
        )
        got_df = dataset.load_data()

        self.assertListEqual(['Col A', 'Col C', 'output'], got_df.columns.to_list())

if __name__ == '__main__':
    unittest.main()