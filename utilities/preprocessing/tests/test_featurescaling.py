from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import unittest

from utilities.preprocessing import featurescaling

class TestFeatureScaling(unittest.TestCase):
    def test_standardize(self):
        df_train = pd.DataFrame({
            'col': [1, 2, 3, 4, 5],
        })
        df_test = pd.DataFrame({
            'col': [1, 1, 4, 3, 2],
        })
        got_train, got_test = featurescaling.standardize(df_train, df_test, 'col')

        # Note: sklearn.preprocessing.StandardScaler uses a biased estimator of standard
        # deviation (dividing by n instead of n-1), but featurescaling.standardize uses
        # the unbiased estimator (dividing by n-1). This means that the results will be
        # slightly different, which is why the expected values here are manually precomputed
        # instead of using sklearn.preprocessing.StandardScaler.
        expected_train = pd.DataFrame({
            'col': [-1.26491106, -0.63245553, 0., 0.63245553, 1.26491106],
        })
        expected_test = pd.DataFrame({
            'col': [-1.26491106, -1.26491106, 0.63245553, 0., -0.63245553],
        })
        
        pd.testing.assert_frame_equal(expected_train, got_train)
        pd.testing.assert_frame_equal(expected_test, got_test)

    def test_min_max_scale(self):
        df_train = pd.DataFrame({
            'col': [1, 2, 3, 4, 5],
        })
        df_test = pd.DataFrame({
            'col': [1, 1, 4, 3, 2],
        })
        got_train, got_test = featurescaling.min_max_scale(df_train, df_test, 'col')
        
        scaler = MinMaxScaler()
        expected_train = scaler.fit_transform(df_train)
        expected_test = scaler.transform(df_test)

        np.testing.assert_array_equal(expected_train, got_train.to_numpy())
        np.testing.assert_array_equal(expected_test, got_test.to_numpy())
