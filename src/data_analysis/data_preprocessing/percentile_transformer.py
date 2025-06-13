import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_analysis.data_preprocessing.base_data_preprocessor import BasePreprocessor

class PercentileTransformer(BasePreprocessor):
    """
    Transform variable into percentiles (number from 1 - 100)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Percentile Transformer"
        print(f"Initializing {self.name}")
        self.percentiles = None


    def fit(self, variable) -> None:
        variable = np.array(variable)
        assert len(variable.shape) == 1, "Should be 1d, but has shape {}".format(variable.shape)
        self.q = np.arange(1, 101)
        self.percentiles = np.percentile(variable, self.q)

    def transform(self, variable) -> pd.Series:
        variable = np.array(variable)
        assert len(variable.shape) == 1, "Should be 1d, but has shape {}".format(variable.shape)
        assert self.percentiles is not None, "Need to fit PercentileTransformer before transforming"

        transformed_idx = np.array([np.argmin(val > self.percentiles) for val in variable])
        transformed = self.q[transformed_idx]

        return pd.Series(transformed)




