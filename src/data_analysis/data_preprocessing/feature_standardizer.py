import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_analysis.data_preprocessing.base_data_preprocessor import BasePreprocessor

class FeatureStandardizer(BasePreprocessor):
    """
    Standardize Features (for e.g. linear regression)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Feature Standardizer"
        print(f"Initializing {self.name}")

        self.standard_scaler = StandardScaler()

    def fit(self, features_df: pd.DataFrame) -> None:
        assert isinstance(features_df, pd.DataFrame)

        self.standard_scaler.fit(features_df)

    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(features_df, pd.DataFrame)

        transformed = pd.DataFrame(self.standard_scaler.transform(features_df),
                                               columns=features_df.columns)

        return transformed




