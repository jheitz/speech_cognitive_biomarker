from abc import abstractmethod

import pandas as pd

from data_analysis.dataloader.dataset import Dataset

class BasePreprocessor:
    """
    A preprocessor is a sklearn style function that has a fit and transform method
    The fit method is run on training data, and fits some preprocessing model (e.g. data normalization)
    The transform method is run on training and test data, and transforms the model based on the fitted model
    This is to avoid data leakage
    """
    name = "Generic preprocessor"

    def __init__(self, config, constants, run_parameters=None):
        self.config = config
        self.CONSTANTS = constants
        self.run_parameters = run_parameters

    @abstractmethod
    def fit(self, features_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        pass