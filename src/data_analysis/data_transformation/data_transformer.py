from abc import abstractmethod

import pandas as pd

from data_analysis.dataloader.dataset import Dataset

class DataTransformer:
    name = "Generic data transformer"

    def __init__(self, config, constants, run_parameters=None):
        self.config = config
        self.CONSTANTS = constants
        self.run_parameters = run_parameters

    @abstractmethod
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        pass

    def _create_new_feature_df(self, dataset: Dataset, new_features: pd.DataFrame):
        """
        Create new features DataFrame
        If the dataset already has features, we run some checks and add the new features as new columns
        If not, the new features are used directly
        """
        if 'features' in dataset.data_variables:
            assert isinstance(dataset.features, pd.DataFrame), \
                'features must be DataFrame, otherwise cannot concatenate'
            assert dataset.features.shape[0] == new_features.shape[0]
            return pd.concat((dataset.features.reset_index(drop=True),
                                          new_features.copy().reset_index(drop=True)), axis=1)
        else:
            return new_features.copy()
