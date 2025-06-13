from abc import abstractmethod

import numpy as np
import pandas as pd

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from sklearn.linear_model import LinearRegression
from util.helpers import prepare_demographics

class Residuals(DataTransformer):
    """
    Remove the effect of demographic variables (such as age) from a dataframe
    Instead of keeping the raw scores, we want to keep the residuals of a multiple linear regression model
    which predicts the cognitive scores based on demographic variables
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _linear_regression_residuals(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(input_df, pd.DataFrame), f"input_df must be a Pandas DataFrame, but is {type(input_df)}"
        assert isinstance(output_df, pd.DataFrame), f"output_df must be a Pandas DataFrame, but is {type(output_df)}"
        n_rows = input_df.shape[0]
        assert output_df.shape[0] == n_rows

        XY = pd.concat((input_df.reset_index(drop=True), output_df.reset_index(drop=True)), axis=1)
        assert XY.shape[0] == n_rows
        non_nan_filter = ~XY.isna().any(axis=1)
        non_nan_indices = XY.index[non_nan_filter]

        X = XY[input_df.columns].loc[non_nan_indices, :]
        Y = XY[output_df.columns].loc[non_nan_indices, :]

        new_columns = {}
        for target_variable in Y.columns:
            model = LinearRegression()
            y = Y[target_variable]
            model.fit(X, y)
            y_predicted = model.predict(X)
            #print(target_variable, y.dtype)
            residuals = y - y_predicted

            new_columns[target_variable] = pd.Series(np.empty((n_rows,)), index=XY.index)
            new_columns[target_variable][:] = np.nan
            new_columns[target_variable].loc[non_nan_indices] = residuals

        new_df = pd.DataFrame(new_columns)
        return new_df

    def _prepare_new_dataset(self, dataset):
        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config
        return new_dataset


@abstractmethod
def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        pass



