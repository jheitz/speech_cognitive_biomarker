import numpy as np
import pandas as pd

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from sklearn.linear_model import LinearRegression
from util.helpers import prepare_demographics
from data_analysis.data_transformation.residuals import Residuals

class FeatureResiduals(Residuals):
    """
    Remove the effect of demographic variables (such as age) from the features
    Instead of keeping the raw features, we want to keep the residuals of a multiple linear regression model
    which predicts the cognitive scores based on demographic variables
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "FeatureResiduals"
        print(f"Initializing {self.name}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Calculating residuals for features, given {dataset}")

        assert 'demographics' in dataset.data_variables
        assert 'features' in dataset.data_variables

        input, _ = prepare_demographics(dataset.demographics)
        output = dataset.features

        new_output = self._linear_regression_residuals(input, output)

        new_dataset = self._prepare_new_dataset(dataset)
        new_dataset.feature_residuals = new_output

        return new_dataset



