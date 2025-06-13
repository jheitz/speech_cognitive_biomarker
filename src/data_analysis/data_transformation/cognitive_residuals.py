import numpy as np
import pandas as pd

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from sklearn.linear_model import LinearRegression
from util.helpers import prepare_demographics
from data_analysis.data_transformation.residuals import Residuals

class CognitiveResiduals(Residuals):
    """
    Remove the effect of demographic variables (such as age) from the cognitive scores
    Instead of keeping the raw scores, we want to keep the residuals of a multiple linear regression model
    which predicts the cognitive scores based on demographic variables
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "CognitiveResiduals"
        print(f"Initializing {self.name}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Calculating residuals for cognitive scores, given {dataset}")

        assert 'demographics' in dataset.data_variables
        assert 'acs_outcomes_imputed' in dataset.data_variables
        assert 'language_task_scores' in dataset.data_variables

        input, _ = prepare_demographics(dataset.demographics)
        output = pd.concat((dataset.acs_outcomes_imputed, dataset.language_task_scores), axis=1)

        new_output = self._linear_regression_residuals(input, output)

        new_dataset = self._prepare_new_dataset(dataset)
        new_dataset.cognitive_residuals = new_output

        return new_dataset



