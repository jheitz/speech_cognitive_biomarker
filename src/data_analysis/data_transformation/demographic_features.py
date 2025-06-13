import numpy as np
import pandas as pd

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset, DatasetType
from util.helpers import prepare_demographics

class DemographicFeatures(DataTransformer):
    """
    Add demographic features
    E.g. predict a target variable using linguistic + demographic features
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "DemographicFeatures"
        print(f"Initializing {self.name}")

        try:
            self.selected_features = self.config.config_demographic_features.selected_features
        except (AttributeError, KeyError):
            self.selected_features = ["age", "gender_unified", "education_binary", "country"]

        try:
            self.target_variable = self.config.config_model.target_variable
        except:
            self.target_variable = None

        print(f"... using selected_features {self.selected_features}")

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Adding demographic information to feature matrix for {dataset}")

        assert 'demographics' in dataset.data_variables
        assert isinstance(dataset.demographics, pd.DataFrame)

        if 'features' in dataset.data_variables:
            assert isinstance(dataset.features, pd.DataFrame)
            assert dataset.features.shape[0] == dataset.demographics.shape[0]

        if dataset.type == DatasetType.ADReSS:
            demographics_numerical = dataset.demographics[self.selected_features]
        else:
            demographics_numerical, _ = prepare_demographics(dataset.demographics[self.selected_features], include_socioeconomic='socioeconomic' in self.selected_features)

        # fillna for socioeconomic (as this is used as a contron analysis and should thus keep the same number of rows)
        if 'socioeconomic' in self.selected_features:
            demographics_numerical['socioeconomic'] = demographics_numerical['socioeconomic'].fillna(5)

        # remove target variable if available, e.g. for cognitiveAge, where the target is age
        if self.target_variable is not None and self.target_variable in self.selected_features:
            demographics_numerical = demographics_numerical.drop(columns=[self.target_variable])

        demographics_numerical = demographics_numerical.rename(columns={c: f"dem_{c}" for c in demographics_numerical.columns})

        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config

        new_dataset.features = self._create_new_feature_df(new_dataset, demographics_numerical)

        return new_dataset



