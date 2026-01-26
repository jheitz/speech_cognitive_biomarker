import numpy as np
import pandas as pd
import sys
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '..') # to make the import from parent dir work
from data_analysis.data_preprocessing.helpers.feature_norm_data_calculator import FeatureNormDataCalculator
from data_analysis.data_preprocessing.base_data_preprocessor import BasePreprocessor


class FeatureOutlierRemovalImputation(BasePreprocessor):
    """
    Remove outliers and impute missing values
    Based on /src/data_preparation/preparation_logic/ACS_outlier_removal_imputation.py

    We implement this in a sklearn-style way, with a fit / transform method
    This is important to avoid data leakage: e.g. the norms should only be fit on the train data, not on the test data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "FeatureOutlierRemovalImputation"
        print(f"Initializing {self.name}")

        self.norm_calculator = FeatureNormDataCalculator(self.CONSTANTS)
        self.z_score_limit = 4  # z-score limit defining an outlier
        self.demographic_cols_for_norms = self.norm_calculator.demographic_cols

        self.verbose = False
        self.standardize_features_before_imputation = True  # for BayesianRidge, seems to make no difference

        self.features_to_remove = []  # remove features that are inadequate

    def _remove_inadequate_features(self, features_df):
        # If a feature has more than 80% of samples with a certain value, we remove it.
        # These features do not have a lot of variance (and probably not a lot of prognostic value)
        # However, the cause problems with the outlier removal logic
        most_frequent_value = features_df.apply(lambda column: column.mode()[0], axis=0)
        is_most_frequent_value = features_df == most_frequent_value
        count_most_frequent_value = is_most_frequent_value.sum(axis=0)
        proportion_of_most_frequent_value = count_most_frequent_value / features_df.shape[0]
        cutoff_fraction = 0.8
        high_proportion_of_most_frequent_value = proportion_of_most_frequent_value > cutoff_fraction
        self.features_to_remove = high_proportion_of_most_frequent_value[high_proportion_of_most_frequent_value].index
        print(f"... removing {len(self.features_to_remove)} features because their distribution is not good for outlier removal (more than {cutoff_fraction*100}% of values are the same): {list(self.features_to_remove)}")
        return features_df.drop(columns=self.features_to_remove)

    def _fit_outlier_removal(self, features, demographics, sample_names):
        # remove inadequate features
        features = self._remove_inadequate_features(features.copy())

        # fit the feature-specific norms
        features_with_demographics = pd.concat((demographics[self.demographic_cols_for_norms].reset_index(drop=True),
                                        features.reset_index(drop=True)),
                                       axis=1)

        # fit norms
        self.norm_calculator.create_norms(features_with_demographics)

    def _fit_imputation(self, features):
        # imputation_basic_estimator = ExtraTreesRegressor(n_estimators=20)
        # imputation_basic_estimator = RandomForestRegressor(n_estimators=100)
        imputation_basic_estimator = BayesianRidge()  # default model

        self.imputer = IterativeImputer(
            estimator=imputation_basic_estimator,
            max_iter=30,
            initial_strategy='mean',
            imputation_order='ascending',  # from features with the fewest missing values to most
            min_value=-np.inf,
            max_value=np.inf,
            verbose=0,
            random_state=0,
        )

        data_for_imputation = features.copy()

        if self.verbose:
            print("Feature statistics before imputation:")
            print(data_for_imputation.describe().loc[['count', 'mean', 'std'], :].T)

        if self.standardize_features_before_imputation:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(data_for_imputation)
            data_for_imputation = pd.DataFrame(self.standard_scaler.transform(data_for_imputation),
                                               columns=data_for_imputation.columns)
            if self.verbose:
                print("...Standardized:")
                print(data_for_imputation.describe().loc[['count', 'mean', 'std'], :].T)

        self.imputer.fit(data_for_imputation)

    def _transform_outlier_removal(self, features, demographics, sample_names):
        # remove inadequate features
        features = features.drop(columns=self.features_to_remove)

        # add demographics, for norms
        features_with_demographics = pd.concat((demographics[self.demographic_cols_for_norms].reset_index(drop=True),
                                        features.reset_index(drop=True)),
                                       axis=1)

        # calculating z-scores based on norms prepared in fit() method
        z_scores = self.norm_calculator.calculate_norm_scores(features_with_demographics)
        z_columns = [c for c in z_scores.columns if '_Z' in c]
        norm_scores_outliers = z_scores[z_columns].copy()
        norm_scores_outliers[z_columns] = np.logical_or(np.abs(norm_scores_outliers[z_columns]) >= self.z_score_limit,
                                                        norm_scores_outliers[z_columns].isna())
        norm_scores_outliers['num_outliers'] = norm_scores_outliers.apply(lambda row: row[z_columns].sum(), axis=1)
        print("Distribution of number of outliers per participant:",
              norm_scores_outliers['num_outliers'].value_counts().to_dict())

        #if self.run_parameters is not None:
        #    pd.DataFrame({
        #        'sample_name': sample_names,
        #        'num_outliers': norm_scores_outliers['num_outliers'],
        #    }).to_csv(os.path.join(self.run_parameters.results_dir, "outliers_per_participant.csv"))
        #    norm_scores_outliers[z_columns].apply(lambda col: col.sum(), axis=0).to_csv(
        #        os.path.join(self.run_parameters.results_dir, "outliers_per_feature.csv"))

        features_transformed = features.copy()

        # now remove the value from features, where norm_scores_outliers is True
        for row_idx, row in norm_scores_outliers.query("num_outliers > 0").iterrows():
            for col in z_columns:
                if row[col] == True:  # an outlier
                    original_col = col.replace('_Z', '')
                    features_transformed.loc[row_idx, original_col] = np.nan

        return features_transformed

    def _transform_imputation(self, features):
        # Impute missing values using multivariate imputation
        # The missing values are a result of the outlier removal
        # Based on self.standard_scaler and self.imputer which was fit in the _fit_imputation method
        data_for_imputation = features.copy()

        if self.standardize_features_before_imputation:
            data_for_imputation = pd.DataFrame(self.standard_scaler.transform(data_for_imputation),
                                               columns=data_for_imputation.columns)
            if self.verbose:
                print("...Standardized:")
                print(data_for_imputation.describe().loc[['count', 'mean', 'std'], :].T)

        data_imputed = self.imputer.transform(data_for_imputation)
        data_imputed = pd.DataFrame(data_imputed, columns=data_for_imputation.columns)
        if self.standardize_features_before_imputation:
            data_imputed = pd.DataFrame(self.standard_scaler.inverse_transform(data_imputed),
                                             columns=data_for_imputation.columns)

        if self.verbose:
            print("Imputed data statistics:")
            print(data_imputed.describe().loc[['count', 'mean', 'std'], :].T)

        return data_imputed

    def _remove_demographics_from_features(self, features_df):
        # remove demographic columns from features -> we don't do outlier removal on these
        dem_cols_in_features = [c for c in features_df.columns if c[:4] == 'dem_']
        print(f"... temporarily removing demographic columns from the features, for outlier detection: {dem_cols_in_features}")
        demographics_in_features = features_df[dem_cols_in_features]
        remaining_features = features_df.drop(columns=dem_cols_in_features)
        return remaining_features, demographics_in_features


    def fit(self, features, demographics, sample_names):
        """
        Fit the feature-specific norms and the imputer
        """
        print("Fitting feature norms for outlier removal")

        # remove demographic columns from features -> we don't do outlier removal on these
        features, demographics_in_features = self._remove_demographics_from_features(features)

        # fit the feature-specific norms
        self._fit_outlier_removal(features, demographics, sample_names)

        print("Running outlier removal, s.t. we can fit the imputation model")
        # remove outliers in the same data, based on these norms. This is necessary for the imputation fitting
        features_without_outliers = self._transform_outlier_removal(features, demographics, sample_names)

        # fit imputation model, based on these data
        print("Fitting imputation model")
        self._fit_imputation(features_without_outliers)

    def transform(self, features, demographics, sample_names):
        """
        Apply outlier removal and imputation, based on the fitted models prepared by the fit method
        """
        print("Transforming data: Outlier removal (based on previously calculated norms) and imputation (based on previously calculated fit)")
        # remove demographic columns from features -> we don't do outlier removal on these
        features, demographics_in_features = self._remove_demographics_from_features(features)

        features_without_outliers = self._transform_outlier_removal(features, demographics, sample_names)
        features_imputed = self._transform_imputation(features_without_outliers)

        # add the demographic cols back in
        features_imputed = pd.concat((features_imputed, demographics_in_features), axis=1)

        return features_imputed

