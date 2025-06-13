import numpy as np
import pandas as pd
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '..') # to make the import from parent dir work
from util.acs_norm_data import OriginalACSNormDataCalculator, RegressionBasedACSNormDataCalculator


class ACSOutlierRemovalImputation:
    """
    Remove outliers and impute missing values
    """
    def __init__(self, constants, version_logic=1):
        self.constants = constants

        # outlier removal using ACS norms
        norm_version = 'regression-based'  # Alternative: norm_version = 'original'
        if norm_version == 'original':
            self.acs_norm_calculator = OriginalACSNormDataCalculator(self.constants, log_problems=True)
            self.z_score_limit = 3  # z-score limit defining an outlier
            self.demographic_cols_for_norms = ['age', 'gender_unified', 'education_binary']
        elif norm_version == 'regression-based':
            self.acs_norm_calculator = RegressionBasedACSNormDataCalculator(self.constants, version_logic=version_logic)
            self.z_score_limit = 4  # z-score limit defining an outlier
            if version_logic == 1:
                self.demographic_cols_for_norms = ['age', 'gender_unified']
            elif version_logic == 2:
                self.demographic_cols_for_norms = ['age', 'gender_unified', 'education_binary', 'country']
            else:
                raise ValueError()

    def _remove_outliers(self, acs_outcomes, demographics):
        print("... removing outliers")
        acs_outcomes_for_norms = acs_outcomes.merge(
            demographics[self.demographic_cols_for_norms + ['study_submission_id']],
            on="study_submission_id", how="left"
        )
        z_scores = self.acs_norm_calculator.calculate_norm_scores(acs_outcomes_for_norms)
        z_columns = [c for c in z_scores.columns if '_Z' in c]
        norm_scores_outliers = z_scores[['study_submission_id'] + z_columns].copy()
        norm_scores_outliers[z_columns] = np.logical_or(np.abs(norm_scores_outliers[z_columns]) >= self.z_score_limit,
                                                        norm_scores_outliers[z_columns].isna())
        norm_scores_outliers['num_outliers'] = norm_scores_outliers.apply(lambda row: row[z_columns].sum(), axis=1)
        print("Distribution of number of outliers per participant:",
              norm_scores_outliers['num_outliers'].value_counts().to_dict())

        # now remove the value from acs_outcomes, where norm_scores_outliers is True
        for _, row in norm_scores_outliers.query("num_outliers > 0").iterrows():
            for col in z_columns:
                if row[col] == True:  # an outlier
                    acs_outcomes_loc = acs_outcomes.query(f"study_submission_id == {row.study_submission_id}")
                    assert acs_outcomes_loc.shape[0] == 1
                    acs_outcomes_idx = acs_outcomes_loc.index[0]
                    original_col = col.replace('_Z', '')
                    acs_outcomes.loc[acs_outcomes_idx, col.replace('_Z', '')] = np.nan

                    # also set derived columns to nan if outlier
                    if original_col in ["connect_the_dots_I_time_msec", "connect_the_dots_II_time_msec"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['connect_the_dots_difference', 'connect_the_dots_fraction']] = np.nan
                    if original_col in ["digit_sequence_1_correct_series", "digit_sequence_2_correct_series"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['digit_sequence_difference', 'digit_sequence_fraction']] = np.nan
                    if original_col in ["wordlist_correct_words"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['wordlist_correct_trial1', 'wordlist_correct_trial2',
                                                            'wordlist_correct_trial3', 'wordlist_correct_trial4',
                                                            'wordlist_correct_trial5', 'wordlist_learning']] = np.nan
                    if original_col in ["place_the_beads_total_extra_moves"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['place_the_beads_extramoves_per_trial']] = np.nan
                    if original_col in ["box_tapping_total_correct"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['box_tapping_span_2x', 'box_tapping_span_1x']] = np.nan
                    if original_col in ["digit_sequence_1_correct_series"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['digit_sequence_1_span_2x', 'digit_sequence_1_span_1x']] = np.nan
                    if original_col in ["digit_sequence_2_correct_series"]:
                        acs_outcomes.loc[acs_outcomes_idx, ['digit_sequence_2_span_2x', 'digit_sequence_2_span_1x']] = np.nan

        return acs_outcomes

    def _impute_missing_values(self, acs_outcomes, verbose=False):
        # Impute missing values using multivariate imputation
        # The missing values are a result of the outlier removal
        print("... imputing missing values")

        standardize_variables = True  # for BayesianRidge, seems to make no difference
        # imputation_basic_estimator = ExtraTreesRegressor(n_estimators=20)
        # imputation_basic_estimator = RandomForestRegressor(n_estimators=100)
        imputation_basic_estimator = BayesianRidge()  # default model

        imputer = IterativeImputer(
            estimator=imputation_basic_estimator,
            max_iter=30,
            initial_strategy='mean',
            imputation_order='ascending',  # from features with the fewest missing values to most
            min_value=-np.inf,
            max_value=np.inf,
            verbose=0,
            random_state=0,
        )

        cols_to_keep = self.constants.ACS_MAIN_OUTCOME_VARIABLES_EXTENDED + ["study_submission_id"]
        data_for_imputation = acs_outcomes.copy()[cols_to_keep]

        if verbose:
            print("ACS outcomes statistics:")
            print(data_for_imputation.describe().loc[['count', 'mean', 'std'], :].T)

        if standardize_variables:
            standard_scaler = StandardScaler()
            standard_scaler.fit(data_for_imputation)
            data_for_imputation = pd.DataFrame(standard_scaler.transform(data_for_imputation),
                                               columns=data_for_imputation.columns)
            if verbose:
                print("...Standardized:")
                print(data_for_imputation.describe().loc[['count', 'mean', 'std'], :].T)

        data_imputed = imputer.fit_transform(data_for_imputation)
        data_imputed = pd.DataFrame(data_imputed, columns=data_for_imputation.columns)
        if standardize_variables:
            data_imputed = pd.DataFrame(standard_scaler.inverse_transform(data_imputed),
                                             columns=data_for_imputation.columns)

        if verbose:
            print("Imputed data statistics:")
            print(data_imputed.describe().loc[['count', 'mean', 'std'], :].T)

        return data_imputed