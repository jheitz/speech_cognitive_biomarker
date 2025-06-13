import numpy as np
import pandas as pd
import logging
import os

from util.helpers import prepare_demographics


class ACSNormDataCalculator:
    def __init__(self):
        pass

    def calculate_norm_scores(self, data):
        # calculate z-scores for participant's ACS outcomes
        # data should be a pandas dataframe with each row corresponding to one individual
        # and columns for the demographic data and outcome variables
        pass

    def norm_bands(self, demographic_data, outcome_variable):
        # give norm bands for outcome variable, given demographic data
        # demographic_data should be a dataframe with each row corresponding to demographic data (e.g. specific age)
        # where the norm band (mean +- std) is returned
        pass


class OriginalACSNormDataCalculator(ACSNormDataCalculator):

    def __init__(self, CONSTANTS, log_problems=False):
        self.CONSTANTS = CONSTANTS

        # log issues with data
        self.log_problems = log_problems

        # should be given in input data
        self.demographic_input_cols = ['age', 'gender_unified', 'education_binary', 'computer_experience']

        # columns where the original scores are in msec, but the norms in sec
        self.msec_cols = ['connect_the_dots_I_time_msec', 'connect_the_dots_II_time_msec', 'fill_the_grid_total_time']

    def _prepare_data_for_acs_norms(self, data):
        print("... [ACS original norms] Prepare data")

        demographic_cols = ['study_submission_id'] + self.demographic_input_cols if 'study_submission_id' in data.columns else self.demographic_input_cols
        outcomes_in_data = [c for c in data.columns if c in self.CONSTANTS.ACS_MAIN_OUTCOME_VARIABLES_FOR_ORIGINAL_NORMS]
        data = data[demographic_cols + outcomes_in_data].copy()

        # cast msec to sec (to be compatible with norms)
        for col in [c for c in self.msec_cols if c in outcomes_in_data]:
            new_col = f"{col}_for_norms"
            data[new_col] = data[col] / 1000

        data['gender'] = data['gender_unified'].apply(lambda val: 0 if val == 'f' else 1 if val == 'm' else None)
        data = data.drop(columns=['gender_unified'])

        # Age: "Age" (coded in whole years)
        data.loc[:, 'age'] = data.loc[:, 'age'].round(0).astype(int)

        # "Education" (coded as 0 for Verhage 1-5/ISCED 0-4, 1 for Verhage 6-7/ISCED 5-8)
        # see https://ec.europa.eu/eurostat/statistics-explained/index.php?title=International_Standard_Classification_of_Education_(ISCED)#ISCED_1997_.28fields.29_and_ISCED-F_2013
        data['education'] = data.loc[:, 'education_binary'].apply(
            lambda x: 0 if x == 'low-education' else 1 if x in ['high-education'] else x)

        # Recode computer usage
        data['computer_experience'] = data['computer_experience'].replace('>60', 60).astype(float)

        return data

    def _load_norm_data(self, norm_names):
        # Load norms from Excel files
        norms_files = {
            "connect_the_dots_I_time_msec": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_11.csv"),
            "connect_the_dots_II_time_msec": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_13.csv"),
            "wordlist_correct_words": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_14.csv"),
            "avg_reaction_speed": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_17.csv"),
            "place_the_beads_total_extra_moves": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_19.csv"),
            "box_tapping_total_correct": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_21.csv"),
            "fill_the_grid_total_time": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_23.csv"),
            "wordlist_delayed_correct_words": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_25.csv"),
            "digit_sequence_1_correct_series": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_28.csv"),
            "digit_sequence_2_correct_series": os.path.join(self.CONSTANTS.ACS_NORMATIVE_DATA, "norms_O_30.csv"),
        }
        # norm data, renamed to match our naming conventions
        norm_data = {norm_name: pd.read_csv(norms_files[norm_name]).rename(columns={
            'Age': 'age',
            'Education': 'education',
            'Sex': 'gender',
            'Comp_exp': 'computer_experience'
        }) for norm_name in norms_files.keys() if norm_name in norm_names}

        return norm_data

    def _calculate_norm_scores(self, data):
        # Calculate ACS norm scores is based on norms_data_edited.R, the R script written by Maryse from the ACS team
        print("... [ACS original norms] Calculating normative scores of ACS")

        # prepare data
        outcomes_in_data = [c for c in data.columns if c in self.CONSTANTS.ACS_MAIN_OUTCOME_VARIABLES_FOR_ORIGINAL_NORMS]
        data = self._prepare_data_for_acs_norms(data)

        norm_data = self._load_norm_data(outcomes_in_data)

        # Function to calculate normed scores
        def calculate_norms(raw_scores, norms_df, outcome_variable):
            normed_scores = pd.Series(index=raw_scores.index, dtype=float)
            for i, row in raw_scores.iterrows():
                matching_norms = norms_df[
                    (norms_df['age'] == row['age']) &
                    (norms_df['gender'] == row['gender']) &
                    (norms_df['education'] == row['education']) &
                    (norms_df['computer_experience'] == row['computer_experience'])
                    ]
                if not matching_norms.empty:
                    try:
                        score_column = str(int(row['raw_score']))
                    except:
                        score_column = row['raw_score']
                    if score_column in matching_norms.columns:
                        norm_score = matching_norms.iloc[0][score_column]
                        normed_scores.at[i] = norm_score
                    else:
                        valid_values = [int(c) for c in matching_norms.columns if c.isdigit()]
                        valid_range = (min(valid_values), max(valid_values))
                        logging.warning(f"No norm score for raw_score {row['raw_score']} ({outcome_variable}). Valid range is {valid_range}")
                        try:
                            raw_score_int = int(np.round(row['raw_score']))
                            if valid_range[0] > raw_score_int:  # value too small
                                normed_scores.at[i] = -8.0  # z-value of -8
                            elif valid_range[1] < raw_score_int:  # value too small
                                normed_scores.at[i] = 8.0  # z-value of 8
                        except:
                            pass
                else:
                    pass
            return normed_scores

        # round float columns and cast to int
        float_cols = data.select_dtypes(include='float').columns
        data[float_cols] = np.round(data[float_cols]).astype('Int64')

        if self.log_problems:
            # check if demographics allow for norm lookup (which has age range, requires non-nans, etc.)
            if 'study_submission_id' not in data.columns:
                logging.error("Please provide column study_submission_id for logging potential problems with calculating the original ACS norms")
            else:
                for idx, row in data.iterrows():
                    norm_names_where_participant_has_no_data = []
                    for norm_name in norm_data:
                        norm_here = norm_data[norm_name]
                        matching_rows = norm_here[
                            (norm_here['age'] == row['age']) &
                            (norm_here['gender'] == row['gender']) &
                            (norm_here['education'] == row['education']) &
                            (norm_here['computer_experience'] == row['computer_experience'])
                            ]
                        if matching_rows.shape[0] == 0:
                            norm_names_where_participant_has_no_data.append(norm_name)
                    if len(norm_names_where_participant_has_no_data) > 0:
                        demographics_data_participant = ", ".join([f"{key}: {val}" for key, val in row.items() if key in ["age", "gender", "education", "computer_experience"]])
                        logging.warning(f'Cannot calculate norms {"/".join(norm_names_where_participant_has_no_data)} for submission {row["study_submission_id"]} with data {demographics_data_participant}')

        # Calculate normed scores
        ACS_norm_scores = data.copy()
        for norm_name in outcomes_in_data:
            if norm_name in norm_data:
                print(f"Calculating norms {norm_name}...", end=" ")
                norm_name_prepared_column = norm_name if norm_name not in self.msec_cols else f"{norm_name}_for_norms"
                ACS_norm_scores[f'{norm_name}_Z'] = calculate_norms(
                    ACS_norm_scores[['age', 'gender', 'education', 'computer_experience'] + [norm_name_prepared_column]].rename(columns={norm_name_prepared_column: 'raw_score'}),
                    norm_data[norm_name],
                    norm_name
                )
                print("done.")
            else:
                raise ValueError(f"Invalid norm name {norm_name}")

        return ACS_norm_scores

    def _get_norm_bands(self, norm_values, outcome_variable, age=None, gender=None, education=None, computer_experience=None):
        subset = norm_values.copy()
        assert 18 <= age <= 86, f"Invalid age {age}, must be between 18 and 86 for original ACS norms"
        subset = subset[subset['age'] == age]

        if gender is not None:
            subset = subset[subset['gender'] == gender]
        if education is not None:
            subset = subset[subset['education'] == education]
        if computer_experience is not None and not pd.isna(computer_experience):
            if isinstance(computer_experience, (tuple, list)):
                subset = subset[subset['computer_experience'].isin(computer_experience)]
            else:
                subset = subset[subset['computer_experience'] == computer_experience]

        # get average over all dimensions that are None
        value_cols = [c for c in subset.columns if c.isdigit()]
        subset = subset.mean()

        subset['mean_raw_score'] = int(subset[value_cols].index[subset[value_cols].abs().argmin()])
        subset['1_std_band'] = (int(subset[value_cols].index[(subset[value_cols]-1).abs().argmin()]), int(subset[value_cols].index[(subset[value_cols]+1).abs().argmin()]))
        subset['2_std_band'] = (int(subset[value_cols].index[(subset[value_cols]-2).abs().argmin()]), int(subset[value_cols].index[(subset[value_cols]+2).abs().argmin()]))
        subset['3_std_band'] = (int(subset[value_cols].index[(subset[value_cols]-3).abs().argmin()]), int(subset[value_cols].index[(subset[value_cols]+3).abs().argmin()]))
        subset['4_std_band'] = (int(subset[value_cols].index[(subset[value_cols]-4).abs().argmin()]), int(subset[value_cols].index[(subset[value_cols]+4).abs().argmin()]))

        if outcome_variable in self.msec_cols:
            subset['mean_raw_score'] = subset['mean_raw_score'] * 1000
            subset[['1_std_band', '2_std_band', '3_std_band', '4_std_band']] = \
                subset[['1_std_band', '2_std_band', '3_std_band', '4_std_band']].map(lambda r: (r[0] * 1000, r[1] * 1000))

        subset = subset.drop(index=value_cols)

        return subset


    def calculate_norm_scores(self, data):
        missing_norm_cols = [c for c in self.demographic_input_cols if c not in data.columns]
        assert len(missing_norm_cols) == 0, f"Demographic column {missing_norm_cols} missing for norm calculation"

        return self._calculate_norm_scores(data)


    def norm_bands(self, demographic_data, outcome_variable):
        assert all([c in demographic_data.columns for c in self.demographic_input_cols])

        response = self._prepare_data_for_acs_norms(demographic_data.copy())

        norm_data = self._load_norm_data([outcome_variable])
        def get_bands(row):
            return self._get_norm_bands(norm_data[outcome_variable], outcome_variable, age=row['age'], gender=row['gender'],
                                        education=row['education'], computer_experience=row['computer_experience'])
        norm_bands = response.apply(get_bands, axis=1)
        #response = pd.concat([response, norm_bands], axis=1)

        return norm_bands





class RegressionBasedACSNormDataCalculator(ACSNormDataCalculator):

    def __init__(self, CONSTANTS, version_logic=1):
        self.CONSTANTS = CONSTANTS
        self.version_logic = version_logic

        if version_logic == 1:
            # version 1: original logic with age, gender, mouse_type. This was motivated by a bias analysis
            self.demographic_input_cols = ['age', 'gender_unified', 'mouse_type']  # should be given as input
            self.independent_cols_for_regression = ['age', 'gender', 'mouse_type', 'intercept']  # will be used by regression

            # norms were defined by jupyter notebook /luha-prolific-study/analyses/2024/kw29/create_ACS_norms.ipynb
            self.norm_definitions = pd.read_csv(os.path.join(self.CONSTANTS.RESOURCES_DIR, 'ACS_norms_regression.csv'))

        elif version_logic == 2:
            # version 2: new logic with the default demographic variables, to be more consistent with the rest of the paper
            self.demographic_input_cols = ['age', 'gender_unified', 'education_binary', 'country']  # should be given as input
            self.independent_cols_for_regression = self.demographic_input_cols + ['intercept']  # will be used by regression

            # norms were defined by jupyter notebook /luha-prolific-study/analyses/kw02/acs_norms/02_create_ACS_norms_new_logic.ipynb.ipynb
            self.norm_definitions = pd.read_csv(os.path.join(self.CONSTANTS.RESOURCES_DIR, 'ACS_norms_regression_v2.csv'))

        else:
            raise ValueError()


        self.acs_variables_for_norms = self.CONSTANTS.ACS_MAIN_OUTCOME_VARIABLES_EXTENDED


    def _prepare_data(self, data):
        print("... [ACS regression-based norms]  Prepare data")

        demographic_cols = ['study_submission_id'] + self.demographic_input_cols if 'study_submission_id' in data.columns else self.demographic_input_cols
        outcome_cols = [c for c in data.columns if c in self.acs_variables_for_norms]
        data = data[demographic_cols + outcome_cols].copy()

        if self.version_logic == 1:
            data['gender'] = data['gender_unified'].apply(lambda x: 0 if x == 'f' else 1 if x == 'm' else None)
            data = data.drop(columns=['gender_unified'])
            data['mouse_type'] = data['mouse_type'].apply(
                lambda x: 0 if x == 'standard' else 1 if x == 'trackpad' else None)

            # fill na values for gender and mouse_type with 0.5. In the regression, this means that
            # the norm value between male and female / standard and trackpad is used
            gender_na = np.sum(data['gender'].isna())
            mouse_type_na = np.sum(data['mouse_type'].isna())
            if gender_na > 0:
                logging.warning(f"There are {gender_na} ({gender_na / data.shape[0] * 100:.1f}%) rows with NA gender, i.e. no valid gender provided. We will use the mean of male and female")
            if mouse_type_na > 0:
                logging.warning(f"There are {mouse_type_na} ({mouse_type_na / data.shape[0] * 100:.1f}%) rows with NA mouse_type, i.e. no valid mouse_type provided. We will use the mean of standard and trackpad")
            data.loc[:, ['gender', 'mouse_type']] = data.loc[:, ['gender', 'mouse_type']].fillna(0.5)


        elif self.version_logic == 2:
            demographics_prepared, _ = prepare_demographics(data[self.demographic_input_cols])
            data.loc[:, self.demographic_input_cols] = demographics_prepared.loc[:, self.demographic_input_cols]

        # Age: "Age" (coded in whole years)
        data.loc[:, 'age'] = data.loc[:, 'age'].round(0).astype(int)

        return data

    def _get_norm_params(self, outcome):
        # norm parameters / definitions (regression slopes and intercept) for a given outcome variable
        row = self.norm_definitions.query(f"outcome == '{outcome}'")
        assert row.shape[0] == 1, f"Should have one norm definition for {outcome}, but there are {row.shape[0]}"
        row = row.iloc[0]
        std_residuals = row.std_residuals
        params = row.drop(index=['outcome', 'std_residuals']).to_dict()
        return params, std_residuals

    def _calculate_norm_scores(self,  data):
        data = self._prepare_data(data)
        data['intercept'] = 1  # intercept for linear regression

        outcomes_in_data = [c for c in data.columns if c in self.acs_variables_for_norms]
        assert len(outcomes_in_data) > 0, f"No outcomes for z-score calculation in data. Should be at least one of {self.acs_variables_for_norms}, but there are only {list(data.columns)}"
        for outcome in self.acs_variables_for_norms:
            print("... [ACS regression-based norms] Outcome: ", outcome)

            params, std_residuals = self._get_norm_params(outcome)
            params_string = {key: f"{val:.2f}" for key, val in params.items()}

            def pred(X):
                return X.dot(pd.Series(params))

            residuals = pred(data[self.independent_cols_for_regression]) - data[outcome]
            data[f'{outcome}_Z'] = residuals / std_residuals

        return data

    def calculate_norm_scores(self, data):
        missing_norm_cols = [c for c in self.demographic_input_cols if c not in data.columns]
        assert len(missing_norm_cols) == 0, f"Demographic column {missing_norm_cols} missing for norm calculation"

        return self._calculate_norm_scores(data)

    def norm_bands(self, demographic_data, outcome_variable):
        assert all([c in demographic_data.columns for c in self.demographic_input_cols])

        params, std_residuals = self._get_norm_params(outcome_variable)

        def pred(X):
            return np.array(X.dot(pd.Series(params)).astype(np.float64))

        response = self._prepare_data(demographic_data.copy())
        response['intercept'] = 1  # intercept for linear regression

        response['norm_mean'] = pred(response[self.independent_cols_for_regression])
        response['norm_std'] = std_residuals

        return response




class RobustRegressionBasedACSNormDataCalculator(RegressionBasedACSNormDataCalculator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.version_logic != 1:
            raise NotImplementedError("RobustRegressionBasedACSNormDataCalculator only implements version_logic==1. To fix this, create the respective norms in /Users/jheitz/git/luha-prolific-study/analyses/kw02/acs_norms/02_create_ACS_norms_new_logic.ipynb (with reference to the 2024/kw29 version")

        # norms were defined by jupyter notebook /luha-prolific-study/analyses/kw27/create_ACS_norms.ipynb
        self.norm_definitions = pd.read_csv(os.path.join(self.CONSTANTS.RESOURCES_DIR, 'ACS_norms_robust_regression.csv'))

