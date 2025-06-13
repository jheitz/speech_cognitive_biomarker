import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from util.helpers import prepare_demographics
import logging

class FeatureNormDataCalculator:
    """
    This class is modelled after RegressionBasedACSNormDataCalculator
    It should provide norm data for (linguistic) features, based on a regression model taking some
    demographic information into account.
    This is used to detect (and remove) outliers
    """

    def __init__(self, CONSTANTS):
        self.CONSTANTS = CONSTANTS
        self.demographic_cols = ['age', 'gender_unified', 'education_binary', 'country']
        self.norms = None

    def _create_norms_for_col(self, col, data, plot=False):
        print(f"\nIteratively creating linear regression norms for {col}")

        new_outliers = np.inf
        previous_outliers = None
        n_outliers = -1

        data_here = data[self.demographic_cols + [col]].copy()
        n_na = data_here.shape[0] - data_here.dropna().shape[0]
        print(f"Removing # NA for column {col}: {n_na} ({(n_na) / data_here.shape[0] * 100:.2f}%)")
        data_here = data_here.dropna()
        data_here['residuals'] = 0  # residuals from model training -> to iteratively define outliers and remove them
        data_here['z_score'] = 0  # z-score, based on residuals
        data_here['is_outlier'] = False
        data_here['intercept'] = 1  # adding constant for linear regression

        print("Removing # outliers:", end=" ")
        while new_outliers > 0:
            # remove outliers based on previous fit
            data_here['is_outlier_before'] = data_here['is_outlier']
            data_here['is_outlier'] = np.where(np.abs(data_here['z_score']) > 4, True, False)
            previous_outliers = n_outliers
            n_outliers = np.sum(data_here['is_outlier'])
            new_outliers = n_outliers - previous_outliers
            print(n_outliers, end=" ")

            if new_outliers > data_here.shape[0] * 0.5:
                print(f"Outlier removal failed: More than 50% of all samples are outliers ({n_outliers} / {data_here.shape[0]}). Something must be wrong with the distribution of values / the estimated norm model?")
                print("std_residuals", std_residuals)
                print("Distribution of predictions for non-outlier samples in the last run", model.predict(data_here.query("not is_outlier_before")[self.demographic_cols + ['intercept']]).describe())
                raise ValueError(f"Outlier removal failed")

            # train linear regression without outliers
            X_train = data_here.query("is_outlier == False")[self.demographic_cols + ['intercept']]
            y_train = data_here.query("is_outlier == False")[col]

            model = sm.OLS(y_train, X_train).fit()

            # calculate residuals, standard deviation of residuals, and z-scores
            data_here['prediction'] = model.predict(data_here[self.demographic_cols + ['intercept']])
            data_here['residuals'] = data_here[col] - model.predict(data_here[self.demographic_cols + ['intercept']])
            std_residuals = model.resid.std()
            assert np.abs(np.mean(data_here.query("is_outlier == False")['residuals'])) < 1e-10
            data_here['z_score'] = data_here['residuals'] / std_residuals

        print("")
        outlier_list = f"{list(data_here.query('is_outlier').index)[:6]}{'...' if len(data_here.query('is_outlier').index) > 6 else ''}"
        print(f"Final model based on {X_train.shape[0]} training instances (having removed {n_outliers} outliers ({outlier_list}) and {n_na} NA values from a total of {data.shape[0]})")

        # print(model.summary())
        params = model.params.to_dict()
        params_string = {key: f"{val:.3f}" for key, val in params.items()}
        print(f"Obtained linear fit parameters: {params_string}")

        if plot:
            fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

            plotting_range_for_col = {
                'age': (60, 90),
                'gender_unified': (0.5, 0.5),
                'education_binary': (0.5, 0.5),
                'country': (0.5, 0.5),
                'const': (1, 1)
            }
            demographic_cols_for_plotting = self.demographic_cols + ['const']
            X_plot = pd.DataFrame([plotting_range_for_col[col][0] for col in demographic_cols_for_plotting],
                                  [plotting_range_for_col[col][1] for col in demographic_cols_for_plotting],
                                  columns=demographic_cols_for_plotting)
            ax1.plot(X_plot['age'], model.predict(X_plot), label="avg gender, avg mouse_type", color="blue")
            ax1.fill_between(X_plot['age'], model.predict(X_plot) - std_residuals, model.predict(X_plot) + std_residuals,
                             label="avg demographics", alpha=0.1, color="blue")
            ax1.fill_between(X_plot['age'], model.predict(X_plot) - 2 * std_residuals,
                             model.predict(X_plot) + 2 * std_residuals, alpha=0.1, color="blue")
            # ax1.fill_between(X_plot['age'], model.predict(X_plot)-3*std_residuals, model.predict(X_plot)+3*std_residuals, alpha=0.1, color="blue")
            # ax1.fill_between(X_plot['age'], model.predict(X_plot)-4*std_residuals, model.predict(X_plot)+4*std_residuals, alpha=0.1, color="blue")
            ax1.plot(data_here.query("is_outlier").age, data_here.query("is_outlier")[col], '.', markersize=2, color='red',
                     label=f"Outliers (n={data_here.query('is_outlier').shape[0]})")
            ax1.plot(data_here.query("not is_outlier").age, data_here.query("not is_outlier")[col], '.', markersize=2,
                     color='green', label=f"Non-Outliers (n={data_here.query('not is_outlier').shape[0]})")
            ax1.set_title(f"Norm for {col}")
            ax1.legend()

            ax3.plot(data_here.query("not is_outlier").age, data_here.query("not is_outlier")[col] - model.predict(X_train),
                     'o', markersize=3, alpha=0.1)
            ax3.set_title("Residual scatter plot for non-outliers")

            plt.show()

        return {**params, 'outcome': col, 'std_residuals': std_residuals}

    def _prepare_data(self, data):
        #print("... [feature regression-based norms]  Prepare data")
        feature_names = [c for c in data.columns if c not in self.demographic_cols]
        data = data[self.demographic_cols + feature_names].copy()

        # check that dtypes of demographic columns are numeric
        # this is to ensure that the demographics have been prepared already (gender=0/1, education=0/1, etc.)
        # if not, it cannot be used here.
        all([pd.api.types.is_numeric_dtype(dtype) for dtype in data[self.demographic_cols].dtypes]), \
            "Demographic columns need to be prepared before running outlier removal. They should be dtype numeric"
        #data[self.demographic_cols], _ = prepare_demographics(data[self.demographic_cols])

        # fill na values for gender and education with 0.5. In the regression, this means that
        # the norm value between male and female / standard and trackpad is used
        gender_na = np.sum(data['gender_unified'].isna())
        education_na = np.sum(data['education_binary'].isna())
        country_na = np.sum(data['country'].isna())
        if gender_na > 0:
            logging.warning(f"There are {gender_na} ({gender_na / data.shape[0] * 100:.1f}%) rows with NA gender, i.e. no valid gender provided. We will use the mean of male and female")
        if education_na > 0:
            logging.warning(f"There are {education_na} ({education_na / data.shape[0] * 100:.1f}%) rows with NA education, i.e. no valid education provided. We will use the mean")
        if country_na > 0:
            logging.warning(f"There are {country_na} ({country_na / data.shape[0] * 100:.1f}%) rows with NA country, i.e. no valid country provided. We will use the mean")
        data.loc[:, ['gender_unified', 'education_binary', 'country']] = data.loc[:, ['gender_unified', 'education_binary', 'country']].fillna(0.5)

        # Age: "Age" (coded in whole years)
        data.loc[:, 'age'] = data.loc[:, 'age'].round(0).astype(int)

        return data

    def _get_norm_params(self, outcome):
        # norm parameters / definitions (regression slopes and intercept) for a given outcome variable
        assert self.norms is not None
        row = self.norms.query(f"outcome == '{outcome}'")
        assert row.shape[0] == 1, f"Should have one norm definition for {outcome}, but there are {row.shape[0]}"
        row = row.iloc[0]
        std_residuals = row.std_residuals
        params = row.drop(index=['outcome', 'std_residuals']).to_dict()
        return params, std_residuals

    def _calculate_norm_scores(self, data):
        data = self._prepare_data(data)
        feature_names = [c for c in data.columns if c not in self.demographic_cols]
        data['intercept'] = 1  # intercept for linear regression

        outcomes_in_data = [c for c in data.columns if c in feature_names]
        assert len(
            outcomes_in_data) > 0, f"No outcomes for z-score calculation in data. Should be at least one of {feature_names}, but there are only {list(data.columns)}"
        for outcome in feature_names:
            print("... [feature regression-based norms] Outcome: ", outcome)

            params, std_residuals = self._get_norm_params(outcome)
            params_string = {key: f"{val:.2f}" for key, val in params.items()}

            def pred(X):
                return X.dot(pd.Series(params))

            residuals = pred(data[self.demographic_cols + ['intercept']]) - data[outcome]
            data[f'{outcome}_Z'] = residuals / std_residuals

        return data

    def create_norms(self, data):
        missing_norm_cols = [c for c in self.demographic_cols if c not in data.columns]
        assert len(missing_norm_cols) == 0, f"Demographic column {missing_norm_cols} missing for norm calculation"

        data = self._prepare_data(data)

        feature_names = [c for c in data.columns if c not in self.demographic_cols]
        norms = [self._create_norms_for_col(col, data) for col in feature_names]
        self.norms = pd.DataFrame(norms)

    def calculate_norm_scores(self, data):
        if self.norms is None:
            logging.warning("Calculate norms before calculating norms")

        missing_norm_cols = [c for c in self.demographic_cols if c not in data.columns]
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