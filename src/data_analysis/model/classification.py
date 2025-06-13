import numpy as np
import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import shap

from data_analysis.dataloader.dataset import DatasetType
from data_analysis.model.base_model import BaseModel
from util.helpers import python_to_json, prepare_demographics, calculate_cohens_d
from data_analysis.data_preprocessing.feature_outlier_removal_imputation import FeatureOutlierRemovalImputation
from data_analysis.data_preprocessing.feature_standardizer import FeatureStandardizer


class Classification(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__("Classification", *args, **kwargs)

        # specificity is the recall of the negative class
        specificity = lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, pos_label=0)
        self.binary_metrics = {
            'accuracy': metrics.accuracy_score,
            'balanced_accuracy': metrics.balanced_accuracy_score,
            'f1': metrics.f1_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'specificity': specificity,
            'confusion_matrix': metrics.confusion_matrix
        }

        # Effect size (Cohen's d, and correlation coefficient r) of the prediction with the true label
        # This should be comparable to typical effect sizes in Psychology
        # Consider e.g. Meyer et al, 2001, Psychological testing and psychological assessment: A review of evidence and issues
        def cohens_d(y_true, y_pred):
            y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
            group1 = y_pred_arr[y_true_arr == 1]
            group2 = y_pred_arr[y_true_arr == 0]
            assert len(group1) + len(group2) == len(y_pred)
            return calculate_cohens_d(group1, group2)
        # consider regression.py for a discussion of the difference of explained_variance_score and r^2
        self.continous_metrics = {
            'roc_auc': metrics.roc_auc_score,
            'effect_size_cohens_d': cohens_d,
            'effect_size_r': lambda y_true, y_pred: np.corrcoef(y_pred, y_true)[0, 1],
            'r^2': lambda y_true, y_pred: metrics.r2_score(y_true, y_pred),
            'explained_variance': lambda y_true, y_pred: metrics.explained_variance_score(y_true, y_pred),
            # 'roc_curve': metrics.roc_curve
        }

        self.target_variable = None
        if self.config.config_model.target_variable is not None:
            self.target_variable = self.config.config_model.target_variable
        assert self.target_variable is not None, f"Target variable missing"

        try:
            self.model_name = self.config.config_model.model_name
        except (AttributeError, KeyError):
            self.model_name = "SVC"
        assert self.model_name in ['RandomForest', 'LogisticRegression', 'SVC']

        if self.model_name == 'SVC':
            try:
                self.svc_parameters = vars(self.config.config_model.svc_parameters)
            except (AttributeError, KeyError):
                self.svc_parameters = dict(kernel='rbf', C=0.1)
            print(f"Using SVC parameters: {self.svc_parameters}")


        valid_classification_labels = ['negative_outliers', 'high_low_performers', 'negative_outliers_simple', 'classification_target']
        try:
            self.classification_labels = self.config.config_model.classification_labels
        except (AttributeError, KeyError):
            self.classification_labels = "high_low_performers"
        assert self.classification_labels in valid_classification_labels

        try:
            self.cv_splits = self.config.config_model.cv_splits
        except (AttributeError, KeyError):
            self.cv_splits = 10

        valid_data_preprocessors = ['Outlier Removal and Imputation', 'Feature Standardizer']
        try:
            self.data_preprocessors = self.config.data_preprocessors
        except (AttributeError, KeyError):
            self.data_preprocessors = []
        assert all([p in valid_data_preprocessors for p in
                    self.data_preprocessors]), f"Invalid data preprocessor: {[p for p in self.data_preprocessors if p not in valid_data_preprocessors]}"

        for p in self.data_preprocessors:
            if p == 'Outlier Removal and Imputation':
                self.feature_outlier_removal_imputation = FeatureOutlierRemovalImputation(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            elif p == 'Feature Standardizer':
                self.feature_standardizer = FeatureStandardizer(config=self.config, constants=self.CONSTANTS, run_parameters=self.run_parameters)
            else:
                raise ValueError(f"Invalid data preprocessor {p}")

        try:
            self.calculate_shap = self.config.config_model.calculate_shap
        except (AttributeError, KeyError):
            if self.model_name == 'SVC':
                self.calculate_shap = False  # because it's slow
            else:
                self.calculate_shap = True

        print(f"Using target_variable={self.target_variable}, model_name={self.model_name}, cv_splits={self.cv_splits},",
              f"data_preprocessors={self.data_preprocessors}, classification_labels={self.classification_labels}")


    def _log_results(self, message, filename):
        print(message)
        with open(os.path.join(self.run_parameters.results_dir, filename), "a") as f:
            f.write(message)
            f.write("\n")

    def prepare_data(self):
        if self.model_name in ['LinearRegression', 'Lasso']:
            assert 'Feature Standardizer' in self.config.data_preprocessors, "Linear Regression requires Standardization of features"

    def _write_correlation_matrix(self, features: pd.DataFrame, file_postfix=""):
        n_features = features.shape[1]
        f = plt.figure(figsize=(n_features*.3+4, n_features*.3))
        correlation_matrix = features.corr()
        plt.matshow(correlation_matrix, fignum=f.number, cmap=plt.get_cmap("bwr"), vmin=-1, vmax=1)
        plt.xticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=14,
                   rotation=45, ha='left', rotation_mode='anchor')
        plt.yticks(range(features.select_dtypes(['number']).shape[1]), features.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, f"feature_correlation{file_postfix}.png"))
        plt.close()

        with open(os.path.join(self.run_parameters.results_dir, f"feature_correlation{file_postfix}.txt"), "w") as file:
            file.write(python_to_json(correlation_matrix))

    def _write_feature_distribution(self, features, labels):
        def plot_distribution(features_low, features_high, all_features):
            feature_names = all_features.columns
            positions_low = [i * 3 for i in range(len(feature_names))]
            positions_high = [i * 3 + 1 for i in range(len(feature_names))]
            tick_positions = [i * 3 + 0.5 for i in range(len(feature_names))]

            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            # pooled std, which should be used for Cohen's d (cf. https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)
            low_var, high_var = features_low.var(axis=0, ddof=1), features_high.var(axis=0, ddof=1)
            low_n, high_n = features_low.shape[0], features_high.shape[0]
            pooled_std = np.sqrt(((low_n - 1) * low_var + (high_n - 1) * high_var) / (low_n + high_n - 2))

            # normalized values
            low_normalized = (features_low - mean) / pooled_std
            high_normalized = (features_high - mean) / pooled_std

            # diff
            diff = low_normalized.mean() - high_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            # sorted
            low_normalized = low_normalized[sorted_features]
            high_normalized = high_normalized[sorted_features]
            diff = diff[sorted_features]

            fig, (ax, ax3, ax2) = plt.subplots(3, 1, figsize=(len(feature_names) / 4 + 2, 10), height_ratios=(1, 1, 2))
            fig.suptitle(f'Distribution of normalized feature values low vs. high {self.target_variable}')
            ax2.boxplot(low_normalized.values, sym="", positions=positions_low,
                                         showmeans=True, meanline=True,
                                         patch_artist=True, boxprops=dict(facecolor="#ffa8a8"))
            ax2.boxplot(high_normalized.values, sym="", positions=positions_high,
                                        showmeans=True, meanline=True,
                                        patch_artist=True, boxprops=dict(facecolor="#a8d9ff"))
            ax2.set_xticks(tick_positions, sorted_features, rotation=45, ha='right', rotation_mode='anchor')
            ax2.set_ylabel("Normalized value of feature")
            ax2.axhline(0, linestyle=":", color="k", alpha=0.5)

            blue_patch = mpatches.Patch(color='#ffa8a8', label='Low performers')
            green_patch = mpatches.Patch(color='#a8d9ff', label='High performers')
            ax2.legend(handles=[blue_patch, green_patch])

            ax.plot(tick_positions, np.abs(diff), ".-", color="k", linewidth=1)
            ax.set_xticks([], [])
            ax.set_ylabel("Effect size\n[Cohen's d]")
            #ax.set_ylim([0, 1.6])
            ax.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])

            # Calculate correlation coefficient (r) as a measure of effect size
            # This can either be done from Cohen's d (based on Lakens, 2013, Calculating and reporting effect sizes..., Front. Psychol.)
            # Or calculated directly using the pearson correlation between the feature value and the label
            cohen_d = np.abs(diff)
            total_n = all_features.shape[0]
            r_from_cohens_d = cohen_d / np.sqrt(cohen_d ** 2 + (total_n ** 2 - 2 * total_n) / (low_n * high_n))
            features_for_correlation = pd.concat((features_low, features_high), axis=0)
            features_for_correlation['label'] = np.concatenate((np.zeros(low_n), np.ones(high_n)))
            r_direct = features_for_correlation.corr(method='pearson').loc['label']
            r_direct = r_direct[sorted_features]
            ax3.plot(tick_positions, r_from_cohens_d, ".-", linewidth=1, label="From Cohen's d")
            ax3.plot(tick_positions, np.abs(r_direct), ".-", linewidth=1, label="Calculated directly")
            ax3.legend()
            ax3.set_xticks([], [])
            ax3.set_ylabel("Effect size\n[Correlation coefficient r]")
            # ax3.set_ylim([0, 1.6])
            ax3.set_xlim(ax2.get_xlim()[0], ax2.get_xlim()[1])

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution.png"))
            plt.close()

        def plot_individual_distribution(features_low, features_high, all_features, figsize_individual):
            feature_names = all_features.columns

            n_cols = int(np.ceil(np.sqrt(len(feature_names))))
            n_rows = int(np.ceil(len(feature_names) / n_cols))
            print("n_rows, n_cols", n_rows, n_cols)
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(n_cols * figsize_individual[0], n_rows * figsize_individual[1]))

            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1 or n_cols == 1:
                axes = [axes]

            # mean and std, normalize values, calculate difference, sort features accordingly
            mean = all_features.mean(axis=0)
            std = all_features.std(axis=0)

            low_normalized = (features_low - mean) / std
            high_normalized = (features_high - mean) / std

            diff = low_normalized.mean() - high_normalized.mean()
            sorted_features = list(diff.sort_values(key=abs, ascending=False).index)

            features_low = features_low[sorted_features]
            features_high = features_high[sorted_features]

            for feature, ax in zip(sorted_features, [col for row in axes for col in row]):
                ax.boxplot(features_low[feature], sym="", positions=[0],
                           showmeans=True, meanline=True)
                ax.boxplot(features_high[feature], sym="", positions=[1],
                           showmeans=True, meanline=True)
                ax.set_xticks([0, 1], ['Dementia', 'Control'], rotation=45, ha='right', rotation_mode='anchor')
                ax.set_ylabel("Feature Value")
                ax.set_title(feature, fontsize=10)
                ax.axhline(mean[feature], linestyle=":", color="k", alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(self.run_parameters.results_dir, "feature_distribution_individual.png"))
            plt.close()

        df = features.copy()
        df['label'] = labels
        all_features = df.drop(columns=['label'])
        features_low = df.query("label == 1").drop(columns=['label'])
        features_high = df.query("label == 0").drop(columns=['label'])

        plot_distribution(features_low, features_high, all_features)
        plot_individual_distribution(features_low, features_high, all_features, figsize_individual=(2, 2))

    def _get_target_variable(self):
        demographics, _ = prepare_demographics(self.data.demographics)

        if self.data.type == DatasetType.ADReSS:
            potential_target_df = pd.concat([
                pd.DataFrame({'classification_target': self.data.classification_target}),
                demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
            ], axis=1)
        else:
            potential_target_df = pd.concat([
                self.data.factor_scores_theory.reset_index(drop=True),
                self.data.cognitive_overall_score.reset_index(drop=True),
                self.data.factor_scores.reset_index(drop=True),
                self.data.acs_outcomes_imputed.reset_index(drop=True),
                self.data.language_task_scores.reset_index(drop=True),
                demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
            ], axis=1)

        return potential_target_df[self.target_variable]

    def _plot_target_variable_final(self):
        target_variable = self._get_target_variable()
        classification_target = self.data.classification_target

        pd.Series(classification_target).value_counts().to_csv(os.path.join(self.run_parameters.results_dir, "classification_target_distribution.csv"))

        plt.hist(target_variable[classification_target == 0], bins=20, label=f"classification_target=0 (n={target_variable[classification_target == 0].shape[0]})")
        plt.hist(target_variable[classification_target == 1], bins=20, label=f"classification_target=1 (n={target_variable[classification_target == 1].shape[0]})")
        plt.legend()
        plt.title(f"Distribution of classification target based on {self.target_variable}")
        plt.xlabel(self.target_variable)
        plt.ylabel("# samples")
        plt.savefig(os.path.join(self.run_parameters.results_dir, f"target_variable_distribution.png"))
        plt.close()

    def _bin_by_target_var_and_extract_extremes(self):
        # Prepare the dataset by binning the data based on the target variable (high / middle / low performers)
        # dropping the middle bin, and making it a binary classification task

        target_continuous = self._get_target_variable()
        bin_assignment, bins = pd.qcut(target_continuous, q=3, labels=['low', 'middle', 'high'], retbins=True)
        print(f"Binning the target variable {self.target_variable} into 3 bins: {bins}")
        print("Binned statistics:", bin_assignment.value_counts().to_dict())

        # prepare classification target -> binary version for low=0, high=1
        self.data.classification_target = np.array(bin_assignment.replace({'low': 0, 'middle': None, 'high': 1}))

        if len(target_continuous.value_counts()) < 50:
            for bin_label in ['low', 'middle', 'high']:
                vals = target_continuous[bin_assignment == bin_label]
                val_counts = vals.value_counts()
                plt.bar(val_counts.keys(), val_counts.values, label=f"{bin_label} (n={len(vals)})")
        else:
            histogram_bin_width = (bins[-1] - bins[0]) / 50
            for bin_start, bin_end, bin_label in zip(bins[:-1], bins[1:], ['low', 'middle', 'high']):
                print(bin_end, bin_start, bin_label)
                histogram_n_bin_here = round((bin_end - bin_start) / histogram_bin_width)
                vals = target_continuous[bin_assignment == bin_label]
                plt.hist(vals, bins=histogram_n_bin_here, label=f"{bin_label} (n={len(vals)})")

        # [plt.axvline(c) for c in bins]
        plt.legend()
        plt.title(f"Target variable binning\nBoundaries: {bins}")
        plt.xlabel(self.target_variable)
        plt.ylabel("# samples")
        plt.savefig(os.path.join(self.run_parameters.results_dir, f"target_variable_binned.png"))
        plt.close()

        index_low_high = bin_assignment[bin_assignment.isin(['low', 'high'])].index
        dataset_without_middle = self.data.subset_from_indices(index_low_high)

        self.data = dataset_without_middle
        print("New dataset for classification:", self.data)

        self._plot_target_variable_final()

    def _create_negative_outlier_classification_target(self):
        # Prepare the dataset by extracting the classification targets (labels) which are either
        # a) negative outliers, taking demographics into account
        # b) the rest (non-outliers)

        cognitive_outliers_all = json.load(open(self.CONSTANTS.COGNITIVE_NEGATIVE_OUTLIERS, "r"))
        assert self.target_variable in cognitive_outliers_all.keys(), f"Target variable {self.target_variable} not in cognitive outlier list at {self.CONSTANTS.COGNITIVE_NEGATIVE_OUTLIERS}"
        cognitive_outlier_sample_names = cognitive_outliers_all[self.target_variable]

        # assign target based on whether or not each sample is an outlier according to the above list
        self.data.classification_target = np.where(pd.Series(self.data.sample_names).isin(cognitive_outlier_sample_names), 1, 0)

        print("Classification target (negative outliers) stats:", pd.Series(self.data.classification_target).value_counts())

        self._plot_target_variable_final()

    def _create_negative_outlier_simple_classification_target(self):
        # Prepare the dataset by extracting the classification targets (labels) which are either
        # a) negative outliers, independent of demographics
        # b) the rest (non-outliers)

        target_continuous = self._get_target_variable()
        mean, std = np.mean(target_continuous), np.std(target_continuous)

        # assign target based on whether or not each sample is an outlier according to the above list
        self.data.classification_target = np.where(target_continuous < mean - std*1.96, 1, 0)

        print("Classification target (negative outliers) stats:", pd.Series(self.data.classification_target).value_counts())

        self._plot_target_variable_final()

    def _plot_roc(self, file_path, predictions_flat, labels_flat):
        plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        # draw thicker line for total results
        fpr, tpr, _ = metrics.roc_curve(labels_flat, predictions_flat)
        auroc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=r"Overall ROC (area = %0.2f)" % auroc, lw=4)

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve (n={len(predictions_flat)})')
        plt.legend(loc='best')
        plt.savefig(file_path)
        plt.close()

    def _plot_precision_recall(self, file_path, predictions_flat, labels_flat):
        plt.figure(figsize=(10, 10))

        # draw thicker line for total results
        precisions, recalls, thresholds = metrics.precision_recall_curve(labels_flat, predictions_flat)
        #auc = metrics.auc(precisions, recalls)
        no_skill = np.sum(labels_flat) / len(labels_flat)
        plt.plot(recalls, precisions, label="Precision-Recall", lw=1, marker=".")
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', c='k', alpha=0.5)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve (n={len(predictions_flat)})')
        plt.legend(loc='best')
        plt.savefig(file_path)
        plt.close()

    def _preprocess_data(self, X_train, demographics_train, sample_names_train, X_test, demographics_test, sample_names_test):
        if 'Outlier Removal and Imputation' in self.data_preprocessors:
            # remove and impute feature outliers
            # note that we take care not to have data leakage here, by fitting on training data only
            self.feature_outlier_removal_imputation.fit(X_train, demographics_train, sample_names_train)
            X_train = self.feature_outlier_removal_imputation.transform(X_train, demographics_train, sample_names_train)
            X_test = self.feature_outlier_removal_imputation.transform(X_test, demographics_test, sample_names_test)

        if 'Feature Standardizer' in self.data_preprocessors:
            # now we standardize the feature values (mean 0, std 1)
            self.feature_standardizer.fit(X_train)
            X_train = self.feature_standardizer.transform(X_train)
            X_test = self.feature_standardizer.transform(X_test)

        return X_train, X_test

    def run(self):
        if self.classification_labels == 'high_low_performers':
            # create binary classification label by binning dataset and keeping low and high performers, dropping middle
            self._bin_by_target_var_and_extract_extremes()
        elif self.classification_labels == 'negative_outliers':
            # create binary classification label according to norm-based negative outlier detection
            self._create_negative_outlier_classification_target()
        elif self.classification_labels == 'negative_outliers_simple':
            # create binary classification label according to norm-based negative outlier detection
            self._create_negative_outlier_simple_classification_target()
        elif self.classification_labels == 'classification_target':
            # take directly from data
            pass
        else:
            raise ValueError(f"Invalid classification labels: {self.classification_labels}")

        if self.data.type == DatasetType.ADReSS:
            demographics = self.data.demographics
        else:
            demographics, _ = prepare_demographics(self.data.demographics)

        if self.data.type == DatasetType.ADReSS:
            classification_df = pd.concat([
                self.data.features.reset_index(drop=True),
                demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
                pd.DataFrame({'classification_target': self.data.classification_target}).reset_index(),
            ], axis=1)
            classification_df['mean_composite_cognitive_score_binned'] = 0  # for compatibility
            classification_df['age_binned'] = pd.qcut(classification_df['age'], q=2)
            stratification_columns = [c for c in demographics.columns if c != 'age'] + ['age_binned', 'classification_target']
        else:
            classification_df = pd.concat([
                self.data.features.reset_index(drop=True),
                demographics.reset_index(drop=True),  # demographics are necessary for stratified CV
                pd.DataFrame({'classification_target': self.data.classification_target}).reset_index(),
                self.data.mean_composite_cognitive_score.reset_index(drop=True), # mean composite score, necessary for stratified CV
            ], axis=1)
            classification_df['mean_composite_cognitive_score_binned'] = pd.qcut(classification_df['mean_composite_cognitive_score'], q=4)
            classification_df['age_binned'] = pd.qcut(classification_df['age'], q=5)
            stratification_columns = [c for c in demographics.columns if c != 'age'] + ['age_binned', 'mean_composite_cognitive_score_binned', 'classification_target']

        classification_df['sample_name'] = np.array(self.data.sample_names)

        X_cols = list(self.data.features.columns)
        assert self.target_variable not in X_cols, f"Target variable {self.target_variable} is in X_cols ({X_cols})"

        # dropping nan rows
        non_nan_filter = ~classification_df.isna().any(axis=1)
        nan_row_sample_names = classification_df.loc[~non_nan_filter].sample_name
        n_rows_before = classification_df.shape[0]
        if np.sum(~non_nan_filter) > 0:
            print(f"\n\nATTENTION: null values in {np.sum(~non_nan_filter)} rows / {n_rows_before} (sample names {nan_row_sample_names.to_list()})\n\n")
        classification_df = classification_df.dropna()
        classification_df = classification_df.reset_index(drop=True)

        columns_to_keep = X_cols + demographics.columns.to_list() + ['classification_target', 'sample_name', 'mean_composite_cognitive_score_binned', 'age_binned']
        columns_to_keep_no_duplicates = list(set(columns_to_keep))
        classification_df = classification_df[columns_to_keep_no_duplicates]
        classification_df['test_split'] = np.ones((classification_df.shape[0], )) * -1
        assert np.all(classification_df.index == np.arange(classification_df.shape[0]))  # make sure index is reset

        self._write_correlation_matrix(classification_df[X_cols])
        self._write_feature_distribution(classification_df[X_cols], classification_df['classification_target'])

        if self.cv_splits > 1:
            # cross validation --> add test_split column to dataframe to split later
            classification_df['test_split'] = np.ones((classification_df.shape[0],)) * -1
            kfold = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=1)
            stratification_array = classification_df[stratification_columns].apply(lambda row: "".join(row.astype(str)), axis=1)

            for split_idx, (train_indices, test_indices) in enumerate(kfold.split(classification_df, y=stratification_array)):
                assert np.all(classification_df.iloc[test_indices, classification_df.columns.get_loc('test_split')] == -1)  # make sure test samples do not overlap
                classification_df.iloc[test_indices, classification_df.columns.get_loc('test_split')] = split_idx
        elif self.cv_splits == 1:
            # no cross validation, train on train, test on test, using the dataset.data_split assignments
            # check dataloader comments for details on how the splitting was performed
            train_test_split = pd.DataFrame({'split': self.data.data_split, 'sample_name': self.data.sample_names})
            classification_df = classification_df.merge(train_test_split, on='sample_name', how='left')
            classification_df['test_split'] = np.where(classification_df['split'] == 'test', 0, -1)  # -1 -> will never be used as test split
        else:
            raise ValueError(f"Invalid cv_splits value {self.cv_splits}")

        print("CV split statistics", classification_df.test_split.value_counts(), "\n")

        # store data to disk
        classification_df.to_csv(os.path.join(self.run_parameters.results_dir, "classification_data.csv"), index=False)

        if self.model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced')
        elif self.model_name == 'SVC':
            model = SVC(**self.svc_parameters, **{'class_weight': 'balanced'})
        elif self.model_name == 'LogisticRegression':
            model = LogisticRegression(penalty=None, class_weight='balanced', solver='lbfgs', max_iter=1000)
        else:
            raise ValueError(f"Invalid model name {self.model_name}")

        def split_Xy(Xy):
            return Xy[X_cols], Xy['classification_target'], Xy[demographics.columns], Xy['sample_name']

        def predict(X):
            if self.model_name == 'SVC':
                continuous = model.decision_function(X)
                binary = (continuous > 0).astype(int)
            else:
                continuous = model.predict_proba(X)[:, 1]
                binary = np.round(continuous).astype(int)
            return continuous, binary

        # collected data over splits
        computed_metrics_train_collected, computed_metrics_test_collected = [], []
        for split_idx in range(self.cv_splits):
            computed_metrics_train, computed_metrics_test = {}, {}

            Xy_train, Xy_test = classification_df.query(f"test_split != {split_idx}").reset_index(drop=True), classification_df.query(f"test_split == {split_idx}").reset_index(drop=True)
            X_train, y_train, demographics_train, sample_names_train = split_Xy(Xy_train)
            X_test, y_test, demographics_test, sample_names_test = split_Xy(Xy_test)

            X_train, X_test = self._preprocess_data(X_train, demographics_train, sample_names_train,
                                                    X_test, demographics_test, sample_names_test)

            model.fit(X_train, y_train)

            # training error
            predictions_train, predictions_train_binary = predict(X_train)
            computed_metrics_binary = {name: self.binary_metrics[name](y_train, predictions_train_binary) for name in self.binary_metrics}
            computed_metrics_continuous = {name: self.continous_metrics[name](y_train, predictions_train) for name in self.continous_metrics}

            computed_metrics_train = {**computed_metrics_binary, **computed_metrics_continuous}
            computed_metrics_train['predictions'] = predictions_train
            computed_metrics_train['labels'] = y_train
            computed_metrics_train['sample_names'] = sample_names_train

            # test error
            predictions_test, predictions_test_binary = predict(X_test)
            computed_metrics_binary = {name: self.binary_metrics[name](y_test, predictions_test_binary) for name in self.binary_metrics}
            computed_metrics_continuous = {name: self.continous_metrics[name](y_test, predictions_test) for name in self.continous_metrics}

            computed_metrics_test = {**computed_metrics_binary, **computed_metrics_continuous}
            computed_metrics_test['predictions'] = predictions_test
            computed_metrics_test['labels'] = y_test
            computed_metrics_test['sample_names'] = sample_names_test
            computed_metrics_test['features'] = X_test

            if self.calculate_shap:
                # Create the explainer and calculate SHAP values
                if self.model_name == 'RandomForest':
                    explainer = shap.TreeExplainer(model)
                elif self.model_name == 'SVC':
                    explainer = shap.KernelExplainer(model.predict, X_train)
                else:
                    explainer = shap.Explainer(model.predict, X_train)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):
                    # There are two outputs, for each output class, happens for RandomForest
                    assert len(shap_values) == 2, f"Expected one shap_value output per class, but it's {len(shap_values)}"
                    shap_values = shap_values[1]
                else:
                    # Only one output, for LogisticRegression
                    pass
                shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
                computed_metrics_test['shap_values'] = shap_values_df

            computed_metrics_train_collected.append(computed_metrics_train)
            computed_metrics_test_collected.append(computed_metrics_test)


        for metric_name in list(self.continous_metrics.keys()) + list(self.binary_metrics.keys()):
            scores_train = [metrics[metric_name] for metrics in computed_metrics_train_collected]
            scores_test = [metrics[metric_name] for metrics in computed_metrics_test_collected]
            self._log_results(f"CV {metric_name}: {np.mean(scores_test):.3f}+-{np.std(scores_test):.3f}", 'results.txt')
            self._log_results(f"  (Train: {np.mean(scores_train):.3f}+-{np.std(scores_train):.3f})", 'results.txt')

        with open(os.path.join(self.run_parameters.results_dir, 'sample_names.json'), "w") as f:
            sample_names_train = [list(metrics['sample_names']) for metrics in computed_metrics_train_collected]
            sample_names_test = [list(metrics['sample_names']) for metrics in computed_metrics_test_collected]
            json.dump({'sample_names_train': sample_names_train, 'sample_names_test': sample_names_test}, f)

        with open(os.path.join(self.run_parameters.results_dir, 'metrics.json'), "w") as f:
            all_metric_names = list(self.continous_metrics.keys()) + list(self.binary_metrics.keys())
            cast_val = lambda val: val.tolist() if isinstance(val, np.ndarray) else val
            all_metrics = [{m: cast_val(metrics[m]) for m in all_metric_names} for metrics in computed_metrics_test_collected]
            json.dump(all_metrics, f)

        with open(os.path.join(self.run_parameters.results_dir, 'predictions.json'), "w") as f:
            all_vals = [list(split['predictions']) for split in computed_metrics_test_collected]
            json.dump(all_vals, f)
        with open(os.path.join(self.run_parameters.results_dir, 'true_labels.json'), "w") as f:
            all_vals = [list(split['labels']) for split in computed_metrics_test_collected]
            json.dump(all_vals, f)

        all_labels_flat = [elem for split in computed_metrics_test_collected for elem in split['labels']]
        all_predictions_flat = [elem for split in computed_metrics_test_collected for elem in split['predictions']]
        if self.model_name == 'SVC':
            all_predictions_flat_binary = (np.array(all_predictions_flat) > 0).astype(int)
        else:
            all_predictions_flat_binary = np.round(all_predictions_flat).astype(int)
        computed_metrics_binary = {name: self.binary_metrics[name](all_labels_flat, all_predictions_flat_binary) for name in self.binary_metrics}
        computed_metrics_continuous = {name: self.continous_metrics[name](all_labels_flat, all_predictions_flat) for name in self.continous_metrics}
        for metric_name in list(self.continous_metrics.keys()):
            self._log_results(f"[Union of test sets] {metric_name}: {computed_metrics_continuous[metric_name]:.3f}", 'results_union_of_test_sets.txt')
        for metric_name in list(self.binary_metrics.keys()):
            print(metric_name)
            if metric_name == 'confusion_matrix':
                self._log_results(f"[Union of test sets] {metric_name}: {computed_metrics_binary[metric_name]}", 'results_union_of_test_sets.txt')
            else:
                self._log_results(f"[Union of test sets] {metric_name}: {computed_metrics_binary[metric_name]:.3f}", 'results_union_of_test_sets.txt')

        # plot roc and precision-recall curves
        self._plot_roc(os.path.join(self.run_parameters.results_dir, "roc.png"), all_predictions_flat, all_labels_flat)
        self._plot_precision_recall(os.path.join(self.run_parameters.results_dir, "precision_recall.png"), all_predictions_flat, all_labels_flat)

        if self.calculate_shap:
            # write shap values
            shap_values_collected = [metrics['shap_values'] for metrics in computed_metrics_test_collected]
            features_collected = [metrics['features'] for metrics in computed_metrics_test_collected]
            if len(shap_values_collected) > 0:
                shap_values = pd.concat(shap_values_collected)
                features = pd.concat(features_collected)
                shap.summary_plot(shap_values.to_numpy(), features, show=False)
                plt.savefig(os.path.join(self.run_parameters.results_dir, "shap.png"))
                plt.close()
                with open(os.path.join(self.run_parameters.results_dir, "shap_values.txt"), "w") as file:
                    file.write(python_to_json(shap_values))




