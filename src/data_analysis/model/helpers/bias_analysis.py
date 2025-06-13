import numpy as np
import pandas as pd
import os
import torch
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from util.helpers import prepare_demographics, mean_absolute_percentile_error, composite_target_to_string_mapping

class BiasAnalysis:
    def __init__(self, data, target_variable, title="Bias Analysis"):
        self.title = title
        self.data = data  # dataset, including metadata such as demographics
        demographics, _ = prepare_demographics(self.data.demographics)
        self.participant_metadata = pd.concat([
            demographics.reset_index(drop=True),
            self.data.demographics[['ethnicity']].reset_index(drop=True),  # additional demographic column used here but not in other places
            self.data.mean_composite_cognitive_score.reset_index(drop=True),
        ], axis=1)
        self.participant_metadata['sample_name'] = np.array(self.data.sample_names)

        # target, this is only to check that the given targets match the metadata
        self.target_variable = target_variable
        self.target_ground_truth = self.data.factor_scores_theory[[target_variable]]
        self.target_ground_truth['sample_name'] = np.array(self.data.sample_names)

    def _get_metrics_flat(self, metric_name, predictions_flat, targets_flat, model_type='regression'):
        def spearman_correlation_metric(y_true, y_pred):
            return spearmanr(y_true, y_pred).statistic
        def pearson_correlation_metric(y_true, y_pred):
            return pearsonr(y_true, y_pred).statistic

        if model_type == 'classification':
            predictions_rounded = np.round(np.array(predictions_flat))
            metrics = {
                'accuracy': sk_metrics.accuracy_score(targets_flat, predictions_rounded),
            }
            if len(np.unique(targets_flat)) > 1:
                metrics = {**metrics,
                           'recall / sensitivity': sk_metrics.recall_score(targets_flat, predictions_rounded),
                           'specificity': sk_metrics.recall_score(targets_flat, predictions_rounded, pos_label=0),
                           'precision': sk_metrics.precision_score(targets_flat, predictions_rounded),
                           'auroc': sk_metrics.roc_auc_score(targets_flat, predictions_flat),
                           'average_precision': sk_metrics.average_precision_score(targets_flat, predictions_flat),
                           'log_loss': sk_metrics.log_loss(targets_flat, predictions_flat)}

        elif model_type == 'regression':
            metrics = {
                'explained_variance_score': sk_metrics.explained_variance_score(targets_flat, predictions_flat),
                'pearson_correlation': pearson_correlation_metric(targets_flat, predictions_flat),
                'spearman_correlation': spearman_correlation_metric(targets_flat, predictions_flat),
                'mean_absolute_error': sk_metrics.mean_absolute_error(targets_flat, predictions_flat),
                'mean_absolute_percentile_error': mean_absolute_percentile_error(targets_flat, predictions_flat),
                'r2_score': sk_metrics.r2_score(targets_flat, predictions_flat),
            }

        else:
            raise ValueError(f"Invalid type: {model_type}")

        #try:
        #    res = metrics[metric_name]
        #except:
        #    print(f"Cannot calculate metric {metric_name} on bootstrap sample. Continue...")
        #    res = None

        res = metrics[metric_name]
        return res

    def _get_boostrapped_metrics(self, group_df, metric_name, n_bootstrap_samples, model_type):
        # draw bootstrap samples, calculate metric for each sample, return metrics
        metrics = []
        for i in range(n_bootstrap_samples):
            sample = group_df.sample(n=group_df.shape[0], replace=True, axis=0)
            if metric_name == 'wer':
                sample_metric = sample.WER.mean()
            else:
                sample_metric = self._get_metrics_flat(metric_name, sample.prediction, sample.target, model_type)
            if sample_metric is not None:
                metrics.append(sample_metric)
        return metrics

    # def _flatten_data(self, *data):
    #     # flatten list of splits
    #     assert np.all([len(data_item) == len(data[0]) for data_item in data]), "Shape mismatch of outer list"
    #     for i in range(1, len(data)):
    #         for j in range(len(data[0])):
    #             assert np.all([len(data[i][j]) == len(data[0][j])]), "Shape mismatch of inner list"
    #
    #     data_flattened = []
    #     for data_item in data:
    #         flat = [p for split in data_item for p in split]
    #         # move tensors to cpu if given
    #         flat = np.array([p.cpu() if type(p) == torch.Tensor else p for p in flat])
    #         data_flattened.append(flat)
    #
    #     return data_flattened

    def _show_violin_plots_criteria(self, metric_name, predictions, target, sample_names_with_metadata, criteria,
                                    n_bootstrap_samples, plot_path, model_type, figsize=(7, 6), title=None):
        df = sample_names_with_metadata.copy()

        df['prediction'] = predictions
        df['target'] = target

        distributions = []
        targets = []
        group_sizes = []
        criterium_sizes = []
        target_distributions = []
        metric_name_changes = []


        for crit_idx, criterium in enumerate(criteria):
            metric_name_here = metric_name  # metric can be changed for certain criteria
            if criterium == 'gender_unified':
                groups = {'Female': df.query("gender_unified == 0"), 'Male': df.query("gender_unified == 1")}
            elif criterium == 'education_binary':
                groups = {'low-education': df.query("education_binary == 0"), 'high-education': df.query("education_binary == 1")}
            elif criterium == 'ethnicity':
                ethnicity_cleaned = df['ethnicity'].replace({'DATA_EXPIRED': 'Other/NA', 'Asian': 'Other/NA', 'Other': 'Other/NA'})
                unique_values = ethnicity_cleaned.value_counts().keys()
                unique_values = [val for val in unique_values if val != 'Other/NA'] # other is too few -> very high variance results
                groups = {val: df[ethnicity_cleaned == val] for val in unique_values}
            elif criterium == 'country':
                groups = {'uk': df.query("country == 0"), 'usa': df.query("country == 1")}
            elif criterium == 'age':
                # according to train dataset statistics
                groups = {
                    'Age<63': df.query("age < 63"),
                    'Age=63-65': df.query("age >= 63 and age < 66"),
                    'Age=66-68': df.query("age >= 66 and age < 69"),
                    'Age>=69': df.query("age >= 69")
                }
            elif criterium == 'target':
                bins_description = ['very_low', 'low', 'high', 'very_high']
                binned_data, bins = pd.qcut(target, q=4, retbins=True, labels=bins_description)
                bin_labels = [f"target({low:.2f}-{high:.2f})" for low, high in zip(bins[:-1], bins[1:])]
                groups = {bin_label: df[binned_data == bin_value] for bin_label, bin_value in zip(bin_labels, bins_description)}
                if metric_name in ['explained_variance_score', 'spearman_correlation']:
                    # explained_variance / correlation is very bad here, because the fit is much worse when only looking at the subset of people with similar target scores
                    metric_name_here = 'mean_absolute_error'
                metric_name_changes.append({
                    'start_position': np.sum(criterium_sizes) + crit_idx - 1,
                    'metric_name': metric_name_here
                })
            elif criterium == 'age_alternative':
                bins_description = ['very_low', 'low', 'high', 'very_high']
                binned_data, bins = pd.qcut(df['age'], q=4, retbins=True, labels=bins_description)
                bin_labels = [f"Age({low:.2f},{high:.2f})" for low, high in zip(bins[:-1], bins[1:])]
                groups = {bin_label: df[binned_data == bin_value] for bin_label, bin_value in zip(bin_labels, bins_description)}
            else:
                raise ValueError()
            criterium_sizes.append(len(groups))

            for group_name in groups:
                distributions.append(self._get_boostrapped_metrics(groups[group_name], metric_name_here,
                                                                   n_bootstrap_samples, model_type))
                targets.append(group_name)
                group_sizes.append(groups[group_name].shape[0] / df.shape[0] * 100)  # in percent
                if model_type == 'regression':
                    target_distribution = groups[group_name]['target']
                elif model_type == 'classification':
                    raise NotImplementedError()
                    n_samples = groups[group_name].shape[0]
                    target_distribution = (groups[group_name].query("Label == 1").shape[0] / n_samples, groups[group_name].query("Label == 0").shape[0] / n_samples)
                target_distributions.append(target_distribution)

        # plot positions s.t. there are gaps between the criteria
        criterium_sizes_cumulative = np.cumsum(criterium_sizes)
        positions = [pos for start, end, offset in zip([0, *criterium_sizes_cumulative[:-1]], criterium_sizes_cumulative, range(len(criterium_sizes_cumulative))) for pos in range(start+offset, end+offset)]

        fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=(3, 1, 1))

        # violin plot of performance per group
        ax.set_title(title if title is not None else f"{self.title} (n={df.shape[0]})")
        #ax.violinplot(distributions, positions=positions, showextrema=False, showmedians=True, quantiles=[[0.025, 0.975] for _ in range(len(positions))])
        ax.boxplot(distributions, positions=positions, showfliers=False)
        ax.set_xticks(positions, [])
        ax.grid(False)
        ax.set_ylabel(f"{metric_name.replace('r2_score', 'Coefficient of determination')}\n(bootstrap samples)", fontsize=8)
        for metric_name_change in metric_name_changes:
            # when metric changes, make this explicit in plot by adding a new label
            ax.text(metric_name_change['start_position'], np.mean(ax.get_ylim()), metric_name_change['metric_name'], rotation=90, va='center', fontsize=8)

        # bar plots with group sizes
        ax2.bar(positions, group_sizes, alpha=0.5)
        [ax2.text(pos, val / 2, f"{int(val)}%", ha="center", va='center', fontsize=8) for pos, val in
         zip(positions, group_sizes) if val > 0.1]
        ax2.set_ylabel("Subjects (%)", fontsize=8)
        ax2.set_xticks(positions, [])
        ax2.grid(False)

        # bar plot with distribution of labels
        if model_type == 'regression':
            ax3.violinplot(target_distributions, positions=positions, showextrema=False, showmedians=True)
            ax3.legend(ncol=2, fontsize=8, loc="upper right")
            ax3.set_xticks(positions, targets, rotation=45, ha='right')
            ax3.set_ylabel(composite_target_to_string_mapping.get(self.target_variable, self.target_variable), fontsize=8)
        elif model_type == 'classification':
            raise ValueError()
            ax3.bar(positions, np.array(target_distributions)[:, 0], label="AD", alpha=0.5)
            [ax3.text(pos, val / 2, f"{int(val * 100)}%", ha="center", va='center', fontsize=8) for pos, val in zip(positions, np.array(ad_control_fracs)[:, 0]) if val > 0.1]
            ax3.bar(positions, np.array(ad_control_fracs)[:, 1], bottom=np.array(ad_control_fracs)[:, 0], label="Ctrl", alpha=0.5)
            [ax3.text(pos, bottom + val / 2, f"{int(val * 100)}%", ha="center", va='center', fontsize=8) for pos, bottom, val in zip(positions, np.array(ad_control_fracs)[:, 0], np.array(ad_control_fracs)[:, 1]) if val > 0.1]
            ax3.set_ylim([0, 1.6])
            ax3.legend(ncol=2, fontsize=8, loc="upper right")
            ax3.set_yticks([0.5, 1])
            ax3.set_xticks(positions, targets, rotation=45, ha='right')
            ax3.set_ylabel("# AD / # Ctrl", fontsize=8)
        ax3.grid(False)


        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)

        return


    def bias_analysis(self, predictions, targets, sample_names, metric_name,
                      criteria=("age", 'gender_unified', 'education_binary', 'country', 'target'),  # 'ethnicity'
                      plot_path=None, figsize=None, title=None):
        """
        :param predictions (1d): predicted scores
        :param target_variable (1d): true target variable
        :param metadata (multi-dimensional): pd.Dataframe with one row per participant, one column per dimension that you want a bias analysis for
        :param plot_path: path to store result to
        :return:
        """
        assert len(predictions) == len(targets) == len(sample_names), \
            f"Shapes dont match: predictions: {len(predictions)}, labels: {len(targets)}, sample_names {len(sample_names)}"

        sample_names_with_metadata = pd.DataFrame({'sample_name': sample_names}).merge(self.participant_metadata, on="sample_name", how="left")
        assert np.all([np.array(sample_names_with_metadata.sample_name) == np.array(sample_names)]), "Sample name order changed?"

        # check to make sure target is the same
        sample_names_with_target = pd.DataFrame({'target': targets, 'sample_name': sample_names}).merge(self.target_ground_truth, on="sample_name", how="left")
        if not np.all(sample_names_with_target['target'] == sample_names_with_target[self.target_variable]):
            cor = sample_names_with_target.corr(method="spearman").iloc[0,2]
            print(f"Attention: target variable from dataset is not the same as provided targets here. The reason could be data_preprocessors changing the target? Spearman correlation={cor}")
            assert cor > 0.95, f"Data target vs. provided target are too different (spearman correlation = {cor})"

        if plot_path is None:
            print("No path provided for bias analysis, skipping it... ")
            return

        n_bootstrap_samples = 500
        model_type = 'regression'  # 'classification'

        print(f"Calculating bias analysis for {metric_name} and criteria {criteria}")
        return self._show_violin_plots_criteria(metric_name, predictions, targets, sample_names_with_metadata, criteria,
                                                n_bootstrap_samples=n_bootstrap_samples, plot_path=plot_path,
                                                model_type=model_type, figsize=figsize, title=title)




