import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

from data_analysis.model.base_model import BaseModel

class TestRetestAnalysis(BaseModel):
    """
    Test-Retest analysis of linguistic features -> how stable are they across different tasks
    """

    def __init__(self, *args, **kwargs):
        super().__init__("TestRetestAnalysis", *args, **kwargs)

    def prepare_data(self):
        pass

    def run(self):
        X_original = self.data.features
        Y_original = pd.concat((self.data.acs_outcomes_imputed, self.data.factor_scores, self.data.language_task_scores), axis=1).drop(columns=['sample_name'])
        #Y_original = self.data.factor_scores.drop(columns=['sample_name'])

        # remove any rows / participants with nan values, either in X or Y
        x_cols = X_original.columns.tolist()
        y_cols = Y_original.columns.tolist()
        XY = pd.concat([X_original, Y_original], axis=1).dropna()
        X = XY[x_cols].reset_index(drop=True)
        Y = XY[y_cols].reset_index(drop=True)

        n_components = min(len(x_cols), len(y_cols))
        #n_components = 5
        cca = CCA(n_components=n_components)

        # Fit the CCA model to X and Y
        cca.fit(X, Y)

        # Transform X and Y to canonical variables
        X_c, Y_c = cca.transform(X, Y)

        # Score the CCA model
        score = cca.score(X, Y)

        correlation_matrix = np.corrcoef(X_c.T, Y_c.T)
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix[n_components:, :n_components], annot=n_components <= 10)
        plt.title('Canonical Variables Correlation Matrix')
        plt.xlabel("X components")
        plt.ylabel("Y components")
        plt.savefig(os.path.join(self.run_parameters.results_dir, 'cca_correlation_matrix.png'))

        correlation_matrix = np.corrcoef(X.T, Y.T)
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix[X.shape[1]:, :X.shape[1]], annot=X.shape[1] <= 10)
        plt.title('Data correlation matrix')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(np.arange(X.shape[1])+0.5, X.columns, rotation=45, ha='right')
        plt.yticks(np.arange(Y.shape[1])+0.5, Y.columns)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, 'data_correlation.png'))

        correlation_matrix = np.corrcoef(X.T, X_c.T)
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix[X.shape[1]:, :X.shape[1]], annot=X.shape[1] <= 10)
        plt.title('Correlation with components for X')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(np.arange(X.shape[1])+0.5, X.columns, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, 'correlation_X_with_components.png'))

        correlation_matrix = np.corrcoef(Y.T, Y_c.T)
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_matrix[Y.shape[1]:, :Y.shape[1]], annot=Y.shape[1] <= 10)
        plt.title('Correlation with components for Y')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(np.arange(Y.shape[1])+0.5, Y.columns, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_parameters.results_dir, 'correlation_Y_with_components.png'))


        # describe correlations
        def get_component_desc(col_names, component_weights):
            weights = pd.DataFrame({'feature': col_names, 'weight': component_weights}).sort_values(by='weight', key=abs, ascending=False)
            weights = weights[weights.weight.abs() > 0.2]
            return " / ".join([f"{row.feature} ({row.weight:.2f})" for _, row in weights.iterrows()])
        for i in range(n_components):
            print(f"{i+1}th component: Correlation {correlation_matrix[n_components+i, i]:.2f}")
            print("   ", get_component_desc(x_cols, cca.x_weights_[:, i]))
            print("   ", get_component_desc(y_cols, cca.y_weights_[:, i]))


        # Print the score
        print(score)