import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RandomSamplingRegressor(BaseEstimator, RegressorMixin):
    """
    Regressor that randomly samples from the train distribution of targets
    """
    def fit(self, X, y):
        self.y_train_ = np.array(y)
        return self

    def predict(self, X):
        return np.random.choice(self.y_train_, size=X.shape[0], replace=True)