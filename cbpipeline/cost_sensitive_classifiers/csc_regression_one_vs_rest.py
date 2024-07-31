import math
import numpy as np
import copy

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    LassoCV,
    LogisticRegression,
    Ridge,
)


class CSCRegressionOneVsRest:
    """_summary_
    Run a multioutput regression to predict costs for each context-arm pair.
    Output the policy/classifier than minimizes the average predicted cost.
    """

    def __init__(
        self,
        base_regressor=Ridge(),
    ) -> None:
        self.cost_regressor = MultiOutputRegressor(base_regressor)

    def fit(self, X, Y) -> None:
        self.cost_regressor.fit(X, Y)

    def predict(self, X):
        Y_hat = np.array(self.cost_regressor.predict(X))
        return np.argmin(Y_hat, axis=1)
