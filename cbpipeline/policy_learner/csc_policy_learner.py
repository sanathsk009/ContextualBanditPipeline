import numpy as np
import copy
from cbpipeline.cost_sensitive_classifiers import CSCRegressionOneVsRest
from sklearn.linear_model import LinearRegression
from cbpipeline.policy_learner.policy_learner_base import PolicyLearner


class CSCPolicyLearner(PolicyLearner):
    def __init__(
        self, csclassifier=CSCRegressionOneVsRest(), regressor=LinearRegression()
    ):
        self.csclassifier = csclassifier
        self.regressor = regressor

    def fit(self, X, Y, A, P, actions):
        scores = self._get_dr_scores(X, Y, A, P, actions, self.regressor)
        self.csclassifier.fit(X, -scores)

    def predict(self, X):
        return self.csclassifier.predict(X)
