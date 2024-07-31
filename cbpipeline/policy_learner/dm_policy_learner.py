import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from cbpipeline.policy_learner.policy_learner_base import PolicyLearner


class DMPolicyLearner(PolicyLearner):
    def __init__(self, regressor=LinearRegression()):
        self.regressor = regressor

    def fit(self, X, Y, A, P, actions):
        self.actions = actions
        self.final_model = {}
        for action in actions:
            idx = np.all(A == action, axis=1)
            X_a = X[idx]
            Y_a = Y[idx]

            model_a = copy.deepcopy(self.regressor)
            model_a.fit(X_a, Y_a)
            self.final_model[action] = model_a

    def predict(self, X):
        actions = self.actions

        Y_hat = np.ones((len(X), len(actions)))
        # Predict outcome for all eval contexts
        for action in actions:
            Y_hat[:, np.argmax(action)] = self.final_model[action].predict(X)
        learnt_policy_arms = np.argmax(Y_hat, axis=1)
        return learnt_policy_arms
