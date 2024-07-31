import numpy as np
import copy
from cbpipeline.cost_sensitive_classifiers import PolicyTreeCSC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from cbpipeline.policy_learner.policy_learner_base import PolicyLearner


class PolicyLearnerUsedInCharitableGivingExperiment(PolicyLearner):
    def __init__(self, regressor=RandomForestRegressor(), train_frac=0.8):
        self.regressor = regressor
        self.train_frac = train_frac

    def fit(self, X, Y, A, P, actions):
        scores = self._get_dr_scores(X, Y, A, P, actions, self.regressor)

        ## Taking this part of code from: https://github.com/gsbDBI/toronto/blob/6450e6a6725084330fafea05b2d211eedfc6c9ae/toronto/utils.py#L9
        n = len(X)
        train_frac = self.train_frac
        t_cv1 = np.arange(int(train_frac * n))  # train
        t_cv2 = np.arange(int(train_frac * n), n)  # evaluation
        # Cross-validation step
        tree_cv_values = []
        for depth in [1, 2]:
            tree_cv = PolicyTreeCSC(depth=depth).fit(X[t_cv1], -scores[t_cv1])
            ws_cv = tree_cv.predict(X[t_cv2])
            tree_cv_value = np.mean(scores[t_cv2, ws_cv])
            tree_cv_values.append(tree_cv_value)

        # Selects depth with highest point-estimated value
        best_depth = np.argmax(tree_cv_values) + 1
        self.csclassifier = PolicyTreeCSC(depth=best_depth)
        self.csclassifier.fit(X, -scores)

    def predict(self, X):
        return self.csclassifier.predict(X)
