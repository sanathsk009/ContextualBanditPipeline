import math
import numpy as np
import copy
import scipy.stats as stats

from typing import Any, Dict, Sequence, Hashable
from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.preprocessing import StandardScaler

from cbpipeline.cost_sensitive_classifiers import CSCRegressionOneVsRest


class LocalizedErrorEstimator:
    def __init__(
        self,
        costsensitive_classifier=CSCRegressionOneVsRest(),
        max_error=1,
    ) -> None:
        """_summary_

        Args:
            base_model (regression/classification model, optional): Choose the base model for CSC. Defaults to LinearRegression().
            max_error (int, optional): Set the max error based on reward bounds. Defaults to 1.
        """
        # Setting self.costsensitive_classifier.
        self.costsensitive_classifier = costsensitive_classifier

        # self.errors starts with max value, and shrinks if we estimate smaller errors.
        self.max_error = max_error

    @property
    def params(self) -> Dict[str, Any]:
        dict = {
            "family": "LocalizedErrorEstimation",
            "max_error": self.max_error,
        }

    def fit(self, X, A, Y, P, Y_hat, X_train, Y_hat_train, estimated_opt_policy, delta):
        # Store delta
        self.delta = delta

        # Store estimated_opt_policy and Adversarial policy
        self.estimated_opt_policy = estimated_opt_policy
        self.estimated_opt_policy_arms = estimated_opt_policy.predict(X)

        # Store contexts and normalize them to improve performance.
        Xscaler = StandardScaler()
        Xscaler.fit(X)
        X_norm = Xscaler.transform(X)
        self.X = X_norm
        # Datalength.
        self.datasize = len(self.X)
        # Y_hat
        self.Y_hat = Y_hat

        # Lower bound P
        P = np.maximum(P, len(P[0]) / pow(self.datasize, 0.5))

        # Error according to a policy is the difference between policy evaluation scores and Y_hat.
        # We stick to DR method for policy evaluation.
        Y_ind = A * Y[:, np.newaxis]  # This is the indicator times reward term.
        Y_hat_ind = np.multiply(
            A, Y_hat
        )  # This is the indicator times predicted reward term.
        # We want to compare the difference between DR_scores and direct method scores. The direct method part in DR cancels out.
        Y_diff = Y_ind - Y_hat_ind
        self.localized_score = Y_diff / P

        # Get diff for estimated_opt_policy
        diff_score_for_estimated_opt_policy = self.localized_score[
            np.arange(self.datasize), self.estimated_opt_policy_arms
        ]
        interval_diff_score_for_estimated_opt_policy = stats.t.interval(
            1 - self.delta,
            self.datasize - 1,
            loc=np.mean(diff_score_for_estimated_opt_policy),
            scale=stats.sem(diff_score_for_estimated_opt_policy),
        )
        # Subtract lower interval of this diff score.
        self.localized_score = (
            self.localized_score - interval_diff_score_for_estimated_opt_policy[0]
        )

        # Now we only want to solve a localized version of max self.localized_score subject to R_{hatf}(pi)-R_{hatf}(pi_con) >= -bound.
        # Get estimated_opt_policy_value.
        self.estimated_opt_policy_value = np.mean(
            Y_hat[np.arange(self.datasize), self.estimated_opt_policy_arms]
        )

        try:
            self.localized_error = self._get_constraint_bound()
            print(
                "Localized error",
                self.localized_error,
                np.sqrt(len(A[0]) * np.log(1 / self.delta) / self.datasize),
            )

        except:
            self.localized_error = self.max_error

    def _get_constraint_bound(self):
        constraint_bound = 1
        lagrange = 0
        constraint_satisfied = False
        can_shrink = True

        while can_shrink:
            while not constraint_satisfied:
                lagrange_score = self.localized_score - lagrange * self.Y_hat
                costsensitive_classifier = copy.deepcopy(self.costsensitive_classifier)
                costsensitive_classifier.fit(self.X, -lagrange_score)
                candidate_adv_policy_arms = costsensitive_classifier.predict(self.X)

                # Check constraint
                if (
                    np.mean(
                        self.Y_hat[np.arange(self.datasize), candidate_adv_policy_arms]
                    )
                    >= self.estimated_opt_policy_value - constraint_bound
                ):
                    constraint_satisfied = True
                else:
                    lagrange = 1.25 * lagrange + 0.1
                    lagrange = int(lagrange * 100) / 100

            candidate_adv_scores = self.localized_score[
                np.arange(self.datasize), candidate_adv_policy_arms
            ]
            interval_candidate_adv_scores = stats.t.interval(
                1 - self.delta,
                self.datasize - 1,
                loc=np.mean(candidate_adv_scores),
                scale=stats.sem(candidate_adv_scores),
            )
            new_constraint_bound = interval_candidate_adv_scores[1]
            print("further localizing", lagrange)

            if new_constraint_bound >= constraint_bound:
                can_shrink = False
            else:
                constraint_bound = new_constraint_bound

        return constraint_bound

    def _get_localized_error(self):
        return self.localized_error
