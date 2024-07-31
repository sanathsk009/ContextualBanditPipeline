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


class ErrorEstimator:
    def __init__(
        self,
        base_model=LinearRegression(),
        max_error=1,
        csc_option=1,
        unbiased_evaluator_option=1,
        combine_error_option=1,
    ) -> None:
        """_summary_

        Args:
            base_model (regression/classification model, optional): Choose the base model for CSC. Defaults to LinearRegression().
            max_error (int, optional): Set the max error based on reward bounds. Defaults to 1.
            csc_option (int, optional): Select the cost sensitive classifier. Options range from [1,3]. Defaults to 1.
            unbiased_evaluator_option (int, optional): Select DR vs IPW based error estimation. Options range from [1,2]. Defaults to 1.
        """
        self.csc_option = csc_option
        # Setting self.costsensitive_classifier.
        if csc_option == 1:
            # Our CSC method, which is most foolproof.
            self.base_model = base_model
            self.costsensitive_classifier = CSCRegressionOneVsRest(base_model)

        if csc_option == 2:
            # base_model is a CSClassifier.
            self.costsensitive_classifier = base_model

        # self.errors starts with max value, and shrinks if we estimate smaller errors.
        self.max_error = max_error
        self.conservative_error = max_error
        self.nonconservative_error = max_error
        # Select DR vs IPW based error estimation. Options range from [1,2]. Defaults to 1.
        self.unbiased_evaluator_option = unbiased_evaluator_option
        # how do we want to combine conservative and non concervative error
        self.combine_error_option = combine_error_option

    @property
    def params(self) -> Dict[str, Any]:
        dict = {
            "family": "ErrorEstimation",
            "max_error": self.max_error,
            "csc_option": self.csc_option,
            "unbiased_evaluator_option": self.unbiased_evaluator_option,
            "combine_error_option": self.combine_error_option,
            "regressor": 0,
        }
        if self.csc_option == 1:
            dict["regressor"] = type(self.base_model).__name__

    def fit(self, X, A, Y, P, Y_hat):
        """_summary_

        Args:
            X (np.array): Array of contexts.
            A (np.array): Array of selected arms. Each row is a one-hot vector indicating which arm is selected.
            Y (np.array): Vector of observed outcomes.
            P (np.array): Array of arm selection probabilities.
            Y_hat (np.array): Array of estimated arms rewards. Each row is a vector indicating estimated arm rewards.
        """
        # Adversarial policy
        self.adv_policies = {}

        # Store contexts and normalize them to improve performance.
        Xscaler = StandardScaler()
        Xscaler.fit(X)
        X_norm = Xscaler.transform(X)
        self.X = X_norm
        # Datalength.
        self.datasize = len(self.X)
        # Store greedy policy arms
        self.greedy_policy_arms = np.argmax(Y_hat, axis=1)

        # Lower bound P
        P = np.maximum(P, len(P[0]) / pow(self.datasize, 0.25))

        # (Normalized) Error according to a policy is a ratio of two average scores. |Avg_numerator_score|/sqrt(Avg_denominator_score).
        # Here avg_numerator_score captures estimation error for policy \pi.
        # Denominator term is based on scores for policy cover (a surrogate for variance of policy evaluation).
        self.D = 1 / P

        # Numerator scores compare Y_hat with policy evaluation scores. Use DR or IPS policy evaluation approaches.
        Y_ind = A * Y[:, np.newaxis]
        Y_hat_ind = np.multiply(A, Y_hat)
        if self.unbiased_evaluator_option == 1:
            # "Double Robust Option": The direct method part cancel out. Only difference is in the outcomes.
            # We want to compare the difference between DR_scores and direct method scores. This will be our neumerator terms.
            Y_diff = Y_ind - Y_hat_ind
            self.N = Y_diff / P

        elif self.unbiased_evaluator_option == 2:
            # "IPW option"
            # Compute inverse propensity scores.
            IPS_scores = Y_ind / P

            # Note that Y_hat is the direct method scores.
            # We want to compare the difference between IPS_scores and direct method scores. This will be our neumerator terms.
            self.N = IPS_scores - Y_hat

        # error = max_(\pi) |N(\pi)|/\sqrt(D(\pi)). Where N and D are average of some policy based score.
        # Here N(\pi) captures estimation error for policy \pi.
        # D(\pi) captures cover (measure of policy estimation variance) of policy \pi.
        # We convert the above into cost sensitive classification problems.
        try:
            self._reduction_to_CSC()
            # Get error for greedy policy
            error = min(
                self._error_for_given_policy_arms(self.greedy_policy_arms),
                self.max_error,
            )
            self.conservative_error = max(error, self.conservative_error)
            # Calculate error for all possibly adversarial policies.
            for policy in self.adv_policies.values():
                candidate_adversarial_policy_arms = policy.predict(self.X)
                candidate_error = self._error_for_given_policy_arms(
                    candidate_adversarial_policy_arms
                )
                error = max(error, candidate_error)

            self.nonconservative_error = min(self.nonconservative_error, error)
            self.error = self._combine_errors()
            print(
                self.nonconservative_error,
                self.conservative_error,
                np.sqrt(1 / self.datasize),
                self.error,
            )
        except:
            self.error = self.max_error

    def _reduction_to_CSC(self):
        """_summary_
        We want to find the largest (normalized) error possible under any policy.
        That is, we want to solve max_(\pi) |N(\pi)|/\sqrt(D(\pi)). Where N and D are average of some policy based score.
        We convert the above into cost sensitive classification problems.

        Main solution:
        We basically argue above can be reduced to find smallest ratio such that, max_(\pi) |N(\pi)| - ratio*D(\pi)<=0.
        This ratio * sqrt(D(\pi)) is the solution to the above problem.
        The corresponding policy should be the one with the largest error.

        Important note:
        Optimization errors in cost sensitive classification may not be able to identify when max_(\pi) N(\pi) - ratio*D(\pi)<=0.
        So ratio may be larger than the original objective value of selected policy.
        """
        # To find the smallest ratio, we start with ratio = 1.
        # Keep dropping ratio unless next ratio doesn't satisfy max_(\pi) |N(\pi)| - ratio*D(\pi)<=0.

        # Initialize ratio and reduction rates. Check_both_signs helps ensure the objective was resolved for both signs.
        ratio = 1
        drop_ratio_rate = 10
        check_both_signs = 2

        # Keep dropping ratio unless ratio is too small.
        while check_both_signs == 2 and ratio > 0.1 / self.datasize:
            # Verify the CSC conditions are met for both signs.
            check_both_signs = 0
            max_adv_variance = 0
            for sign in [-1, 1]:
                # check is true if max_(\pi) N[sign](\pi) - ratio*D(\pi)<=0.
                # advarsarial_variance = D(\pi)
                check, advarsarial_variance = self._check_ratio(ratio, sign * self.N)
                check_both_signs += check
                max_adv_variance = max(advarsarial_variance, max_adv_variance)
                self.adv_policies[len(self.adv_policies) + 1] = copy.deepcopy(
                    self.costsensitive_classifier
                )

            # Update errors if conditions are met for both signs, and drop ratio.
            if check_both_signs == 2:
                # calculate candidate (normalized) error corresponding to current candidate ratio level.
                candidate_error = ratio * np.sqrt(max_adv_variance)
                self.conservative_error = min(self.conservative_error, candidate_error)
                ratio = ratio / drop_ratio_rate
            # Else drop ratio by a smaller quantity.
            else:
                ratio = ratio * drop_ratio_rate
                drop_ratio_rate = 1 + drop_ratio_rate / 4
                ratio = ratio / drop_ratio_rate
                if drop_ratio_rate >= 2:
                    check_both_signs = 2

    def _check_ratio(self, ratio, N):
        """_summary_

        Args:
            ratio (float): We want to check if max_(\pi) N(\pi) - ratio*D(\pi)<=0
            N (array): Provide an array of N (numerator values) for each arm at each sample

        Returns:
            check (bool): True if max_(\pi) N(\pi) - ratio*D(\pi)<=0.
            adversarial_variance (floot): return D(\pi) for the above policy.
        """
        # Scores to be maximized be an adversarial policy.
        scores = N - ratio * (self.D - np.sqrt(1 / 2)) + np.sqrt(1 / len(self.N))

        # Finding a policy that maximizes these scores.
        self.costsensitive_classifier.fit(self.X, -scores)

        # Candidate adversarial policy arms on observed contexts.
        candidate_adversarial_policy_arms = self.costsensitive_classifier.predict(
            self.X
        )

        # Adversarial scores
        adversarial_scores = scores[
            np.arange(self.datasize), candidate_adversarial_policy_arms
        ]

        # Is the ratio sufficient to get a reasonable adversary?
        check = np.sum(adversarial_scores) <= 0
        self.check = check

        # What is the variance of this policy?
        denominator_scores = self.D[
            np.arange(self.datasize), candidate_adversarial_policy_arms
        ]
        advarsarial_variance = np.sum(denominator_scores) / self.datasize

        return check, advarsarial_variance

    def get_error(self):
        """_summary_
        Calculate error for any potentially adversarial policy and take the max.

        Returns:
            float: squared error estimate
        """

        return np.power(self.error, 2)

    def _error_for_given_policy_arms(self, candidate_adversarial_policy_arms):
        """_summary_

        Args:
            candidate_adversarial_policy_arms (list): Get list of arms for candidate adversarial policy

        Returns:
            float: return corresponding error
        """
        # Get donominator scores for candidate adversarial policy.
        denominator_scores = self.D[
            np.arange(self.datasize), candidate_adversarial_policy_arms
        ]
        # Get cover (surrogate variance) for candidate adversarial policy.
        avg_denominator = np.mean(denominator_scores)
        stde_denominator = stats.sem(denominator_scores)
        interval_denominator = stats.t.interval(
            0.95,
            len(denominator_scores) - 1,
            loc=avg_denominator,
            scale=stde_denominator,
        )
        lbound_denominator = max(interval_denominator[0], 1)
        # Get numerator scores (estimation error) for candidate adversarial policy.
        numerator_scores = self.N[
            np.arange(self.datasize), candidate_adversarial_policy_arms
        ]
        # Get average estimation error for candidate adversarial policy.
        avg_numerator = np.mean(numerator_scores)
        stde_numerator = stats.sem(numerator_scores)
        interval_numerator = stats.t.interval(
            0.95, len(numerator_scores) - 1, loc=avg_numerator, scale=stde_numerator
        )
        upbound_numerator = max(abs(interval_numerator[0]), abs(interval_numerator[1]))
        # Return (normalized) error for candidate adversarial policy.
        # candidate_error = upbound_numerator / np.sqrt(lbound_denominator)
        candidate_error = upbound_numerator / np.sqrt(
            avg_denominator
        )  # using avg instead of lbound_denominator to be less sensitive to arm elimination. Shouldn't be a big difference in other cases.
        return candidate_error

    def _combine_errors(self):
        if self.combine_error_option == 1:
            error = self.nonconservative_error
        if self.combine_error_option == 2:
            error = np.sqrt(self.nonconservative_error * self.conservative_error)
        if self.combine_error_option == 3:
            error = self.conservative_error

        if error == np.nan:
            error = self.max_error
        return error
