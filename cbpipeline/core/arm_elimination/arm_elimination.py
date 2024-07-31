import math
import numpy as np
import copy

from typing import Any, Dict, Sequence, Hashable
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from cbpipeline.cost_sensitive_classifiers import CSCRegressionOneVsRest


class ArmEliminator:
    def __init__(
        self,
        base_model=LinearRegression(),
        csc_option=1,
    ) -> None:
        """_summary_

        Args:
            base_model (regression/classification model, optional): Choose the base model for CSC. Defaults to LinearRegression().
            csc_option (int, optional): Select the cost sensitive classifier. Options range from [1,3]. Defaults to 1.
            unbiased_evaluator_option (int, optional): Select DR vs IPW based error estimation. Options range from [1,2]. Defaults to 1.
        """
        # Setting self.costsensitive_classifier.
        if csc_option == 1:
            # Our CSC method, which is most foolproof.
            self.costsensitive_classifier = CSCRegressionOneVsRest(base_model)

        if csc_option == 2:
            # base_model is a CSClassifier.
            self.costsensitive_classifier = base_model

    def fit(self, X, A, Y, P, Y_hat, error, cover_bound, model, actions, training_data):
        """_summary_
        This fit function performs three tasks:
            - Stores data as class variables.
            - Calculates sub-optimality scores. Useful for evaluating estimated gap from estimated optimal policy.
            - Calculates estimation gap to determine the threshold at which high-avg-sub-optimality scores prove sub-optimality.

        Args:
            X (np.array): Array of contexts.
            A (np.array): Array of selected arms. Each row is a one-hot vector indicating which arm is selected.
            Y (np.array): Vector of observed outcomes.
            P (np.array): Array of arm selection probabilities.
            Y_hat (np.array): Array of estimated arms rewards. Each row is a vector indicating estimated arm rewards.
            PotOptArms (np.array): Array of potentially optimal arms. Each row is a vector of 0/1 indicating if the corresponding arm is included.
            error (float): Estimated error bound.
            cover_bound (float): Estimated bound on optimal cover.
            model (dict): Reward model estimates for each arm.
            actions (list): list of actions
            training_data (tuple): (X, A, Y, P) from original training set (not error estimation).
        """
        # Save training data in case it is needed training_data=(X_train, A_train, Y_train, P_train)
        self.training_data = training_data
        self.error_estimation_data = (X, A, Y, P)
        self.actions = actions

        # Save model
        self.model = model
        # Store Data.
        # Save Y_hat
        self.Y_hat = Y_hat
        # Store contexts.
        self.X = X
        # Datalength.
        self.datasize = len(self.X)
        # Datalength.
        self.datasize = len(self.X)
        # Save error and cover bound.
        self.error = error
        self.cover_bound = cover_bound
        # Save Y_hat
        self.Y_hat = Y_hat

        # Lower bound P
        P = np.maximum(P, max(1 / np.sqrt(self.datasize), error))

        # Calculate suboptimality scores. (Part 1)
        # Using only Y_hat, because only reward model error was measured.
        estimated_opt_arms = np.argmax(Y_hat, axis=1)
        estimated_opt_value = np.mean(
            Y_hat[np.arange(self.datasize), estimated_opt_arms]
        )
        # Finally get suboptimality scores.
        self.subopt_scores = estimated_opt_value - Y_hat

        # Calvulate estimated gap. (Part 2)
        # Get potential opt bound.
        estimated_opt_cover = np.mean(
            1 / P[np.arange(self.datasize), estimated_opt_arms]
        )
        # self.estimated_gap = (
        #     np.sqrt(cover_bound) + np.sqrt(estimated_opt_cover)
        # ) * error
        self.estimated_gap = (
            np.sqrt(min(estimated_opt_cover, cover_bound)) * error
        )  # estimated_opt_cover may be tighter than cover_bound.
        self.estimated_gap = min(1, self.estimated_gap)

    def elimcheck(self, PotOptArms):
        """_summary_

        Args:
            PotOptArms (np.array): Array of potentially optimal arms. Each row is a vector of 0/1 indicating if the corresponding arm is included.

        Returns:
            Tuple:
            - Component one is true if there is a policy that is nearopt but not covered.
            - Component two is the suggested policy.
        """

        # Store set of potentially optimal arms.
        self.PotOptArms = PotOptArms

        # Search for the best policy (smallest average subopt scores that is not covered by potentially optimal arms)
        # Lagrange mulptiplier penalizing policies covered by self.PotOptArms.
        lagrange_multiplier = self.estimated_gap / 100
        # Rate at which lagrange_multiplier is increased.
        increasing_lm_rate = 5  # 1.5 # Moving from 1.5 to improve run time.

        while lagrange_multiplier <= 10:
            # print("now here")
            # Add subopt_scores with penalization for coverage by PotOptArms.
            scores = self.subopt_scores + lagrange_multiplier * self.PotOptArms
            # Solve the associated CSC problem.
            self.costsensitive_classifier.fit(self.X, scores)
            # Find the set of arms recommended by the learnt candidate policy.
            candidate_uncovered_policy_arms = self.costsensitive_classifier.predict(
                self.X
            )
            candidate_uncovered_policy_arms_one_hot = np.zeros(self.Y_hat.shape)
            candidate_uncovered_policy_arms_one_hot[
                np.arange(self.datasize), candidate_uncovered_policy_arms
            ] = 1
            # Check if the candidate policy is covered.
            true_if_candidate_covered = np.min(
                candidate_uncovered_policy_arms_one_hot <= self.PotOptArms
            )

            # Break if not covered, increase lagrange_multiplier if covered.
            if true_if_candidate_covered:
                lagrange_multiplier = lagrange_multiplier * increasing_lm_rate
            else:
                break
        # Loop ideally exits with a candidate_policy that is not covered by PotOptArms and has the smallest average sub-opt score among these policies.
        # If candidate_policy is still covered by PotOptArms, then the lagrange_multiplier probably needed to be much larger than we expected.

        # Calculare average subopt score for this final candidate policy.
        avg_suboptscore_for_final_candidate_policy = np.mean(
            self.subopt_scores[
                np.arange(self.datasize), candidate_uncovered_policy_arms
            ]
        )
        true_if_candidate_nearopt = (
            avg_suboptscore_for_final_candidate_policy <= self.estimated_gap
        )
        return (
            true_if_candidate_nearopt
            and (not true_if_candidate_covered),  # True if near opt and not covered
            copy.deepcopy(self.costsensitive_classifier),
            candidate_uncovered_policy_arms_one_hot,
        )

    def _gap_thresholded_elimination_with_policy_addition(self):
        gap = 0.1 / self.datasize
        policy_list = []
        gap_multiplier = 1.5

        # Get Y_hat and Y_greedy.
        Y_hat = self.Y_hat
        Y_greedy = np.max(Y_hat, axis=1)
        # Get a single column of greedy estimates
        Y_greedy = Y_greedy.reshape((-1, 1))
        PotOptArms = Y_greedy - Y_hat <= gap

        while gap <= 2:
            (
                true_if_near_opt_not_covered,
                policy,
                candidate_uncovered_policy_arms,
            ) = self.elimcheck(PotOptArms)
            if true_if_near_opt_not_covered:
                gap = gap * gap_multiplier
                policy_list.append(policy)
                PotOptArms += (
                    Y_greedy - Y_hat <= gap
                ) + candidate_uncovered_policy_arms
                PotOptArms = PotOptArms > 0
            else:
                break

        if gap >= 1:
            self.gap = 1
            self.policy_list = []
        else:
            self.gap = gap
            self.policy_list = policy_list

        return (self.gap, self.policy_list)

    def predict(self, contexts, actions):
        theta_hat = np.ones((len(contexts), len(actions)))
        for action in actions:
            theta_hat[:, np.argmax(action)] = self.model[action].predict(contexts)
        theta_max = np.max(Y_hat, axis=1)
        theta_max = theta_max.reshape((-1, 1))
        surviving_arms = theta_max - theta_hat <= self.gap
        for policy in self.policy_list:
            arms = policy.predict(contexts)
            surviving_arms[np.arange(len(contexts)), arms] = 1
        return surviving_arms

    def default_eliminator(self):
        self._gap_thresholded_elimination_with_policy_addition()


class BootstrapArmEliminator(ArmEliminator):

    def __init__(
        self,
        base_model=LinearRegression(),
        csc_option=1,
        num_bootstraps=100,
        bootsrap_regressor=LinearRegression(),
    ) -> None:
        """_summary_

        Args:
            base_model (regression/classification model, optional): Choose the base model for CSC. Defaults to LinearRegression().
            csc_option (int, optional): Select the cost sensitive classifier. Options range from [1,3]. Defaults to 1.
            unbiased_evaluator_option (int, optional): Select DR vs IPW based error estimation. Options range from [1,2]. Defaults to 1.
        """
        super().__init__(base_model=base_model, csc_option=csc_option)
        self.num_bootstraps = num_bootstraps
        self.bootsrap_regressor = bootsrap_regressor

    def _bootstrap_eliminator(self):
        self.bootstrap_interval_adjuster = 1
        multiplier = 1.25

        # fit bootstrap model.
        self._update_bootstrap_model()

        # Get bootstrap estimate and default intervals on error dataset.
        X_error = self.error_estimation_data[0]
        Y_hat_bootstrap_mean, Y_hat_bootstrap_interval = (
            self._get_bootstrap_estimates_and_intervals(X_error)
        )

        # Update PotOptArms.
        true_if_near_opt_not_covered = True
        while true_if_near_opt_not_covered:
            print("here")
            # Get potentially optimal arms:
            Y_hat_bootstrap_lower = (
                Y_hat_bootstrap_mean
                - self.bootstrap_interval_adjuster * Y_hat_bootstrap_interval
            )
            Y_hat_bootstrap_upper = (
                Y_hat_bootstrap_mean
                + self.bootstrap_interval_adjuster * Y_hat_bootstrap_interval
            )
            Y_hat_bootstrap_lower_greedy = np.max(Y_hat_bootstrap_lower, axis=1)
            Y_hat_bootstrap_lower_greedy = Y_hat_bootstrap_lower_greedy.reshape((-1, 1))
            PotOptArms = Y_hat_bootstrap_upper - Y_hat_bootstrap_lower_greedy >= 0

            true_if_near_opt_not_covered, _, _ = self.elimcheck(PotOptArms)
            if true_if_near_opt_not_covered:
                self.bootstrap_interval_adjuster = (
                    multiplier * self.bootstrap_interval_adjuster
                )
        # print("exit")

    def _update_bootstrap_model(self):
        # get training data.
        self._X_train, self._A_train, self._Y_train, self._P_train = self.training_data

        # fit bootstrap model.
        self.bootstrap_model = {}
        actions = self.actions
        for action in actions:
            idx = np.all(self._A_train == action, axis=1)
            X_a = self._X_train[idx]
            Y_a = self._Y_train[idx]

            model_a = BaggingRegressor(
                copy.deepcopy(self.bootsrap_regressor), n_estimators=self.num_bootstraps
            )
            model_a.fit(X_a, Y_a)
            self.bootstrap_model[action] = model_a

    def _get_bootstrap_estimates_and_intervals(self, X):
        Y_hat_bootstrap = []
        Y_hat_bootstrap_mean = 0
        for _ in range(self.num_bootstraps + 1):
            Y_hat_temp = np.ones((len(X), len(self.actions)))
            for action in self.actions:
                if _ < self.num_bootstraps:
                    Y_hat_temp[:, np.argmax(action)] = (
                        self.bootstrap_model[action].estimators_[_].predict(X)
                    )
                else:
                    Y_hat_temp[:, np.argmax(action)] = self.model[action].predict(X)
            Y_hat_bootstrap.append(Y_hat_temp)
            Y_hat_bootstrap_mean += Y_hat_temp
        Y_hat_bootstrap_mean = Y_hat_bootstrap_mean / len(Y_hat_bootstrap)
        Y_hat_bootstrap_interval = np.array(Y_hat_bootstrap) - Y_hat_bootstrap_mean
        Y_hat_bootstrap_interval = np.maximum.reduce(np.abs(Y_hat_bootstrap_interval))
        # Heurestic width increase to ensure termination:
        Y_hat_bootstrap_interval += min(self.estimated_gap, 1 / self.datasize)

        return Y_hat_bootstrap_mean, Y_hat_bootstrap_interval

    def predict(self, contexts, actions):
        theta_hat, interval = self._get_bootstrap_estimates_and_intervals(contexts)
        theta_hat = theta_hat
        interval = interval
        theta_hat_lower = theta_hat - self.bootstrap_interval_adjuster * interval
        theta_hat_upper = theta_hat + self.bootstrap_interval_adjuster * interval
        theta_hat_lower_greedy = np.max(theta_hat_lower, axis=1)
        theta_hat_lower_greedy = theta_hat_lower_greedy.reshape((-1, 1))
        surviving_arms = theta_hat_upper - theta_hat_lower_greedy >= 0
        return surviving_arms

    def default_eliminator(self):
        self._bootstrap_eliminator()
