import math
import numpy as np
import copy

from typing import Any, Dict, Sequence, Hashable
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split
from cbpipeline.core.error_estimation.error_estimation import ErrorEstimator
from cbpipeline.core.error_estimation.localized_error_estimation import (
    LocalizedErrorEstimator,
)
from cbpipeline.core.arm_elimination.arm_elimination import (
    ArmEliminator,
    BootstrapArmEliminator,
)
from cbpipeline.core.arm_elimination.localized_arm_elimination import (
    LocalizedBootstrapArmEliminator,
)
from cbpipeline.policy_learner import DMPolicyLearner
from cbpipeline.cost_sensitive_classifiers import CSCRegressionOneVsRest

import time


class LocalizedPipeline:
    """_summary_
    __Note__:
    - Regarding "_calculate_alpha":
        - Only "_calculate_alpha" uses a slightly heurestic choice. All other parts are in line with paper's theory.
        - May also want to modify _calculate_alpha to incorporate arm elimination.
    - May want to shrink output of _calculate_risk_adj based on arm elimination.
    """

    def __init__(
        self,
        epoch_schedule: int = 0,
        confidence_parameter: float = 0.95,
        omega: float = 0,
        regressor=LinearRegression(),
        costsensitive_classifier=None,
        localized_error_estimator=None,
        arm_eliminator=None,
        is_elim=False,
        elim_fraction=0.5,
        sequential_elim_fraction=0,
        error_data_ratio=0.5,
    ) -> None:
        # model parameters
        self.model: Dict[Hashable, Any] = {}
        self.regressor = copy.deepcopy(regressor)
        if costsensitive_classifier == None:
            self.costsesitive_classifier = CSCRegressionOneVsRest(
                base_regressor=copy.deepcopy(regressor)
            )
        else:
            self.costsesitive_classifier = copy.deepcopy(costsensitive_classifier)
        self.central_policy = copy.deepcopy(self.costsesitive_classifier)

        if localized_error_estimator == None:
            self.localized_error_estimator = LocalizedErrorEstimator(
                costsensitive_classifier=copy.deepcopy(self.costsesitive_classifier)
            )
        else:
            self.localized_error_estimator = copy.deepcopy(localized_error_estimator)

        if arm_eliminator == None:
            self.arm_eliminator = LocalizedBootstrapArmEliminator(
                costsensitive_classifier=copy.deepcopy(self.costsesitive_classifier),
                bootsrap_regressor=copy.deepcopy(regressor),
            )
        else:
            self.arm_eliminator = copy.deepcopy(arm_eliminator)
        self.arm_eliminator_list = []

        # history
        self._X: np.typing.NDArray[float] = []
        self._A: np.typing.NDArray[float] = []
        self._Y: np.typing.NDArray[float] = []
        self._P: np.typing.NDArray[float] = []

        # time record
        self._t: int = 0
        self._epoch: int = 0

        # exploration parameters
        self._epoch_schedule: int = epoch_schedule  # Sets epoch schedule in learn.
        self._delta: float = 1.0 - confidence_parameter
        self._omega: float = omega  # trade-off parameter
        # We use the term localization error and optimal cover upper bound interchangably.
        self._localized_error = 1
        self._max_beta: float = 0.5  # beta_max in RAPR
        self._risk_adj: float = 1  # eta_m in RAPR
        self._alpha = None  # bound on optimal cover
        self.error_data_ratio = error_data_ratio

        # elimination parameters
        self._max_gap = 1
        self.is_elim = is_elim
        self.elim_fraction = elim_fraction
        self.sequential_elim_fraction = sequential_elim_fraction

        # Track quantities of interest:
        self.tracker = {
            "update_t": [],
            "error": [],
            "risk_adj": [],
            "update_modeltrain_time": [],
            "update_error_time": [],
            "update_armelimination_time": [],
            "update_total_time": [],
        }

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "family": "LocalizedPipeline",
            "regressor": type(self.regressor).__name__,
            "omega": str(self._omega),
            "is_elim": self.is_elim,
            "has_predict_out_of_sample": True,
            "batch_option": True,
            "error_data_ratio": self.error_data_ratio,
            "elim_fraction": self.elim_fraction,
            "sequential_elim_fraction": self.sequential_elim_fraction,
        }

    @property
    def _get_tracker(self) -> Dict[str, Any]:
        return self.tracker

    def learn(
        self,
        context,
        action,
        reward: float,
        probability: float,
        info,
        is_batch=False,
    ) -> None:
        """_summary_:
        Takes context, action, reward, and additional info to update bandit parameters.
        This function simply adds data to the list of context, action, rewards.
        At the end of every epoch (update condition is satisfied), and internal models are updated in self._update.

        Args:
            context (): List of features (for current user).
            action (): One hot vector denoting selected action (for current user).
            reward (float): Real valued number (reward for current user).
            probability (float): Not used.
            info (): Not used.
        """
        if is_batch:
            contexts = np.array(context, dtype=float)
            actions = np.array(action, dtype=float)
            rewards = np.array(reward, dtype=float)
            probabilities = np.array(probability, dtype=float)
        else:
            # Only let context be defined by numerical values.
            contexts = np.array([self._get_valid_context(context)], dtype=float)
            actions = np.array([action], dtype=float)
            rewards = np.array([reward], dtype=float)
            probabilities = np.array([probability], dtype=float)

        batch_len = len(contexts)

        # Add context, action, and rewards to dataset.
        # Action is expected to be a one hot vector.
        if len(self._X) == 0:
            self._X = contexts
            self._A = actions
            self._Y = rewards
            self._P = probabilities

        else:
            self._X = np.vstack([self._X, contexts])
            self._A = np.vstack([self._A, actions])
            self._Y = np.hstack([self._Y, rewards])
            self._P = np.vstack([self._P, probabilities])

        # Set update condition to be true at the end of the epoch.
        # Here is where we define the epoch schedule.
        if self._epoch_schedule == 0:
            update_condition = int(np.log2(max(self._t - 1, 1))) < int(
                np.log2(max(self._t - 1 + batch_len, 1))
            )
        else:
            # update_condition = (self._t % self._epoch_schedule == 0)
            update_condition = ((self._t - 1) // self._epoch_schedule) < (
                (self._t - 1 + batch_len) // self._epoch_schedule
            )

        # Update model at end of epoch (when update_condition is true)
        self._t += batch_len
        if update_condition:
            self._epoch += 1
            if self._t > self._min_uniform_samples:
                self._update()

    def _update(self) -> None:
        """_summary_
        Split stored data for training and error estimation.
        Training data is used to estimate a reward model for each action and save it in self.model.
        Error data is used to estimate error and set exploration parameters.
        """
        self.tracker["update_t"].append(self._t)
        update_start_time = time.time()

        assert len(self._X) > 0
        min_samples_per_action = 10

        # Train test split
        (
            X_train,
            X_error,
            A_train,
            A_error,
            Y_train,
            Y_error,
            P_train,
            P_error,
        ) = train_test_split(
            self._X,
            self._A,
            self._Y,
            self._P,
            test_size=self.error_data_ratio,
            shuffle=False,
        )

        # Train model for each action

        self.model = {}
        actions = self._actions
        for action in actions:
            idx = np.all(A_train == action, axis=1)
            X_a = X_train[idx]
            Y_a = Y_train[idx]

            if X_a.shape[0] < min_samples_per_action:
                return

            model_a = copy.deepcopy(self.regressor)
            model_a.fit(X_a, Y_a)
            self.model[action] = model_a

        # Estimate error and set exploration parameters.
        # To estimate errors, we first need predicted outcomes on error data.
        # Initialize outcome estimates.
        Y_hat_error = np.ones((len(X_error), len(actions)))
        Y_hat_all = np.ones((len(self._X), len(actions)))
        # Predict outcome for all past contexts
        for action in actions:
            Y_hat_error[:, np.argmax(action)] = self.model[action].predict(X_error)
            Y_hat_all[:, np.argmax(action)] = self.model[action].predict(self._X)
        Y_hat_train = Y_hat_all[: len(X_train)]

        # Learn base policy
        self.central_policy.fit(X_train, -Y_hat_train)

        modeltrain_end_time = time.time()
        self.tracker["update_modeltrain_time"].append(
            modeltrain_end_time - update_start_time
        )

        # Calculate localized error.

        localized_error_estimator = copy.deepcopy(self.localized_error_estimator)
        localized_error_estimator.fit(
            X_error,
            A_error,
            Y_error,
            P_error,
            Y_hat_error,
            X_train,
            Y_hat_train,
            self.central_policy,
            self._delta / pow(self._epoch, 2),
        )

        self._localized_error = localized_error_estimator._get_localized_error()

        errortrain_end_time = time.time()
        self.tracker["update_error_time"].append(
            errortrain_end_time - modeltrain_end_time
        )
        self.tracker["error"].append(self._localized_error)

        # Calculate risk adjustment and optimal cover.
        self._calculate_risk_adj()
        self.tracker["risk_adj"].append(self._risk_adj)

        # Eliminate arms
        armelim_start_time = time.time()
        if self.is_elim:
            self.arm_eliminator.fit(
                self._X,
                self._A,
                self._Y,
                self._P,
                Y_hat_all,
                self._localized_error,
                actions,
                self.model,
                self.central_policy,
                training_data=(self._X, self._A, self._Y, self._P),
            )
            self.arm_eliminator.default_eliminator()
            self.arm_eliminator_list.append(copy.deepcopy(self.arm_eliminator))

        armelim_end_time = time.time()
        self.tracker["update_armelimination_time"].append(
            armelim_end_time - armelim_start_time
        )
        self.tracker["update_total_time"].append(armelim_end_time - update_start_time)

        self._expected_diff = self._risk_adj * self._localized_error

    def _get_most_risky_CASs(
        self, max_beta=None, risk_adj=None, Y_hat=None, Y_greedy=None
    ):
        """_summary_
        Conformal arm sets follow a simple construction. They contain arms with estimated values sufficiently close to the optimal.
        Args:
            max_beta (float): threshold parameter in proportional response. Defaults to None.
            risk_adj (float): risk adjustment parameter. Defaults to None.

        Returns:
            array: Set of arms in the most risky conformal arm sets (using threshold beta value) for every observed context.
        """
        if max_beta == None:
            max_beta = self._max_beta
        if risk_adj == None:
            risk_adj = self._risk_adj

        # Construct conformal arm sets for each observed context corresponding to most risky choice of beta.
        expected_diff = risk_adj * self._localized_error

        # For every action and observed context, get risk maximum beta for which the arm would be included in the corresponding set. (beta_threshold)
        beta_thresholds = np.minimum(
            expected_diff / np.maximum(Y_greedy - Y_hat, expected_diff),
            np.ones(np.shape(Y_hat)),
        )
        # Get arms at each observed context with beta_threshold above beta_max.
        # These correspond to arms in the most risky conformal arm sets.
        above_threshold = beta_thresholds >= max_beta
        return above_threshold

    def _get_most_risky_setsize(
        self, max_beta=None, risk_adj=None, Y_hat=None, Y_greedy=None
    ):
        """_summary_

        Args:
            max_beta (float): threshold parameter in proportional response. Defaults to None.
            risk_adj (float): risk adjustment parameter. Defaults to None.

        Returns:
            float: Avg. size of the most risky conformal arm sets (using threshold beta value) for every observed context.
        """
        if max_beta == None:
            max_beta = self._max_beta
        if risk_adj == None:
            risk_adj = self._risk_adj
        K = len(self._actions)

        # For observed/prior contexts, get risky conformal arm sets and calculate arm set size.
        above_threshold = self._get_most_risky_CASs(
            max_beta=max_beta, risk_adj=risk_adj, Y_hat=Y_hat, Y_greedy=Y_greedy
        )
        arms_for_prior_contexts = above_threshold
        setsize_for_prior_contexts = np.sum(arms_for_prior_contexts, 1)

        # Now use set sizes for prior contexts to calculate
        avg_set_size = np.sum(setsize_for_prior_contexts) / len(
            setsize_for_prior_contexts
        ) + K * np.sqrt(1 / len(setsize_for_prior_contexts))
        avg_set_size = min(K, avg_set_size)
        return avg_set_size

    def _calculate_risk_adj(self):
        """_summary_
        Find largest risk_adj parameter such that:
        - avg_set_size/(1 - self._max_beta) <= K / risk_adj
        - risk_adj <= risk_adj_bound = np.sqrt(omega * K / alpha)

        __todo__:
        - Current approach to risk_adj may be more concervative than necessary. (ideally should fix this)
        - After arm elimination, risk is less. (should incorporate this)
        """
        K = len(self._actions)
        actions = self._actions
        omega = self._omega
        alpha = self._alpha
        risk_adj_bound = np.sqrt(omega * K / alpha)
        risk_adj = 1

        # First need predicted outcomes on observed data.
        # Initialize outcome estimates
        Y_hat = np.ones((len(self._X), len(actions)))
        # Predict outcome for all past contexts
        for action in actions:
            Y_hat[:, np.argmax(action)] = self.model[action].predict(self._X)
        # Greedy corresponds to central policy.
        central_arms = self.central_policy.predict(self._X)
        Y_greedy = Y_hat[np.arange(len(self._X)), central_arms]
        # Get a single column of greedy estimates
        Y_greedy = Y_greedy.reshape((-1, 1))

        # Searching for risk_adj
        while True:
            candidate_risk_adj = risk_adj + 0.1
            avg_set_size = self._get_most_risky_setsize(
                risk_adj=candidate_risk_adj, Y_hat=Y_hat, Y_greedy=Y_greedy
            )
            if (
                candidate_risk_adj <= risk_adj_bound
                and avg_set_size <= (1 - self._max_beta) * K / candidate_risk_adj
            ):
                risk_adj = candidate_risk_adj
            else:
                break

        self._risk_adj = risk_adj

    def predict(self, context, actions, is_batch=False):
        """_summary_
        Output probabilities given context and list of actions.

        Args:
            context (): context
            actions (): list of actions

        Returns:
            Probs: probability over actions.
        """
        if is_batch:
            contexts = context
            pass
        else:
            # Drop contextual features that are not numbers.
            context = self._get_valid_context(context)
            contexts = [context]

        # Update bandit parameters based on first sample.
        if self._alpha == None:
            min_uniform_samples = (
                len(actions) * sum([1.0 / n for n in range(1, len(actions) + 1)]) + 10
            )  # coupon collector
            self._min_uniform_samples = min_uniform_samples * 3
            self._actions = actions
            self._alpha = len(actions)
            if self._omega == 0:
                self._omega = len(actions)

        # uniform policy if not enough actions have been sampled
        if len(self.model) < len(self._actions) or self._t <= self._min_uniform_samples:
            probs = np.ones((len(contexts), len(actions))) / len(actions)
        # else can use past data to construct probability over arms.
        else:
            batch_size, K = len(contexts), len(actions)

            # Predict rewards for all arms (theta_hat) and get estimated optimal reward (theta_max).
            theta_hat = np.ones((batch_size, K))
            # Predict outcome for all past contexts
            for action in actions:
                theta_hat[:, np.argmax(action)] = self.model[action].predict(contexts)
            # theta_max = np.max(theta_hat, axis=1)
            central_arms = self.central_policy.predict(contexts)
            theta_max = theta_hat[np.arange(batch_size), central_arms]
            # Get a single column of greedy estimates
            theta_max = theta_max.reshape((-1, 1))

            # Get non eliminated arms.
            if self.is_elim:
                surviving_arms = self.arm_eliminator.predict(contexts, actions)
                s_arms = np.ones((batch_size, K))
                for elim in self.arm_eliminator_list:
                    s_arms = s_arms * elim.predict(contexts, actions)
                surviving_arms = (
                    1 - self.sequential_elim_fraction
                ) * surviving_arms + self.sequential_elim_fraction * s_arms
            else:
                surviving_arms = np.ones((batch_size, K))

            # Recall, the parameter beta controls the size of conformal arm set.
            # For every arm, construct largest choise of beta for which the arm is in our set.
            self._expected_diff = self._localized_error * self._risk_adj
            beta_thresholds = np.minimum(
                self._expected_diff
                / np.maximum(theta_max - theta_hat, self._expected_diff),
                np.ones(np.shape(theta_hat)),
            )
            sorted_beta_thresholds = np.sort(beta_thresholds, axis=1)

            # With both the beta_thresholds and sorted_beta_thresholds, we can construct the probability distribution over arms.
            probs = 0
            old_beta = 0
            for beta in sorted_beta_thresholds.T:
                beta = np.expand_dims(beta, axis=1)
                # Get conformal arms (ones above threshold).
                conformal_arms = beta_thresholds >= np.minimum(beta, self._max_beta)
                conformal_probs = conformal_arms / np.expand_dims(
                    np.sum(conformal_arms, axis=1), axis=1
                )
                # print("conformal_probs", conformal_probs)
                # Remove eliminated arms for current context. (except ones corresponding to the most risky conformal arm set)
                arm_set = conformal_arms * surviving_arms
                # arm_set = np.maximum(arm_set, beta_thresholds >= self._max_beta)
                arm_set = np.maximum(arm_set, theta_max - theta_hat == 0)
                # Get uniform sampling probabilityies from current set of arms (with an elimination trust threshold).
                elim_fraction = max(
                    self.elim_fraction,
                    np.sqrt(K * np.log(1 / self._delta) / len(self._X)),
                )  # Optimization procedure may not correctly eliminate in presense of less data.
                elim_fraction = min(elim_fraction, 1)
                candidate_probs = (
                    elim_fraction
                    * (arm_set)
                    / np.expand_dims(np.sum(arm_set, axis=1), axis=1)
                    + (1 - elim_fraction) * conformal_probs
                )
                # print("candidate_probs", candidate_probs)

                # uniform sampling from uncertainty set for beta in [0, _max_beta]. Already restricting conformal_arms to not consider beta >= self._max_beta.
                beta_diff = beta - old_beta
                probs += beta_diff * candidate_probs
                old_beta = beta

                # print("probs", probs)

        if is_batch:
            return probs
        else:
            return probs[0]

    def predict_out_of_sample(self, context, actions):
        """_summary_
        Output probabilities given context and list of actions. This is specifically for deploying the learnt policy at the end of the experiment.

        Args:
            context : context
            actions : list of actions

        Returns:
            Probs: probability over actions.
        """
        if self.is_out_of_sample == False:
            self.final_model = {}
            actions = self._actions
            for action in actions:
                idx = np.all(self._A == action, axis=1)
                X_a = self._X[idx]
                Y_a = self._Y[idx]

                model_a = copy.deepcopy(self.regressor)
                model_a.fit(X_a, Y_a)
                self.final_model[action] = model_a
            self.is_out_of_sample == True

        # Drop contextual features that are not numbers.
        context = self._get_valid_contexts(context)

        context = np.array(context).reshape(1, -1)
        theta_hat = np.ones(len(actions))
        for action in actions:
            theta_hat[np.argmax(action)] = self.final_model[action].predict(context)[0]
        theta_max = np.max(theta_hat)

        probs = theta_hat == theta_max
        probs = probs / len(probs)

        return probs

    def _get_valid_context(self, context):
        valid_contexts = []
        for c in context:
            if isinstance(c, (float, int)) or np.isreal(c):
                valid_contexts.append(c)
        return valid_contexts
