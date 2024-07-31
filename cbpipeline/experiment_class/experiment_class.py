import math
import numpy as np
import copy
import scipy.stats as stats

from typing import Any, Dict, Sequence, Hashable
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    LassoCV,
    LogisticRegression,
    Ridge,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from cbpipeline.policy_learner import DMPolicyLearner


class ExperimentClass:
    def __init__(
        self,
        X,
        Y,
        exploration_horizon,
        cblearner=None,
        rounds_of_interest=None,
        policy_learner=DMPolicyLearner(),
        is_batch=True,
        batch_size=100,
        seed=42,
    ) -> None:
        """
        Initialize ExperimentClass.

        Args:
            X: Input array of contexts.
            Y: Input array of potential outcomes.
            exploration_horizon: Integer describing number of rounds for exploration.
            cblearner: Contextual bandit algorithm. Defaults to None. (None corresponds to RCT)
            rounds_of_interest: Array of rounds -- at which we evaluate value of learnt policy. Defaults to None. (None corresponds to empty array)
            policy_learner: __.
            is_batch: Boolian value for batching interactions with algorithm (if possible). Defaults to True.
            batch_size: Size of batch, if we are running algorithm in batches. Defaults to 100.
            seed: Int that defaults to 42. Unused at the moment.
        """
        # self._datasetX: stores contexts (each row is a context vector).
        self._datasetX = X
        # self._datasetY: stores outcomes (each row is a vector of outcomes).
        self._datasetY = Y
        # Get number of arms.
        self._K = len(self._datasetY[0])
        # self.actions: list of actions. Each action is a one hot tuple.
        self.actions = [tuple(a) for a in np.eye(self._K)]
        # self.cblearner: contextual bandit algorithm.
        self.cblearner = cblearner
        # self.policy_learner: learns a policy.
        self.policy_learner = policy_learner
        # self.exploration_horizon: round at which we stop adaptive experimentation.
        self.exploration_horizon = exploration_horizon
        # self.store: dictonary storing experiment details.
        self.store = {}
        # self._rounds_of_interest: rounds for which policy is learnt and evaluated.
        if rounds_of_interest == None:
            self._rounds_of_interest = np.array([])
        else:
            self._rounds_of_interest = np.array(rounds_of_interest)
        self._rounds_of_interest = np.append(
            self._rounds_of_interest, self.exploration_horizon
        )
        self._rounds_of_interest = np.unique(self._rounds_of_interest)

        # self.is_batch: true if is_batch and if self.cblearner allows for it.
        self.is_batch = (
            is_batch
            and (self.cblearner != None)
            and ("batch_option" in self.cblearner.params.keys())
            and (self.cblearner.params["batch_option"])
        )
        if self.is_batch:
            self.batch_size = batch_size
        else:
            self.batch_size = 1

        # history
        self._X: np.typing.NDArray[float] = []
        self._A: np.typing.NDArray[float] = []
        self._Y: np.typing.NDArray[float] = []
        self._P: np.typing.NDArray[float] = []

    def _run_experiment(self):
        # initialize self.store
        self.store["Avg_Exploration_Policy_Value"] = []
        self.store["Learned_Policy_Value"] = []
        self.store["t"] = []
        self.store["max_probs"] = []
        if self.cblearner == None:
            self.store["Algorithm"] = {"family": "RCT"}
        else:
            self.store["Algorithm"] = self.cblearner.params
        # run experiment
        if self.is_batch:
            self._run_batch_experiment()
        else:
            self._run_normal_experiment()

    def _run_batch_experiment(self):
        """
        Implements batched calls to self.cblearner.
        Speeds up simulation time: Contextual bandit algorithms run faster in batches.

        Returns:
            Dict: self.store
        """
        # Initialize round to zero.
        t = 0
        # Run adaptive experiment until t <= self.exploration_horizon.
        while t <= self.exploration_horizon:
            # t_end: Last round in batch + 1.
            t_end = min(t + self.batch_size, self.exploration_horizon + 1)
            # Get batch of contexts.
            contexts = self._datasetX[t:t_end]
            # Get probs from self.cblearner.
            probs = self.cblearner.predict(
                context=contexts, actions=self.actions, is_batch=True
            )
            # Sample action indices.
            action_index = np.apply_along_axis(
                lambda row: np.random.choice(self._K, p=row), axis=1, arr=probs
            )
            # Extract rewards and actions.
            rewards = self._datasetY[np.arange(t, t_end), action_index]
            actions = np.eye(self._K)[action_index]
            # Feed data to self.cblearner.
            if self.cblearner != None:
                self.cblearner.learn(
                    context=contexts,
                    action=actions,
                    reward=rewards,
                    probability=probs,
                    info=None,
                    is_batch=True,
                )

            # Update history
            self._update_history(
                context=contexts, action=actions, reward=rewards, probability=probs, t=t
            )

            # Update self.store
            for evaluation_round in self._rounds_of_interest:
                if (t < evaluation_round) and (t_end >= evaluation_round):
                    self._update_store(evaluation_round)

            t = t_end

        self.store["exploration_rewards"] = self._Y
        return self.store

    def _update_store(self, evaluation_round):
        evaluation_round = int(evaluation_round)  # Only to avoid issues due to 100.0.
        self.store["t"].append(evaluation_round)
        avg_exploration_policy_value = np.mean(self._Y[:evaluation_round])
        self.store["Avg_Exploration_Policy_Value"].append(avg_exploration_policy_value)
        # self._update_learnt_DM_policy(evaluation_round)
        self.policy_learner.fit(
            self._X[:evaluation_round],
            self._Y[:evaluation_round],
            self._A[:evaluation_round],
            self._P[:evaluation_round],
            self.actions,
        )
        # learnt_policy_value = self._evaluate_learnt_DM_policy()
        learnt_policy_value = self.policy_learner.evaluate(
            self._datasetX[self.exploration_horizon :],
            self._datasetY[self.exploration_horizon :],
        )
        self.store["Learned_Policy_Value"].append(learnt_policy_value)
        max_P = np.amax(self._P[:evaluation_round], axis=1)
        self.store["max_probs"].append((np.mean(max_P), np.std(max_P)))

    def _run_normal_experiment(self):
        for t in range(self.exploration_horizon):
            # Run the bandit
            context = tuple(self._datasetX[t])
            if self.cblearner == None:
                probs = np.ones(self._K) / self._K
            else:
                probs = self.cblearner.predict(context=context, actions=self.actions)
            action_index = np.random.choice(len(probs), p=probs)
            reward = self._datasetY[t][action_index]
            action = self.actions[action_index]
            if self.cblearner != None:
                self.cblearner.learn(
                    context=context,
                    action=action,
                    reward=reward,
                    probability=probs,
                    info=None,
                )

            # Update history
            self._update_history(
                context=context, action=action, reward=reward, probability=probs, t=t
            )
            # if (t in self._rounds_of_interest) or (t + 1 == self.exploration_horizon):
            if (t + 1) in self._rounds_of_interest:
                self._update_store(t + 1)

        self.store["exploration_rewards"] = self._Y
        return self.store

    def _update_history(
        self,
        context,
        action,
        reward: float,
        probability: float,
        t: float,
    ) -> None:
        # Update datasets self._X, self._A, and self._Y
        if self.is_batch:
            contexts = np.array(context, dtype=float)
            actions = np.array(action, dtype=float)
            rewards = np.array(reward, dtype=float)
            probabilities = np.array(probability, dtype=float)
        else:
            # Only let context be defined by numerical values.
            contexts = np.array([context], dtype=float)
            actions = np.array([action], dtype=float)
            rewards = np.array([reward], dtype=float)
            probabilities = np.array([probability], dtype=float)

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

    # def _update_learnt_DM_policy(self, evaluation_round):
    #     self.final_model = {}
    #     actions = self.actions
    #     A = self._A[:evaluation_round]
    #     X = self._X[:evaluation_round]
    #     Y = self._Y[:evaluation_round]
    #     for action in actions:
    #         idx = np.all(A == action, axis=1)
    #         X_a = X[idx]
    #         Y_a = Y[idx]

    #         model_a = copy.deepcopy(self.regressor)
    #         model_a.fit(X_a, Y_a)
    #         self.final_model[action] = model_a

    # def _evaluate_learnt_DM_policy(self):
    #     X_eval = self._datasetX[self.exploration_horizon :]
    #     Y_eval = self._datasetY[self.exploration_horizon :]
    #     actions = self.actions

    #     Y_hat = np.ones((len(X_eval), len(actions)))
    #     # Predict outcome for all eval contexts
    #     for action in actions:
    #         Y_hat[:, np.argmax(action)] = self.final_model[action].predict(X_eval)
    #     learnt_policy_arms = np.argmax(Y_hat, axis=1)
    #     learnt_policy_value = np.mean(
    #         Y_eval[np.arange(len(Y_eval)), learnt_policy_arms]
    #     )
    #     return learnt_policy_value
