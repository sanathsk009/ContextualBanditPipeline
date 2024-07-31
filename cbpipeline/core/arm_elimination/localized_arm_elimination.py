import numpy as np

from cbpipeline.core.arm_elimination.arm_elimination import BootstrapArmEliminator
from cbpipeline.cost_sensitive_classifiers import CSCRegressionOneVsRest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# Thought: Make this the main class and get other arm eliminator classes to inherit from this.
class LocalizedBootstrapArmEliminator(BootstrapArmEliminator):
    def __init__(
        self,
        costsensitive_classifier=CSCRegressionOneVsRest(),
        num_bootstraps=100,
        bootsrap_regressor=LinearRegression(),
    ) -> None:
        """_summary_

        Args:
        """
        self.csc_option = 2  # Adding it in case it is needed anywhere for inheritance.
        self.costsensitive_classifier = costsensitive_classifier
        self.num_bootstraps = num_bootstraps
        self.bootsrap_regressor = bootsrap_regressor

    def fit(
        self,
        X,
        A,
        Y,
        P,
        Y_hat,
        localized_error,
        actions,
        model,
        central_policy,
        training_data,
    ):
        """_summary_
        This fit function performs three tasks:
            - Stores data as class variables.
            - Calculates sub-optimality scores. Useful for evaluating estimated gap from estimated optimal policy (central_policy).
            - Calculates estimation gap to determine the threshold at which high-avg-sub-optimality scores prove sub-optimality.

        Args:
            X (np.array): Array of contexts.
            A (np.array): Array of selected arms. Each row is a one-hot vector indicating which arm is selected.
            Y (np.array): Vector of observed outcomes.
            P (np.array): Array of arm selection probabilities.
            Y_hat (np.array): Array of estimated arms rewards. Each row is a vector indicating estimated arm rewards.
            localized_error (float): Estimated error bound.
            model (dict): Reward model estimates for each arm.
            actions (list): list of actions
            central_policy: this is the estimated optimal policy (central_policy).
            training_data (tuple): (X, A, Y, P) from original training set (not error estimation).
        """
        # Save model
        self.model = model
        self.central_policy = central_policy

        # Store contexts.
        self.X = X
        # Datalength.
        self.datasize = len(self.X)
        # Save localized error.
        self.localized_error = localized_error
        # Save Y_hat
        self.Y_hat = Y_hat

        # Lower bound P
        P = np.maximum(P, max(1 / np.sqrt(self.datasize), localized_error))

        # Save training data in case it is needed training_data=(X_train, A_train, Y_train, P_train)
        self.training_data = training_data
        self.error_estimation_data = (X, A, Y, P)
        self.actions = actions

        # Calculate suboptimality scores. (Part 1)
        # Using only Y_hat, because only reward model error was measured.
        estimated_opt_arms = self.central_policy.predict(X)
        estimated_opt_value = np.mean(
            Y_hat[np.arange(self.datasize), estimated_opt_arms]
        )
        # Finally get suboptimality scores.
        self.subopt_scores = estimated_opt_value - Y_hat

        # Calculate estimated gap. (Part 2)
        # Simple under localization.
        self.estimated_gap = self.localized_error

    def predict(self, contexts, actions):
        theta_hat, interval = self._get_bootstrap_estimates_and_intervals(contexts)
        theta_hat = theta_hat
        interval = interval
        theta_hat_lower = theta_hat - self.bootstrap_interval_adjuster * interval
        theta_hat_upper = theta_hat + self.bootstrap_interval_adjuster * interval
        central_policy_arms = self.central_policy.predict(contexts)
        theta_hat_lower_greedy = theta_hat_lower[
            np.arange(len(contexts)), central_policy_arms
        ]
        theta_hat_lower_greedy = theta_hat_lower_greedy.reshape((-1, 1))
        surviving_arms = theta_hat_upper - theta_hat_lower_greedy >= 0
        return surviving_arms

    def _localized_bootstrap_eliminator(self):
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

            # Define greedy based on central policy.
            central_policy_arms = self.central_policy.predict(X_error)
            # Y_hat_bootstrap_lower_greedy = np.max(Y_hat_bootstrap_lower, axis=1)
            Y_hat_bootstrap_lower_greedy = Y_hat_bootstrap_lower[
                np.arange(len(X_error)), central_policy_arms
            ]
            Y_hat_bootstrap_lower_greedy = Y_hat_bootstrap_lower_greedy.reshape((-1, 1))
            PotOptArms = Y_hat_bootstrap_upper - Y_hat_bootstrap_lower_greedy >= 0

            true_if_near_opt_not_covered, _, _ = self.elimcheck(PotOptArms)
            if true_if_near_opt_not_covered:
                self.bootstrap_interval_adjuster = (
                    multiplier * self.bootstrap_interval_adjuster
                )

    def default_eliminator(self):
        self._localized_bootstrap_eliminator()
