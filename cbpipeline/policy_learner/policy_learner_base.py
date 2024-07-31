import numpy as np
import copy


class PolicyLearner:
    def __init__(self):
        pass

    def fit(self, X, Y, A, P, actions):
        pass

    def predict(self, X):
        # Return a vector of arm indices.
        pass

    def evaluate(self, X_eval, Y_eval):
        learnt_policy_arms = self.predict(X_eval)
        learnt_policy_value = np.mean(
            Y_eval[np.arange(len(Y_eval)), learnt_policy_arms]
        )
        return learnt_policy_value

    def _get_dr_scores(self, X, Y, A, P, actions, regressor):
        Y_ind = A * Y[:, np.newaxis]

        # Get cross fitted Y_hat.
        Y_hat = np.ones((len(X), len(actions)))  # initialize
        # split data for cross fitting.
        n = len(X)
        split = int(n / 2)
        # Compute cross fitted Y_hat.
        for action in actions:
            X_1, Y_1, A_1 = X[:split], Y[:split], A[:split]
            X_2, Y_2, A_2 = X[split:], Y[split:], A[split:]

            Y_hat[split:, np.argmax(action)] = self._get_plugin_for_out(
                X_1, Y_1, A_1, X_2, action, regressor
            )  # predictions for second half.
            Y_hat[:split, np.argmax(action)] = self._get_plugin_for_out(
                X_2, Y_2, A_2, X_1, action, regressor
            )  # predictions for first half.

        Y_hat_ind = np.multiply(A, Y_hat)

        return Y_hat + (Y_ind - Y_hat_ind) / P

    def _get_plugin_for_out(self, X_train, Y_train, A_train, X_out, action, regressor):
        idx = np.all(A_train == action, axis=1)
        X_a = X_train[idx]
        Y_a = Y_train[idx]
        model_a = copy.deepcopy(regressor)
        model_a.fit(X_a, Y_a)

        return model_a.predict(X_out)
