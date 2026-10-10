"""A simple per-item ridge-regression model of the click rate q(x, a)."""

import numpy as np


class RidgeRewardModel:
    """q_hat(x, a) = clip(mean_reward + x . theta_a, 0, 1), one ridge regression per item."""

    def __init__(self, n_actions, reg=10.0):
        self.n_actions = n_actions
        self.reg = reg
        self.mean_reward = 0.0
        self.theta = None

    def fit(self, context, action, reward):
        dim = context.shape[1]
        self.mean_reward = float(reward.mean())
        self.theta = np.zeros((self.n_actions, dim))

        # Fit on residuals so items with little data shrink toward the global click rate
        residual = reward - self.mean_reward
        for a in range(self.n_actions):
            rows = action == a
            if not rows.any():
                continue
            x_a = context[rows]
            gram = x_a.T @ x_a + self.reg * np.eye(dim)
            self.theta[a] = np.linalg.solve(gram, x_a.T @ residual[rows])
        return self

    def predict(self, context):
        """Return an (n, n_actions) matrix of predicted click rates."""
        return np.clip(self.mean_reward + context @ self.theta.T, 0.0, 1.0)


def cross_fitted_predictions(context, action, reward, n_actions, n_folds=2, reg=10.0, seed=0):
    """Predict each row with a model that never saw that row (avoids overfitting bias in DM/DR)."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, size=len(action))

    q_hat = np.zeros((len(action), n_actions))
    for k in range(n_folds):
        train, held_out = fold != k, fold == k
        model = RidgeRewardModel(n_actions, reg).fit(context[train], action[train], reward[train])
        q_hat[held_out] = model.predict(context[held_out])
    return q_hat
