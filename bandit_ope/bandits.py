"""Bandit algorithms with a common interface: select_action(x) and update(x, a, r)."""

import numpy as np


class BasePolicy:
    """Interface every bandit algorithm implements."""

    name = "base"

    def __init__(self, n_actions, seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, x):
        raise NotImplementedError

    def update(self, x, action, reward):
        """Learn from the observed reward of the chosen action (default: no learning)."""

    def _argmax_random_ties(self, scores):
        """Argmax that breaks exact ties uniformly at random."""
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))


# ---------------------------------------------------------------------------
# Context-free algorithms (they ignore the user features x)
# ---------------------------------------------------------------------------


class RandomPolicy(BasePolicy):
    """Picks an item uniformly at random: the baseline every algorithm must beat."""

    name = "random"

    def select_action(self, x):
        return int(self.rng.integers(self.n_actions))


class EpsilonGreedy(BasePolicy):
    """With probability epsilon explore randomly, otherwise show the best-looking item."""

    name = "epsilon_greedy"

    def __init__(self, n_actions, epsilon=0.05, seed=0):
        super().__init__(n_actions, seed)
        self.epsilon = epsilon
        self.counts = np.zeros(n_actions)
        self.reward_sums = np.zeros(n_actions)

    def select_action(self, x):
        # Explore with probability epsilon
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        # Otherwise exploit the empirical click rate of each item
        means = self.reward_sums / np.maximum(self.counts, 1)
        return self._argmax_random_ties(means)

    def update(self, x, action, reward):
        self.counts[action] += 1
        self.reward_sums[action] += reward


class UCB1(BasePolicy):
    """Optimism in the face of uncertainty: mean + c * sqrt(2 ln t / n_a)."""

    name = "ucb"

    def __init__(self, n_actions, c=0.05, seed=0):
        super().__init__(n_actions, seed)
        self.c = c  # scales the bonus; clicks are rare so the textbook c=1 over-explores
        self.counts = np.zeros(n_actions)
        self.reward_sums = np.zeros(n_actions)

    def select_action(self, x):
        # Try every item once before trusting any statistics
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen) > 0:
            return int(self.rng.choice(unseen))

        # Score = empirical mean + exploration bonus that shrinks as an item is tried more
        means = self.reward_sums / self.counts
        total = self.counts.sum()
        bonus = self.c * np.sqrt(2.0 * np.log(total) / self.counts)
        return self._argmax_random_ties(means + bonus)

    def update(self, x, action, reward):
        self.counts[action] += 1
        self.reward_sums[action] += reward


class BetaThompsonSampling(BasePolicy):
    """Keep a Beta posterior over every item's click rate; play the argmax of one sample."""

    name = "thompson"

    def __init__(self, n_actions, prior_alpha=1.0, prior_beta=1.0, seed=0):
        super().__init__(n_actions, seed)
        self.alpha = np.full(n_actions, prior_alpha)  # prior + clicks
        self.beta = np.full(n_actions, prior_beta)  # prior + non-clicks

    def select_action(self, x):
        # Draw one plausible click rate per item and trust it for this round
        sampled_rates = self.rng.beta(self.alpha, self.beta)
        return self._argmax_random_ties(sampled_rates)

    def update(self, x, action, reward):
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward


# ---------------------------------------------------------------------------
# Contextual algorithms (one ridge regression per item)
# ---------------------------------------------------------------------------


class _RidgePerItem(BasePolicy):
    """Shared machinery: per-item ridge regression kept up to date incrementally."""

    def __init__(self, n_actions, dim, reg=1.0, seed=0):
        super().__init__(n_actions, seed)
        # A_inv[a] is the inverse of (reg * I + sum of x x^T over rounds where item a was shown)
        self.A_inv = np.tile(np.eye(dim) / reg, (n_actions, 1, 1))
        self.b = np.zeros((n_actions, dim))  # sum of reward * x for each item
        self.theta = np.zeros((n_actions, dim))  # ridge solution A^-1 b for each item

    def _mean_and_width(self, x):
        """Predicted click rate and its uncertainty (std in units of reward noise) per item."""
        mean = self.theta @ x  # (n_actions,)
        A_inv_x = self.A_inv @ x  # (n_actions, dim)
        width = np.sqrt(np.maximum(A_inv_x @ x, 0.0))  # sqrt(x^T A^-1 x) per item
        return mean, width

    def update(self, x, action, reward):
        # Sherman-Morrison: rank-one update of A^-1 without re-inverting the matrix
        A_inv_x = self.A_inv[action] @ x
        self.A_inv[action] -= np.outer(A_inv_x, A_inv_x) / (1.0 + x @ A_inv_x)
        self.b[action] += reward * x
        self.theta[action] = self.A_inv[action] @ self.b[action]


class LinUCB(_RidgePerItem):
    """Score = predicted click rate + alpha * uncertainty (disjoint LinUCB)."""

    name = "linucb"

    def __init__(self, n_actions, dim, alpha=0.05, reg=1.0, seed=0):
        super().__init__(n_actions, dim, reg, seed)
        self.alpha = alpha

    def select_action(self, x):
        mean, width = self._mean_and_width(x)
        return self._argmax_random_ties(mean + self.alpha * width)


class LinearThompsonSampling(_RidgePerItem):
    """Sample each item's score from its Gaussian posterior and play the best sample."""

    name = "linear_ts"

    def __init__(self, n_actions, dim, v=0.05, reg=1.0, seed=0):
        super().__init__(n_actions, dim, reg, seed)
        self.v = v  # posterior scale, roughly the std of the reward noise

    def select_action(self, x):
        # Sampling theta ~ N(theta_hat, v^2 A^-1) makes theta^T x ~ N(mean, v^2 x^T A^-1 x),
        # so we can sample the scalar scores directly instead of the whole weight vector
        mean, width = self._mean_and_width(x)
        sampled_scores = self.rng.normal(mean, self.v * width)
        return self._argmax_random_ties(sampled_scores)


# ---------------------------------------------------------------------------
# Factory and default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "random": {},
    "epsilon_greedy": {"epsilon": 0.05},
    "ucb": {"c": 0.05},
    "thompson": {},
    "linucb": {"alpha": 0.05},
    "linear_ts": {"v": 0.05},
    "neural": {},
}

# Small grids searched on the tuning split when --sweep is passed
SWEEP_GRIDS = {
    "epsilon_greedy": [{"epsilon": e} for e in (0.01, 0.05, 0.1)],
    "ucb": [{"c": c} for c in (0.01, 0.05, 0.2)],
    "linucb": [{"alpha": a} for a in (0.01, 0.05, 0.2)],
    "linear_ts": [{"v": v} for v in (0.01, 0.05, 0.2)],
}


def build_policy(name, n_actions, dim, seed=0, params=None):
    """Create a policy by name; `params` overrides the defaults."""
    params = {**DEFAULT_PARAMS[name], **(params or {})}

    if name == "random":
        return RandomPolicy(n_actions, seed=seed)
    if name == "epsilon_greedy":
        return EpsilonGreedy(n_actions, seed=seed, **params)
    if name == "ucb":
        return UCB1(n_actions, seed=seed, **params)
    if name == "thompson":
        return BetaThompsonSampling(n_actions, seed=seed, **params)
    if name == "linucb":
        return LinUCB(n_actions, dim, seed=seed, **params)
    if name == "linear_ts":
        return LinearThompsonSampling(n_actions, dim, seed=seed, **params)
    if name == "neural":
        from bandit_ope.neural_bandit import DropoutNeuralBandit  # needs torch

        return DropoutNeuralBandit(n_actions, dim, seed=seed, **params)

    raise ValueError(f"Unknown policy: {name}")
