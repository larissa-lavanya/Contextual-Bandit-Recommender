"""Off-policy estimators: Direct Method, IPS, SNIPS and Doubly Robust (with weight clipping).

Conventions used throughout:
    action_dist : (n, K) matrix, action_dist[i, a] = pi_e(a | x_i), the policy being evaluated
    action      : (n,) logged items
    reward      : (n,) logged clicks
    pscore      : (n,) logging probability pi_b(a_i | x_i)
    q_hat       : (n, K) predicted click rate for every (context, item) pair
"""

import numpy as np


def importance_weights(action_dist, action, pscore, clip=None):
    """w_i = pi_e(a_i | x_i) / pi_b(a_i | x_i), optionally capped at `clip`."""
    rows = np.arange(len(action))
    weights = action_dist[rows, action] / pscore
    if clip is not None:
        weights = np.minimum(weights, clip)
    return weights


def direct_method(action_dist, q_hat):
    """DM: average of the reward model's prediction under the new policy."""
    return float(np.mean(np.sum(action_dist * q_hat, axis=1)))


def ips(action_dist, action, reward, pscore, clip=None):
    """IPS: reweight logged rewards by how much more (or less) the new policy likes the action."""
    weights = importance_weights(action_dist, action, pscore, clip)
    return float(np.mean(weights * reward))


def snips(action_dist, action, reward, pscore, clip=None):
    """Self-normalized IPS: divide by the sum of weights instead of n (lower variance)."""
    weights = importance_weights(action_dist, action, pscore, clip)
    total = np.sum(weights)
    return float(np.sum(weights * reward) / total) if total > 0 else 0.0


def doubly_robust(action_dist, action, reward, pscore, q_hat, clip=None):
    """DR: DM baseline plus an importance-weighted correction of the model's residual."""
    rows = np.arange(len(action))
    weights = importance_weights(action_dist, action, pscore, clip)

    baseline = np.sum(action_dist * q_hat, axis=1)  # model's value of the new policy
    correction = weights * (reward - q_hat[rows, action])  # fix the model where data exists
    return float(np.mean(baseline + correction))


def estimate_value(estimator, action_dist, action, reward, pscore, q_hat=None, clip=None):
    """Dispatch to an estimator by name: 'dm', 'ips', 'snips' or 'dr'."""
    if estimator == "dm":
        return direct_method(action_dist, q_hat)
    if estimator == "ips":
        return ips(action_dist, action, reward, pscore, clip)
    if estimator == "snips":
        return snips(action_dist, action, reward, pscore, clip)
    if estimator == "dr":
        return doubly_robust(action_dist, action, reward, pscore, q_hat, clip)
    raise ValueError(f"Unknown estimator: {estimator}")


def effective_sample_size(weights):
    """(sum w)^2 / sum w^2: how many equally weighted samples the weighted data is worth."""
    denominator = np.sum(weights**2)
    return float(np.sum(weights) ** 2 / denominator) if denominator > 0 else 0.0


def bootstrap_ci(estimate_fn, n_rows, n_boot=300, alpha=0.05, seed=0):
    """Percentile bootstrap CI. `estimate_fn(idx)` computes the estimate on rows `idx`."""
    rng = np.random.default_rng(seed)
    values = [estimate_fn(rng.integers(0, n_rows, size=n_rows)) for _ in range(n_boot)]
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
