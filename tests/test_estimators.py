"""Sanity checks for the off-policy estimators on synthetic data with a known true value."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bandit_ope.estimators import (
    bootstrap_ci,
    direct_method,
    doubly_robust,
    effective_sample_size,
    importance_weights,
    ips,
    snips,
)
from bandit_ope.offline_learning import epsilon_greedy_action_dist

N_ACTIONS = 5


def make_synthetic_logs(n=200_000, seed=0):
    """Uniform-random logging with a known click-probability table q(x, a)."""
    rng = np.random.default_rng(seed)
    n_contexts = 4

    # True click probability for every (context, action) pair
    q_true = rng.uniform(0.05, 0.6, size=(n_contexts, N_ACTIONS))
    context_id = rng.integers(0, n_contexts, size=n)
    action = rng.integers(0, N_ACTIONS, size=n)
    reward = (rng.random(n) < q_true[context_id, action]).astype(float)
    pscore = np.full(n, 1.0 / N_ACTIONS)
    return q_true, context_id, action, reward, pscore


def test_importance_weights_are_clipped():
    dist = np.array([[0.9, 0.1], [0.5, 0.5]])
    weights = importance_weights(dist, np.array([0, 1]), np.array([0.1, 0.5]), clip=5.0)
    assert np.allclose(weights, [5.0, 1.0])


def test_ips_snips_and_dr_recover_the_true_value():
    q_true, context_id, action, reward, pscore = make_synthetic_logs()

    # Evaluate a greedy policy and compute its exact value from the true table
    dist = epsilon_greedy_action_dist(q_true[context_id], 0.0)
    true_value = np.mean(np.sum(dist * q_true[context_id], axis=1))

    # A deliberately terrible reward model (constant 0.1) to show DR still works
    q_bad = np.full((len(action), N_ACTIONS), 0.1)

    assert abs(ips(dist, action, reward, pscore) - true_value) < 0.01
    assert abs(snips(dist, action, reward, pscore) - true_value) < 0.01
    assert abs(doubly_robust(dist, action, reward, pscore, q_bad) - true_value) < 0.01

    # DM with the same bad model is badly biased
    assert abs(direct_method(dist, q_bad) - true_value) > 0.05


def test_clipping_lowers_the_estimate_for_a_positive_reward_policy():
    q_true, context_id, action, reward, pscore = make_synthetic_logs()
    dist = epsilon_greedy_action_dist(q_true[context_id], 0.0)

    # Weights are 5 or 0 here, so clipping at 1 must shrink the IPS value
    assert ips(dist, action, reward, pscore, clip=1.0) < ips(dist, action, reward, pscore)


def test_snips_stays_in_the_reward_range():
    q_true, context_id, action, reward, pscore = make_synthetic_logs(n=5_000)
    dist = epsilon_greedy_action_dist(q_true[context_id], 0.1)
    assert 0.0 <= snips(dist, action, reward, pscore) <= 1.0


def test_effective_sample_size():
    assert np.isclose(effective_sample_size(np.ones(100)), 100.0)
    skewed = np.array([100.0] + [1.0] * 99)
    assert effective_sample_size(skewed) < 10.0


def test_bootstrap_ci_contains_the_mean():
    values = np.random.default_rng(0).normal(1.0, 1.0, size=2_000)
    low, high = bootstrap_ci(lambda idx: values[idx].mean(), len(values), n_boot=200)
    assert low < values.mean() < high
