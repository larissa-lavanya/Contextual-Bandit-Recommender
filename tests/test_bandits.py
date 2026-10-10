"""Behavioral tests for the bandit algorithms and the replay evaluator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from bandit_ope.bandits import build_policy
from bandit_ope.data import BanditFeedback
from bandit_ope.replay import replay_evaluate

N_ACTIONS = 3


def make_log(n=30_000, seed=1234):  # must differ from the policy seed, or their random draws correlate
    """Uniform-random log where the best item depends on which of two user groups we see."""
    rng = np.random.default_rng(seed)

    # Group 0 likes item 0, group 1 likes item 1 (item 2 is bad for both)
    q_true = np.array([[0.8, 0.2, 0.2], [0.2, 0.8, 0.2]])
    group = rng.integers(0, 2, size=n)
    context = np.stack([np.ones(n), group == 0, group == 1], axis=1).astype(float)
    action = rng.integers(0, N_ACTIONS, size=n)
    reward = (rng.random(n) < q_true[group, action]).astype(float)

    return BanditFeedback(context, action, reward, np.full(n, 1.0 / N_ACTIONS), N_ACTIONS)


def run(name, params=None, seed=0):
    log = make_log()
    policy = build_policy(name, N_ACTIONS, log.dim, seed=seed, params=params)
    return replay_evaluate(policy, log)


def test_replay_matches_about_one_in_k_rounds():
    result = run("random")
    assert abs(result.n_matched / result.n_rounds - 1.0 / N_ACTIONS) < 0.02


def test_replay_rejects_non_uniform_logs():
    log = make_log(n=100)
    log.pscore[:] = 0.9
    with pytest.raises(ValueError):
        replay_evaluate(build_policy("random", N_ACTIONS, log.dim), log)


def test_random_policy_matches_the_log_average():
    # Random picks give an unbiased estimate of the average reward in the log
    assert abs(run("random").ctr - make_log().reward.mean()) < 0.03


def test_contextual_algorithms_beat_context_free_ones():
    # The best fixed item earns 0.5 on average, a contextual policy can reach 0.8.
    # Exploration scales are matched to the reward noise here (std ~0.4), not to real CTR data
    linucb = run("linucb", {"alpha": 0.5})
    linear_ts = run("linear_ts", {"v": 0.5})
    context_free = run("epsilon_greedy")

    assert linucb.late_ctr > 0.7
    assert linear_ts.late_ctr > 0.7
    assert linucb.late_ctr > context_free.late_ctr + 0.1


def test_context_free_algorithms_find_the_best_fixed_item():
    for name in ("epsilon_greedy", "ucb", "thompson"):
        params = {"c": 0.3} if name == "ucb" else None
        assert run(name, params).late_ctr > 0.42, name


@pytest.mark.parametrize("name", ["random", "epsilon_greedy", "ucb", "thompson", "linucb", "linear_ts"])
def test_policy_interface(name):
    policy = build_policy(name, N_ACTIONS, dim=3, seed=1)
    x = np.array([1.0, 1.0, 0.0])
    for _ in range(20):
        action = policy.select_action(x)
        assert 0 <= action < N_ACTIONS
        policy.update(x, action, 1.0)


def test_neural_policy_learns_something():
    pytest.importorskip("torch")
    result = run("neural", {"retrain_every": 200, "epochs": 5})
    assert result.late_ctr > make_log().reward.mean()
