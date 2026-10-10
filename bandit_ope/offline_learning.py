"""Learn a policy from logged data with importance-weighted (IPS / SNIPS / DR) objectives."""

import numpy as np
import torch
from torch import nn


class SoftmaxPolicy(nn.Module):
    """Linear softmax policy: pi(a | x) = softmax(W x + b)."""

    def __init__(self, dim, n_actions):
        super().__init__()
        self.linear = nn.Linear(dim, n_actions)

        # Start from the uniform policy
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=1)

    @torch.no_grad()
    def action_dist(self, context):
        """Return pi(a | x) as an (n, n_actions) numpy matrix."""
        x = torch.as_tensor(context, dtype=torch.float32)
        return torch.exp(self(x)).numpy().astype(np.float64)


def epsilon_greedy_action_dist(q_hat, epsilon):
    """Play the item with the highest predicted click rate, but explore with probability epsilon."""
    n, n_actions = q_hat.shape
    dist = np.full((n, n_actions), epsilon / n_actions)
    dist[np.arange(n), q_hat.argmax(axis=1)] += 1.0 - epsilon
    return dist


def _batch_value(objective, log_probs, action, reward, pscore, q_hat, clip):
    """Differentiable estimate of the policy value on one mini-batch."""
    probs = log_probs.exp()
    pi_taken = probs.gather(1, action[:, None]).squeeze(1)

    # Clipped importance weights (clipped entries get zero gradient)
    weights = torch.clamp(pi_taken / pscore, max=clip)

    if objective == "ips":
        return (weights * reward).mean()

    if objective == "snips":
        return (weights * reward).sum() / weights.sum().clamp_min(1e-8)

    if objective == "dr":
        baseline = (probs * q_hat).sum(dim=1)
        residual = reward - q_hat.gather(1, action[:, None]).squeeze(1)
        return (baseline + weights * residual).mean()

    raise ValueError(f"Unknown objective: {objective}")


def train_softmax_policy(
    feedback,
    objective="ips",
    q_hat=None,
    clip=10.0,
    n_steps=500,
    batch_size=4096,
    lr=0.05,
    l2=1e-4,
    entropy_coef=1e-3,
    seed=0,
):
    """Maximize an off-policy estimate of the policy value by mini-batch gradient ascent.

    `q_hat` (an (n, K) cross-fitted reward-model prediction) is required for objective='dr'.
    """
    if objective == "dr" and q_hat is None:
        raise ValueError("objective='dr' needs q_hat")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Move the logged data to tensors once
    x_all = torch.as_tensor(feedback.context, dtype=torch.float32)
    a_all = torch.as_tensor(feedback.action, dtype=torch.long)
    r_all = torch.as_tensor(feedback.reward, dtype=torch.float32)
    p_all = torch.as_tensor(feedback.pscore, dtype=torch.float32)
    q_all = None if q_hat is None else torch.as_tensor(q_hat, dtype=torch.float32)

    policy = SoftmaxPolicy(feedback.dim, feedback.n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    batch_size = min(batch_size, feedback.n_rounds)

    for _ in range(n_steps):
        # Sample a random mini-batch of logged rounds
        idx = torch.as_tensor(rng.integers(0, feedback.n_rounds, size=batch_size))
        log_probs = policy(x_all[idx])
        q_batch = None if q_all is None else q_all[idx]

        value = _batch_value(objective, log_probs, a_all[idx], r_all[idx], p_all[idx], q_batch, clip)

        # Entropy bonus keeps the policy from collapsing too early; L2 shrinks the weights
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        penalty = l2 * (policy.linear.weight**2).sum()
        loss = -(value + entropy_coef * entropy) + penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy
