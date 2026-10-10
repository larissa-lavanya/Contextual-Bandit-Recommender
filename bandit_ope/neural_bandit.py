"""A neural contextual bandit that explores through MC-dropout (an approximate Thompson sampling)."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from bandit_ope.bandits import BasePolicy


class DropoutNeuralBandit(BasePolicy):
    """An MLP predicts the click logit of every item; dropout stays ON when acting.

    Each call to select_action uses a different random sub-network, which behaves like
    sampling one plausible reward model (Gal & Ghahramani, 2016). It is a heuristic
    posterior, not an exact one, but it explores more where the network is unsure.
    """

    name = "neural"

    def __init__(
        self,
        n_actions,
        dim,
        hidden=64,
        dropout=0.2,
        lr=1e-3,
        retrain_every=500,
        epochs=3,
        batch_size=256,
        seed=0,
    ):
        super().__init__(n_actions, seed)
        torch.manual_seed(seed)

        # Network maps a context to one logit per item
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.retrain_every = retrain_every
        self.epochs = epochs
        self.batch_size = batch_size

        # Replay buffer of observed (context, item, reward) triples
        self.buffer_x, self.buffer_a, self.buffer_r = [], [], []
        self.new_since_training = 0
        self.is_trained = False

    def select_action(self, x):
        # Cold start: act randomly until the first training round
        if not self.is_trained:
            return int(self.rng.integers(self.n_actions))

        # Dropout is active in train mode, so every call samples a different sub-network
        self.net.train()
        with torch.no_grad():
            logits = self.net(torch.as_tensor(x, dtype=torch.float32))
        return self._argmax_random_ties(logits.numpy())

    def update(self, x, action, reward):
        self.buffer_x.append(np.asarray(x, dtype=np.float32))
        self.buffer_a.append(action)
        self.buffer_r.append(reward)
        self.new_since_training += 1

        # Retrain periodically (per-round gradient steps would be far too slow)
        if self.new_since_training >= self.retrain_every:
            self._fit()
            self.new_since_training = 0
            self.is_trained = True

    def _fit(self):
        """A few epochs of mini-batch training on everything seen so far."""
        x = torch.as_tensor(np.stack(self.buffer_x))
        a = torch.as_tensor(self.buffer_a, dtype=torch.long)
        r = torch.as_tensor(self.buffer_r, dtype=torch.float32)

        self.net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(a))
            for start in range(0, len(a), self.batch_size):
                batch = order[start : start + self.batch_size]

                # Only the logit of the item that was actually shown has a label
                chosen_logit = self.net(x[batch]).gather(1, a[batch][:, None]).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(chosen_logit, r[batch])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
