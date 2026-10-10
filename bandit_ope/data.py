"""Load the Open Bandit Dataset (OBD) into numpy arrays for bandit experiments."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BanditFeedback:
    """Logged bandit data: one (context, action, reward, propensity) tuple per round."""

    context: np.ndarray  # (n_rounds, dim) user features
    action: np.ndarray  # (n_rounds,) index of the item that was shown
    reward: np.ndarray  # (n_rounds,) 1.0 if clicked, else 0.0
    pscore: np.ndarray  # (n_rounds,) probability the logging policy showed that item
    n_actions: int  # number of items in the campaign

    @property
    def n_rounds(self):
        return len(self.action)

    @property
    def dim(self):
        return self.context.shape[1]

    def subset(self, idx):
        """Return a new BanditFeedback containing only the rows in `idx`."""
        return BanditFeedback(
            context=self.context[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            pscore=self.pscore[idx],
            n_actions=self.n_actions,
        )

    def split_chronological(self, fractions):
        """Split into consecutive blocks (rows are already sorted by time)."""
        bounds = np.cumsum([0.0] + list(fractions)) / np.sum(fractions)
        edges = np.round(bounds * self.n_rounds).astype(int)
        return [self.subset(np.arange(lo, hi)) for lo, hi in zip(edges[:-1], edges[1:])]


class ContextEncoder:
    """One-hot encodes the user features with a fixed vocabulary.

    Fitting the vocabulary on every log we plan to use guarantees that logs from
    different behavior policies end up in the SAME feature space.
    """

    def __init__(self, use_affinity=True):
        self.use_affinity = use_affinity
        self.levels = {}

    def fit(self, logs):
        user_cols = [c for c in logs[0].columns if c.startswith("user_feature")]
        for col in user_cols:
            values = set()
            for log in logs:
                values |= set(log[col].astype(str))
            self.levels[col] = sorted(values)
        return self

    def transform(self, log):
        # Column 0 is a constant bias term
        blocks = [np.ones((len(log), 1))]

        # One-hot block for each categorical user feature
        for col, levels in self.levels.items():
            values = log[col].astype(str).to_numpy()[:, None]
            blocks.append((values == np.array(levels)[None, :]).astype(np.float64))

        # Optional block: affinity of this user to every item (one number per item)
        if self.use_affinity:
            aff_cols = [c for c in log.columns if c.startswith("user-item_affinity")]
            blocks.append(log[aff_cols].to_numpy(dtype=np.float64))

        return np.hstack(blocks)


def resolve_data_path(cli_path=None):
    """Find the OBD folder: CLI argument, $OBD_PATH, ./data, or the sample bundled with obp."""
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    if os.environ.get("OBD_PATH"):
        candidates.append(Path(os.environ["OBD_PATH"]))
    candidates.append(Path("data") / "open_bandit_dataset")

    # The obp package ships a 10k-row sample; locate it without importing obp
    spec = importlib.util.find_spec("obp")
    if spec is not None and spec.submodule_search_locations:
        candidates.append(Path(list(spec.submodule_search_locations)[0]) / "dataset" / "obd")

    for path in candidates:
        if (path / "random").is_dir():
            return path

    raise FileNotFoundError(
        "Could not find the Open Bandit Dataset. Pass --data-path, set OBD_PATH, "
        "put it in ./data/open_bandit_dataset, or run `pip install obp --no-deps` "
        "to use the small bundled sample. See the README for details."
    )


def _read_log(data_path, behavior_policy, campaign):
    """Read one raw CSV (e.g. random/men/men.csv) sorted by timestamp."""
    folder = Path(data_path) / behavior_policy / campaign
    log = pd.read_csv(folder / f"{campaign}.csv", index_col=0)
    log["timestamp"] = pd.to_datetime(log["timestamp"], format="ISO8601", utc=True)
    n_actions = len(pd.read_csv(folder / "item_context.csv", index_col=0))
    return log.sort_values("timestamp", kind="stable").reset_index(drop=True), n_actions


def load_logs(
    data_path,
    campaign="men",
    behavior_policies=("random", "bts"),
    position=1,
    use_affinity=True,
    max_rows=None,
):
    """Load several behavior-policy logs of one campaign into a shared feature space.

    Returns a dict {behavior_policy: BanditFeedback}. OBD positions are 1, 2 and 3;
    we keep a single position so every round is a plain one-item bandit decision.
    """
    raw = {}
    n_actions = None
    for policy in behavior_policies:
        raw[policy], n_actions = _read_log(data_path, policy, campaign)

    # Fit one encoder on all logs so the features line up across policies
    encoder = ContextEncoder(use_affinity).fit(list(raw.values()))

    feedback = {}
    for policy, log in raw.items():
        if position is not None:
            log = log[log["position"] == position].reset_index(drop=True)
        if max_rows is not None:
            log = log.iloc[:max_rows]
        feedback[policy] = BanditFeedback(
            context=encoder.transform(log),
            action=log["item_id"].to_numpy(dtype=np.int64),
            reward=log["click"].to_numpy(dtype=np.float64),
            pscore=log["propensity_score"].to_numpy(dtype=np.float64),
            n_actions=n_actions,
        )
    return feedback
