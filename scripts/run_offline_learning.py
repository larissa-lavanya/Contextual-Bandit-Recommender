"""Part B: learn policies from biased (Bernoulli-TS) logs and check them against random-log ground truth."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bandit_ope` importable

import numpy as np
import pandas as pd

from bandit_ope.cli import add_data_args
from bandit_ope.data import load_logs, resolve_data_path
from bandit_ope.estimators import (
    bootstrap_ci,
    doubly_robust,
    effective_sample_size,
    estimate_value,
    importance_weights,
)
from bandit_ope.offline_learning import epsilon_greedy_action_dist, train_softmax_policy
from bandit_ope.plotting import plot_policy_values
from bandit_ope.reward_model import RidgeRewardModel, cross_fitted_predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--objectives", nargs="+", default=["ips", "snips", "dr"], choices=["ips", "snips", "dr"])
    parser.add_argument("--clips", nargs="+", type=float, default=[10.0, 100.0], help="Importance-weight caps")
    parser.add_argument("--n-steps", type=int, default=500, help="Gradient steps per learned policy")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results/offline")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both behavior policies: bts is used for learning, random only for ground truth
    data_path = resolve_data_path(args.data_path)
    logs = load_logs(data_path, args.campaign, ("random", "bts"), args.position, not args.no_affinity, args.max_rows)
    bts, rnd = logs["bts"], logs["random"]
    n_actions = bts.n_actions
    print(f"bts log: {bts.n_rounds} rounds | random log: {rnd.n_rounds} rounds | {n_actions} items")

    # Chronological split of the bts log: learn on the past, validate and test on the future
    train, val, test = bts.split_chronological((0.6, 0.2, 0.2))

    # Reward models: cross-fitted on train (for the DR objective), plain model for val/test
    q_train = cross_fitted_predictions(train.context, train.action, train.reward, n_actions, seed=args.seed)
    reward_model = RidgeRewardModel(n_actions).fit(train.context, train.action, train.reward)
    q_val = reward_model.predict(val.context)
    q_test = reward_model.predict(test.context)

    # Candidate policies, each represented as a function: context -> (n, K) action probabilities
    candidates = {
        "uniform": lambda ctx: np.full((len(ctx), n_actions), 1.0 / n_actions),
        "regression_greedy": lambda ctx: epsilon_greedy_action_dist(reward_model.predict(ctx), 0.0),
    }
    learned = []
    for objective in args.objectives:
        for clip in args.clips:
            print(f"Training {objective} policy with clip={clip:g} ...")
            policy = train_softmax_policy(
                train,
                objective=objective,
                q_hat=q_train if objective == "dr" else None,
                clip=clip,
                n_steps=args.n_steps,
                lr=args.lr,
                seed=args.seed,
            )
            name = f"{objective}_clip{clip:g}"
            candidates[name] = policy.action_dist
            learned.append(name)

    # Evaluate each candidate three ways: validation DR, test OPE estimates, random-log ground truth
    rows = []
    for name, dist_fn in candidates.items():
        val_dist, test_dist, gt_dist = dist_fn(val.context), dist_fn(test.context), dist_fn(rnd.context)

        row = {"policy": name}
        row["val_dr"] = doubly_robust(val_dist, val.action, val.reward, val.pscore, q_val)

        # Off-policy estimates computed from the bts test split only
        for estimator in ("dm", "ips", "snips", "dr"):
            row[f"test_{estimator}"] = estimate_value(
                estimator, test_dist, test.action, test.reward, test.pscore, q_test
            )
        row["ess_test"] = effective_sample_size(importance_weights(test_dist, test.action, test.pscore))

        # Ground truth: IPS on the random logs is unbiased because the logging policy was uniform
        gt_weights = importance_weights(gt_dist, rnd.action, rnd.pscore)
        row["gt_value"] = float(np.mean(gt_weights * rnd.reward))
        row["gt_ci_low"], row["gt_ci_high"] = bootstrap_ci(
            lambda idx: float(np.mean(gt_weights[idx] * rnd.reward[idx])), rnd.n_rounds, n_boot=200
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("gt_value", ascending=False).reset_index(drop=True)
    table.to_csv(out_dir / "offline_policy_results.csv", index=False)

    # Model selection the way it would work in production: pick the learned policy with the best validation DR
    learned_table = table[table["policy"].isin(learned)]
    selected = learned_table.loc[learned_table["val_dr"].idxmax()]

    print("\nReference CTRs on the random logs: %.5f   |   bts test split average CTR: %.5f" % (rnd.reward.mean(), test.reward.mean()))
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(
        f"\nSelected by validation DR: {selected['policy']} "
        f"(ground-truth CTR {selected['gt_value']:.5f}, 95% CI [{selected['gt_ci_low']:.5f}, {selected['gt_ci_high']:.5f}])"
    )

    plot_policy_values(table, out_dir / "offline_policy_values.png")
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
