"""Part C: how accurate are DM / IPS / SNIPS / DR? Compare each with random-log ground truth."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bandit_ope` importable

import numpy as np
import pandas as pd

from bandit_ope.cli import add_data_args
from bandit_ope.data import load_logs, resolve_data_path
from bandit_ope.estimators import estimate_value, ips
from bandit_ope.offline_learning import epsilon_greedy_action_dist
from bandit_ope.plotting import plot_clipping_tradeoff, plot_ope_error_vs_n
from bandit_ope.reward_model import RidgeRewardModel, cross_fitted_predictions

ESTIMATORS = ("dm", "ips", "snips", "dr")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1000, 3000, 10000, 30000])
    parser.add_argument("--n-reps", type=int, default=30, help="Random subsamples drawn per size")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 0.5, 0.1, 0.0],
                        help="Evaluation policies: epsilon-greedy on a reward model (1.0 = uniform)")
    parser.add_argument("--clips", nargs="+", type=float, default=[1, 3, 10, 30, 100, 300])
    parser.add_argument("--output-dir", default="results/ope")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load both logs: bts is what we evaluate FROM, random provides ground truth
    data_path = resolve_data_path(args.data_path)
    logs = load_logs(data_path, args.campaign, ("random", "bts"), args.position, not args.no_affinity, args.max_rows)
    bts, rnd = logs["bts"], logs["random"]
    n_actions = bts.n_actions

    # The first half of the random log defines the evaluation policies, the second half is ground truth
    rnd_define, rnd_truth = rnd.split_chronological((0.5, 0.5))
    policy_model = RidgeRewardModel(n_actions).fit(rnd_define.context, rnd_define.action, rnd_define.reward)

    # Ground-truth value of each policy: IPS on held-out random logs (unbiased, uniform logging)
    q_truth = policy_model.predict(rnd_truth.context)
    names = {eps: f"eps_greedy({eps:g})" for eps in args.epsilons}
    truth = {}
    for eps, name in names.items():
        dist = epsilon_greedy_action_dist(q_truth, eps)
        truth[name] = ips(dist, rnd_truth.action, rnd_truth.reward, rnd_truth.pscore)
        print(f"ground truth {name:18s} = {truth[name]:.5f}")

    # Only use sample sizes the bts log can actually supply
    sizes = sorted(s for s in args.sizes if s <= bts.n_rounds)
    if len(sizes) < len(args.sizes):
        print(f"Skipping sizes larger than the bts log ({bts.n_rounds} rounds).")
    clip_n = sizes[-1]  # the clipping study runs at the largest size

    # Repeatedly subsample the bts log and re-estimate every policy's value
    rng = np.random.default_rng(args.seed)
    records = []
    for n in sizes:
        print(f"n = {n}: {args.n_reps} repetitions")
        for rep in range(args.n_reps):
            sample = bts.subset(rng.choice(bts.n_rounds, size=n, replace=False))

            # Cross-fitted reward model (for DM and DR) and the policies' predicted click rates
            q_hat = cross_fitted_predictions(sample.context, sample.action, sample.reward, n_actions, seed=rep)
            q_policy = policy_model.predict(sample.context)

            for eps, name in names.items():
                dist = epsilon_greedy_action_dist(q_policy, eps)

                # Unclipped estimators
                for estimator in ESTIMATORS:
                    value = estimate_value(estimator, dist, sample.action, sample.reward, sample.pscore, q_hat)
                    records.append(dict(policy=name, n=n, rep=rep, estimator=estimator, clip=np.inf, value=value))

                # Clipped IPS and DR (only at the largest size, to keep the run fast)
                if n == clip_n:
                    for clip in args.clips:
                        for estimator in ("ips", "dr"):
                            value = estimate_value(
                                estimator, dist, sample.action, sample.reward, sample.pscore, q_hat, clip
                            )
                            records.append(dict(policy=name, n=n, rep=rep, estimator=estimator, clip=clip, value=value))

    # Aggregate repetitions into bias, standard deviation and RMSE
    df = pd.DataFrame(records)
    df["gt"] = df["policy"].map(truth)
    df["sq_err"] = (df["value"] - df["gt"]) ** 2
    stats = (
        df.groupby(["policy", "estimator", "n", "clip"])
        .agg(mean=("value", "mean"), std=("value", "std"), gt=("gt", "first"), mse=("sq_err", "mean"))
        .reset_index()
    )
    stats["bias"] = stats["mean"] - stats["gt"]
    stats["abs_bias"] = stats["bias"].abs()
    stats["rmse"] = np.sqrt(stats["mse"])
    stats["rel_rmse"] = stats["rmse"] / stats["gt"].where(stats["gt"] > 0)  # NaN if the true value is 0
    stats.to_csv(out_dir / "ope_validation.csv", index=False)

    # Print the unclipped estimators at the largest sample size
    summary = stats[(stats["n"] == clip_n) & np.isinf(stats["clip"])]
    table = summary.pivot(index="policy", columns="estimator", values="rel_rmse")[list(ESTIMATORS)]
    print(f"\nRelative RMSE at n = {clip_n} (lower is better):")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))
    bias_table = summary.pivot(index="policy", columns="estimator", values="bias")[list(ESTIMATORS)]
    print(f"\nBias at n = {clip_n} (estimate minus ground truth):")
    print(bias_table.to_string(float_format=lambda v: f"{v:+.5f}"))

    # Clipping study on the policy that is furthest from the logging policy
    focus = names[min(args.epsilons)]
    clip_stats = stats[(stats["policy"] == focus) & (stats["n"] == clip_n)].copy()
    clip_stats["clip_value"] = clip_stats["clip"]
    clip_stats["clip_label"] = clip_stats["clip"].map(lambda c: "none" if np.isinf(c) else f"{c:g}")
    clip_stats = clip_stats[clip_stats["estimator"].isin(["ips", "dr"])]

    plot_ope_error_vs_n(stats, out_dir / "ope_error_vs_n.png")
    plot_clipping_tradeoff(clip_stats, out_dir / "ope_clipping_tradeoff.png", title=f"on {focus}")
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
