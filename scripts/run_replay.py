"""Part A: compare bandit algorithms on the Open Bandit Dataset with unbiased replay evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `bandit_ope` importable

import numpy as np
import pandas as pd

from bandit_ope.bandits import SWEEP_GRIDS, build_policy
from bandit_ope.cli import add_data_args
from bandit_ope.data import load_logs, resolve_data_path
from bandit_ope.estimators import bootstrap_ci
from bandit_ope.plotting import plot_final_ctr, plot_replay_curves
from bandit_ope.replay import replay_evaluate

ALL_ALGOS = ["random", "epsilon_greedy", "ucb", "thompson", "linucb", "linear_ts", "neural"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_args(parser)
    parser.add_argument("--algos", nargs="+", default=ALL_ALGOS, choices=ALL_ALGOS)
    parser.add_argument("--n-runs", type=int, default=3, help="Independent runs (the log is reshuffled each time)")
    parser.add_argument("--max-rounds", type=int, default=None, help="Stream at most N logged rounds per run")
    parser.add_argument("--sweep", action="store_true", help="Tune exploration parameters on a held-out split first")
    parser.add_argument("--tune-frac", type=float, default=0.2, help="Fraction of the log reserved for tuning")
    parser.add_argument("--progress", action="store_true", help="Show a progress bar for every run")
    parser.add_argument("--output-dir", default="results/replay")
    return parser.parse_args()


def run_algorithm(name, params, feedback, seeds, max_rounds, show_progress):
    """Replay one algorithm once per seed; every run streams the log in a different random order."""
    results = []
    for seed in seeds:
        # The shuffle uses its own seed stream so it never correlates with the policy's randomness
        order = np.random.default_rng([seed, 1]).permutation(feedback.n_rounds)
        policy = build_policy(name, feedback.n_actions, feedback.dim, seed, params)
        results.append(replay_evaluate(policy, feedback.subset(order), max_rounds, show_progress))
    return results


def tune_hyperparameters(algos, feedback, seed, max_rounds):
    """Pick the best setting of each algorithm's exploration parameter on the tuning split."""
    best_params = {}
    for name in algos:
        if name not in SWEEP_GRIDS:
            continue

        # Score every setting by the CTR it achieved over the whole run
        scored = []
        for params in SWEEP_GRIDS[name]:
            result = run_algorithm(name, params, feedback, [seed], max_rounds, False)[0]
            scored.append((-1.0 if np.isnan(result.ctr) else result.ctr, params))
            print(f"  tune {name:15s} {params} -> CTR {scored[-1][0]:.5f}")

        best_params[name] = max(scored, key=lambda s: s[0])[1]
    return best_params


def best_fixed_item_ctr(feedback, min_count=30):
    """CTR of the single best item in hindsight (only items shown at least `min_count` times)."""
    counts = np.bincount(feedback.action, minlength=feedback.n_actions)
    sums = np.bincount(feedback.action, weights=feedback.reward, minlength=feedback.n_actions)
    valid = counts >= min_count
    if not valid.any():
        valid = counts > 0
    return float((sums[valid] / counts[valid]).max())


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the uniform-random logs: the only logs for which replay is unbiased
    data_path = resolve_data_path(args.data_path)
    feedback = load_logs(
        data_path, args.campaign, ("random",), args.position, not args.no_affinity, args.max_rows
    )["random"]
    print(f"Loaded {feedback.n_rounds} rounds, {feedback.n_actions} items, {feedback.dim} features from {data_path}")

    # Optionally hold out a tuning split so hyperparameters are not tuned on the evaluation data
    best_params = {}
    eval_feedback = feedback
    if args.sweep:
        perm = np.random.default_rng(12345).permutation(feedback.n_rounds)
        n_tune = int(args.tune_frac * feedback.n_rounds)
        tune_feedback = feedback.subset(perm[:n_tune])
        eval_feedback = feedback.subset(perm[n_tune:])
        print(f"Tuning on {n_tune} rounds, evaluating on {eval_feedback.n_rounds} rounds")
        best_params = tune_hyperparameters(args.algos, tune_feedback, args.seed, args.max_rounds)

    random_ctr = float(eval_feedback.reward.mean())
    best_fixed_ctr = best_fixed_item_ctr(eval_feedback)
    print(f"Reference CTRs -> random: {random_ctr:.5f}, best fixed item: {best_fixed_ctr:.5f}")

    # Run every algorithm for several independent runs
    seeds = [args.seed + i for i in range(args.n_runs)]
    curves, rows = {}, []
    for name in args.algos:
        results = run_algorithm(name, best_params.get(name), eval_feedback, seeds, args.max_rounds, args.progress)
        curves[name] = [r.running_ctr() for r in results if r.n_matched > 0]

        # Summarize: whole-run CTR, second-half CTR, and a bootstrap CI for the second half
        late = np.array([r.late_ctr for r in results])
        whole = np.array([r.ctr for r in results])
        second_half = results[0].matched_rewards[results[0].n_matched // 2 :]
        low, high = bootstrap_ci(lambda idx: second_half[idx].mean(), len(second_half)) if len(second_half) else (np.nan, np.nan)
        rows.append(
            {
                "algorithm": name,
                "params": str(best_params.get(name, "default")),
                "n_matched": float(np.mean([r.n_matched for r in results])),
                "ctr_mean": whole.mean(),
                "late_ctr_mean": late.mean(),
                "late_ctr_std": late.std(),
                "ci_low": low,
                "ci_high": high,
                "lift_vs_random_pct": 100.0 * (late.mean() / random_ctr - 1.0),
            }
        )
        print(f"  {name:15s} late CTR {late.mean():.5f} (+-{late.std():.5f}) over {len(results)} runs")

    # Save the table and the figures
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "replay_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_replay_curves(curves, random_ctr, best_fixed_ctr, out_dir / "replay_curves.png")
    plot_final_ctr(summary, random_ctr, out_dir / "replay_final_ctr.png")
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
