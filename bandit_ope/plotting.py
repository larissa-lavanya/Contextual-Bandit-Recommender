"""Matplotlib helpers that save the figures used in the README."""

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np


def plot_replay_curves(curves, random_ctr, best_fixed_ctr, path, start=50):
    """Running CTR vs. number of matched rounds, averaged over runs (band = +-1 std).

    `curves` maps algorithm name -> list of running-CTR arrays, one per run.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, runs in curves.items():
        length = min(len(run) for run in runs)
        if length <= start:
            continue
        stacked = np.stack([run[:length] for run in runs])
        mean, std = stacked.mean(axis=0), stacked.std(axis=0)
        x = np.arange(start, length)
        ax.plot(x, mean[start:], label=name)
        ax.fill_between(x, (mean - std)[start:], (mean + std)[start:], alpha=0.15)

    # Reference lines: uniform-random policy and the best single item in hindsight
    ax.axhline(random_ctr, color="gray", linestyle="--", label="random (log average)")
    ax.axhline(best_fixed_ctr, color="black", linestyle=":", label="best fixed item (in hindsight)")

    ax.set_xlabel("matched rounds")
    ax.set_ylabel("running CTR estimate")
    ax.set_title("Replay evaluation: learning curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_final_ctr(summary, random_ctr, path):
    """Bar chart of the CTR over the second half of each run, with 95% bootstrap CIs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(summary))
    heights = summary["late_ctr_mean"].to_numpy()
    errors = np.vstack([heights - summary["ci_low"].to_numpy(), summary["ci_high"].to_numpy() - heights])
    ax.bar(x, heights, yerr=np.clip(errors, 0, None), capsize=4, color="tab:blue", alpha=0.8)
    ax.axhline(random_ctr, color="gray", linestyle="--", label="random")

    ax.set_xticks(x)
    ax.set_xticklabels(summary["algorithm"], rotation=30, ha="right")
    ax.set_ylabel("CTR (second half of the run)")
    ax.set_title("Final performance by algorithm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ope_error_vs_n(stats, path):
    """Relative RMSE of each estimator against the number of logged samples, one panel per policy."""
    policies = list(stats["policy"].unique())
    fig, axes = plt.subplots(1, len(policies), figsize=(4.2 * len(policies), 4), sharey=True, squeeze=False)

    for ax, policy in zip(axes[0], policies):
        subset = stats[(stats["policy"] == policy) & np.isinf(stats["clip"])]  # unclipped estimators only
        for estimator, group in subset.groupby("estimator"):
            group = group.sort_values("n")
            ax.plot(group["n"], group["rel_rmse"], marker="o", label=estimator.upper())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(policy)
        ax.set_xlabel("logged samples n")
    axes[0][0].set_ylabel("relative RMSE")
    axes[0][0].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy_values(table, path):
    """Ground-truth CTR of each policy (bars, with CI) next to the four offline estimates (markers)."""
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(table)), 4.8))

    x = np.arange(len(table))
    truth = table["gt_value"].to_numpy()
    errors = np.vstack([truth - table["gt_ci_low"].to_numpy(), table["gt_ci_high"].to_numpy() - truth])
    ax.bar(x, truth, yerr=np.clip(errors, 0, None), color="lightgray", capsize=3, label="ground truth (random logs)")

    # Offline estimates computed from the bts test split
    for estimator, marker in zip(("dm", "ips", "snips", "dr"), ("o", "s", "^", "D")):
        ax.scatter(x, table[f"test_{estimator}"], marker=marker, zorder=3, label=estimator.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(table["policy"], rotation=45, ha="right")
    ax.set_ylabel("estimated CTR")
    ax.set_title("Learned policies: offline estimates vs. ground truth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_clipping_tradeoff(clip_stats, path, title=""):
    """Bias, standard deviation and RMSE of clipped estimators as the clip level changes."""
    estimators = list(clip_stats["estimator"].unique())
    fig, axes = plt.subplots(1, len(estimators), figsize=(5.5 * len(estimators), 4), sharey=True, squeeze=False)

    for ax, estimator in zip(axes[0], estimators):
        group = clip_stats[clip_stats["estimator"] == estimator].sort_values("clip_value")
        ax.plot(group["clip_label"], group["abs_bias"], marker="o", label="|bias|")
        ax.plot(group["clip_label"], group["std"], marker="s", label="std")
        ax.plot(group["clip_label"], group["rmse"], marker="^", label="RMSE")
        ax.set_title(f"{estimator.upper()} {title}")
        ax.set_xlabel("weight clip level")
    axes[0][0].set_ylabel("error in estimated CTR")
    axes[0][0].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
