# Contextual Bandits and Off-Policy Evaluation on the Open Bandit Dataset

A fashion e-commerce site must decide **which item to show a user**. If the user clicks, the site earns a reward of 1, otherwise 0. This project builds, compares and *safely evaluates* policies for that decision using real logged data from ZOZOTOWN (the Open Bandit Dataset).

It answers two questions that every ads/recommendation team faces:

1. **Learning:** how do we find a good policy while still exploring? (bandit algorithms, a neural bandit, offline policy learning)
2. **Evaluation:** how do we know a policy is good **without deploying it**? (replay evaluation and off-policy estimators: DM, IPS, SNIPS, DR, weight clipping)

Every module is short, uses plain numpy/PyTorch, and implements the algorithms from scratch (the `obp` library is only used, optionally, to get a small sample of the data).

---

## Contents

1. [What the project does](#1-what-the-project-does)
2. [Background from the basics](#2-background-from-the-basics)
3. [The dataset](#3-the-dataset)
4. [Project structure](#4-project-structure)
5. [Setup](#5-setup)
6. [Running the experiments](#6-running-the-experiments)
7. [How each module works](#7-how-each-module-works)
8. [Design decisions and caveats](#8-design-decisions-and-caveats)
9. [Reading the results](#9-reading-the-results)
10. [Tests](#10-tests)
11. [Troubleshooting](#11-troubleshooting)
12. [Extending the project](#12-extending-the-project)
13. [Describing this project on a resume](#13-describing-this-project-on-a-resume)
14. [References](#14-references)

---

## 1. What the project does

| Part | Script | Question it answers |
|------|--------|---------------------|
| **A. Online-style comparison** | `scripts/run_replay.py` | Which bandit algorithm earns the most clicks? Context-free (epsilon-greedy, UCB, Thompson sampling) vs contextual (LinUCB, linear Thompson sampling, a neural bandit). Evaluated with *unbiased replay* on logs from a uniform-random policy. |
| **B. Offline policy learning** | `scripts/run_offline_learning.py` | Can we *learn* a policy from logs produced by a different (biased) policy, and pick the best one using only offline estimates? Ground truth comes from the random logs. |
| **C. Evaluator validation** | `scripts/run_ope_validation.py` | How accurate are DM, IPS, SNIPS and DR, how does their error shrink with data, and what does weight clipping do to bias and variance? |

---

## 2. Background from the basics

### 2.1 Multi-armed bandits

Imagine a row of slot machines (arms) with unknown payout rates. Each pull reveals the payout of only that machine. You face the **exploration vs exploitation** dilemma: keep pulling the best machine so far (exploit), or try others in case one is better (explore).

Here, the arms are the items in a campaign (34 for the "men" campaign), a "pull" is showing an item, and the reward is a click. Click rates are tiny (about 0.5%), so rewards are very noisy.

Success is measured by **regret**: the reward you lost relative to always showing the best item. Maximizing cumulative reward and minimizing regret are the same goal.

### 2.2 Context-free strategies (they ignore who the user is)

| Algorithm | Idea |
|-----------|------|
| **Epsilon-greedy** | With probability epsilon show a random item, otherwise show the item with the best observed click rate. |
| **UCB1** | Show the item maximizing `mean + c * sqrt(2 ln t / n_a)`. Rarely shown items get a bigger bonus, so they get tried ("optimism in the face of uncertainty"). |
| **Thompson sampling** | Keep a Beta posterior over each item's click rate, draw one sample per item, show the item with the highest draw. Uncertain items sometimes draw high, so exploration happens naturally. |

### 2.3 Contextual bandits (they use who the user is)

The best item depends on the user. A contextual bandit first sees a context `x` (user features), then chooses an item `a`, then observes a reward `r`.

* **LinUCB** assumes the click rate is linear in the context, with separate weights per item (a ridge regression per item). For each item it computes
  `score = x . theta_a + alpha * sqrt(x^T A_a^-1 x)`: a prediction plus an uncertainty bonus. The bonus is large when `x` points in a direction the item has rarely been shown in.
* **Linear Thompson sampling** uses the same ridge fit but *samples* each item's score from its Gaussian posterior instead of adding a fixed bonus.
* **Neural bandit** (here: an MLP with MC-dropout). One forward pass with dropout switched on behaves like sampling one plausible reward model. This is a heuristic posterior, not an exact one.

**Bandit vs full reinforcement learning.** In a bandit the action does not change the next state (the next user is independent). It is one-step RL. Full RL adds state transitions and long-term credit assignment.

### 2.4 The core difficulty: counterfactuals

Logs contain the reward of only the item that was *actually shown*. For a new policy that would have shown something else, we never saw what would have happened. Two tools solve this.

**Replay evaluation (Li et al., 2011).** If the logging policy chose items *uniformly at random*, stream the log and let your algorithm pick an item each round. If its pick equals the logged item, keep the round (count the reward, let the algorithm learn from it). Otherwise skip it. The kept rounds are a fair sample of what would have happened had the algorithm run live. Cost: only about `1/K` of the rounds are used.

**Off-policy evaluation (OPE).** Estimate a policy's value from logs of *any* policy, as long as we know the logging probabilities (**propensities**). Let `pi_e` be the policy we want to evaluate, `pi_b` the logging policy, `w = pi_e(a|x) / pi_b(a|x)`:

| Estimator | Formula | Character |
|-----------|---------|-----------|
| **DM** (Direct Method) | `mean( sum_a pi_e(a|x) * q_hat(x,a) )` | Low variance, but biased whenever the reward model `q_hat` is wrong. |
| **IPS** (Inverse Propensity Scoring) | `mean( w * r )` | Unbiased, but a rarely logged action gets a huge weight, so variance explodes. |
| **SNIPS** (Self-Normalized IPS) | `sum(w * r) / sum(w)` | Slightly biased, usually much lower variance. |
| **DR** (Doubly Robust) | `mean( sum_a pi_e q_hat + w * (r - q_hat(x, a_logged)) )` | Unbiased if *either* the propensities *or* the reward model is right. Usually the best overall. |
| **Clipping** | `w = min(w, lambda)` | Trades a little bias for much lower variance. |

Useful diagnostics: the **effective sample size** `(sum w)^2 / sum w^2` (how many equally weighted samples the weighted data is worth), and **cross-fitting** (predicting each row with a reward model that never saw that row, which avoids overfitting bias in DM and DR).

**Support condition.** IPS is only valid if `pi_b(a|x) > 0` wherever `pi_e(a|x) > 0`. A deterministic policy that picks an item the logging policy never showed cannot be evaluated.

### 2.5 Offline policy learning

Instead of only *evaluating* policies, we can *learn* one by maximizing an OPE estimate. Here a linear softmax policy `pi(a|x) = softmax(W x + b)` is trained by gradient ascent on a (clipped) IPS, SNIPS or DR objective, with a small entropy bonus and L2 penalty. Model selection is then done on a held-out validation split using the DR estimate, exactly as it would be done in production.

---

## 3. The dataset

**Open Bandit Dataset (OBD)** by ZOZO, Inc. It contains logged impressions from ZOZOTOWN's fashion recommendation system, collected during an A/B test of two policies:

* **`random`**: shows items uniformly at random. Because propensities are uniform, logs from this policy give unbiased replay and ground truth.
* **`bts`**: Bernoulli Thompson sampling, a context-free algorithm. Its propensities vary by item and over time, which makes it a realistic (and tricky) source for off-policy learning.

Each of three **campaigns** (`men`, `women`, `all`) has its own logs. Each impression shows items in **three slots (positions 1, 2, 3)**, and each row of the CSV is one (impression, position) pair. **This project studies a single position** (default: position 1, the top slot), so every round is a plain one-item bandit decision.

Columns of `<policy>/<campaign>/<campaign>.csv`:

| Column | Meaning |
|--------|---------|
| `timestamp` | When the impression happened. |
| `item_id` | The item shown (0 .. K-1). |
| `position` | Slot (1, 2 or 3). |
| `click` | 1 if the user clicked. |
| `propensity_score` | Probability that the logging policy showed this item. |
| `user_feature_0..3` | Four hashed categorical user features. |
| `user-item_affinity_0..K-1` | How strongly this user has interacted with each item before. |

**Context vector used here:** a constant 1 (bias), the one-hot encoding of the four user features, and (optionally) the K affinity values. Pass `--no-affinity` to drop the affinities.

### Getting the data

* **Quick start (sample only).** `pip install obp --no-deps` installs just the package files, which include a **10,000-row sample** per log. The code finds it automatically. After filtering to one position it is only about 3,300 rows, which gives roughly 80 matched rounds in replay and a handful of clicks. **Use it only to check that everything runs; the numbers are meaningless.**
* **Full dataset.** Download it from the ZOZO Research data page (`https://research.zozo.com/data.html`; the `obp` GitHub README also links to it if that page moves) and unzip it. You should end up with:

```
open_bandit_dataset/
  random/{men,women,all}/{men,women,all}.csv  + item_context.csv
  bts/{men,women,all}/{men,women,all}.csv     + item_context.csv
```

Point the scripts to it with `--data-path /path/to/open_bandit_dataset`, or set the environment variable `OBD_PATH`, or place it in `./data/open_bandit_dataset`. Lookup order: `--data-path`, `$OBD_PATH`, `./data/open_bandit_dataset`, the sample bundled with `obp`.

**Memory tip:** the `all` campaign is large. Start with `men` or `women`; use `--max-rows` to cap the number of rows per log.

---

## 4. Project structure

```
contextual-bandit-ope/
├── README.md
├── requirements.txt
├── bandit_ope/                  # the library
│   ├── data.py                  # load OBD, one-hot encode contexts, chronological splits
│   ├── bandits.py               # random, epsilon-greedy, UCB1, Thompson, LinUCB, linear TS, factory
│   ├── neural_bandit.py         # MLP bandit with MC-dropout exploration (PyTorch)
│   ├── replay.py                # unbiased replay evaluation
│   ├── estimators.py            # DM, IPS, SNIPS, DR, clipping, ESS, bootstrap CI
│   ├── reward_model.py          # per-item ridge reward model + cross-fitting
│   ├── offline_learning.py      # softmax policy trained with IPS / SNIPS / DR objectives
│   ├── plotting.py              # figures
│   └── cli.py                   # shared command-line options
├── scripts/
│   ├── run_replay.py            # Part A
│   ├── run_offline_learning.py  # Part B
│   └── run_ope_validation.py    # Part C
├── tests/                       # pytest unit tests (synthetic data with known answers)
└── results/                     # outputs are written here (CSVs and PNGs)
```

---

## 5. Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: the small bundled sample, for a quick check (no extra dependencies)
pip install obp --no-deps
```

Verify the install:

```bash
python -m pytest tests -q            # about 20-30 seconds, 18 tests
```

---

## 6. Running the experiments

All scripts run from the project root and share these data options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--data-path` | auto-detected | Folder containing `random/` and `bts/`. |
| `--campaign` | `men` | `men`, `women` or `all`. |
| `--position` | `1` | Which slot to study (1, 2 or 3). |
| `--no-affinity` | off | Drop the user-item affinity features (smaller, faster). |
| `--max-rows` | none | Keep only the first N rows of each log. |
| `--seed` | `0` | Random seed. |

### Part A: compare bandit algorithms

```bash
python scripts/run_replay.py --campaign men --sweep --n-runs 5
```

What happens:

1. Loads the **random** logs only (replay is only unbiased for uniform logging; the script checks this).
2. With `--sweep`, reserves 20% of the log (`--tune-frac`) to choose each algorithm's exploration parameter, then evaluates on the other 80%. This avoids tuning on the data you report.
3. For each algorithm, runs `--n-runs` independent runs. Each run reshuffles the log, so matched rounds differ.
4. Reports, per algorithm: CTR over the whole run, CTR over the **second half** of the run (after most of the learning), a 95% bootstrap CI for the second half, and the lift over random.

Useful flags: `--algos linucb linear_ts thompson` (subset), `--max-rounds 100000` (shorter runs), `--progress` (progress bars).

Outputs in `results/replay/`: `replay_summary.csv`, `replay_curves.png` (running CTR vs matched rounds, with reference lines for random and the best fixed item in hindsight), `replay_final_ctr.png`.

### Part B: learn a policy offline

```bash
python scripts/run_offline_learning.py --campaign men
```

What happens:

1. Loads the **bts** logs (for learning) and the **random** logs (for ground truth).
2. Splits the bts log chronologically: 60% train, 20% validation, 20% test.
3. Trains softmax policies with IPS, SNIPS and DR objectives and several clip levels (`--clips 10 100`), plus two baselines: `uniform` and `regression_greedy` (argmax of a ridge reward model).
4. Scores every policy three ways:
   * validation DR (used for model selection),
   * test estimates from the bts log (DM, IPS, SNIPS, DR),
   * **ground truth**: IPS on the random logs. The learned policies never see the random logs, and uniform logging makes IPS unbiased.
5. Prints which policy validation DR selected and what its ground-truth CTR is, with a bootstrap CI.

Outputs in `results/offline/`: `offline_policy_results.csv`, `offline_policy_values.png`.

### Part C: validate the evaluators

```bash
python scripts/run_ope_validation.py --campaign men \
    --sizes 1000 3000 10000 30000 100000 --n-reps 30
```

What happens:

1. Builds four evaluation policies of increasing distance from the logging policy: epsilon-greedy on a reward model with epsilon in `{1.0 (uniform), 0.5, 0.1, 0.0 (pure greedy)}`. The reward model is trained on the first half of the random logs.
2. Computes each policy's **ground-truth value** by IPS on the *second half* of the random logs (no leakage).
3. Repeatedly subsamples the bts log (`--sizes`, `--n-reps` times each), computes DM/IPS/SNIPS/DR with a cross-fitted reward model, and aggregates **bias, standard deviation and RMSE** against ground truth.
4. At the largest size, also sweeps the clip level (`--clips`) for IPS and DR on the policy furthest from the logger.

Outputs in `results/ope/`: `ope_validation.csv`, `ope_error_vs_n.png`, `ope_clipping_tradeoff.png`.

Sizes larger than the bts log are skipped with a message.

### Runtime

Measured on a synthetic log (34 items, 61 features): epsilon-greedy about 17 microseconds per round, UCB and Thompson about 25, LinUCB about 70, linear Thompson about 90, the neural bandit about 105. Runtime scales linearly with the number of rounds, and one run over a full single-slot log of the `men` campaign should take seconds to a few minutes per algorithm. The `all` campaign (80 items, more features) is several times heavier.

---

## 7. How each module works

**`data.py`**
`load_logs(...)` reads the CSVs, sorts by time, keeps one position and returns a `BanditFeedback` per behavior policy (context, action, reward, propensity, number of items). A `ContextEncoder` one-hot encodes the user features using a fixed vocabulary fitted on *all requested logs*, so the `random` and `bts` contexts live in the same feature space. `split_chronological` cuts a log into consecutive blocks.

**`bandits.py`**
All algorithms share one interface: `select_action(x)` and `update(x, action, reward)`. LinUCB and linear Thompson sampling share `_RidgePerItem`, which stores the inverse of each item's design matrix and updates it in O(d^2) with the Sherman-Morrison identity instead of re-inverting. Linear Thompson sampling samples the *scalar scores* `N(mean, v^2 x^T A^-1 x)` directly, which has the same distribution as sampling the weight vector but is much cheaper. `build_policy` is a factory, `DEFAULT_PARAMS` and `SWEEP_GRIDS` hold the settings.

**`neural_bandit.py`**
An MLP (two hidden layers of 64, dropout 0.2) outputs one click logit per item. Dropout stays on when choosing actions, so each choice samples a different sub-network. It acts randomly until the first training round, then retrains on the full buffer every 500 matched rounds (per-round gradient steps would be far too slow).

**`replay.py`**
`replay_evaluate` streams a log, keeps the rounds where the algorithm agrees with the logged item, and returns a `ReplayResult` (matched rewards, CTR, second-half CTR, running CTR).

**`estimators.py`**
Plain-numpy DM, IPS, SNIPS, DR with optional clipping, plus `effective_sample_size` and `bootstrap_ci`. Shapes are documented at the top of the file.

**`reward_model.py`**
`RidgeRewardModel` fits one ridge regression per item on the residual from the global click rate, so rarely shown items shrink toward the average. `cross_fitted_predictions` produces out-of-fold predictions.

**`offline_learning.py`**
`SoftmaxPolicy` starts as the uniform policy. `train_softmax_policy` samples mini-batches, computes a differentiable IPS/SNIPS/DR estimate with clipped weights (clipped entries get zero gradient), adds an entropy bonus and an L2 penalty, and takes Adam steps.

---

## 8. Design decisions and caveats

* **One position only.** The three slots of an impression are not independent decisions (the same impression shows three items). Restricting to one slot gives clean single-item propensities. A slate extension is listed in section 12.
* **Positions are 1-indexed** in OBD, so the default is `--position 1`, not 0.
* **Replay requires uniform logging.** That is why Part A uses only the `random` logs, and `replay_evaluate` raises an error otherwise.
* **Replay shuffles the log** within each run. This assumes rounds are i.i.d. (reasonable for a short A/B test, but real traffic has time-of-day and promotion effects).
* **The exploration scale must match the reward noise.** Clicks are rare, so reward noise is small (about `sqrt(p(1-p))`, roughly 0.07 at 0.5% CTR), and the defaults (`alpha = v = 0.05`, `c = 0.05`) reflect that. With the textbook scale of 1, UCB-style algorithms would explore almost uniformly. On a task with much bigger noise (such as the synthetic unit-test task) the right value is larger. This was observed directly while testing: with too small an `alpha`, LinUCB sometimes locks onto a suboptimal item. Use `--sweep`.
* **"Best fixed item in hindsight"** is chosen on the same data it is plotted against, so it is slightly optimistic. Only items shown at least 30 times are considered.
* **Ground truth is itself an estimate.** It is an unbiased IPS estimate from the random logs, so it has its own noise (the bootstrap CI is reported). It also assumes the random and bts logs come from the same user population and period. In OBD they were collected concurrently, but small differences are possible.
* **Chronological splits** are used wherever a model is trained, so that evaluation mimics deploying into the future.
* **Cross-fitting** is used for DM and DR whenever the reward model is trained on the same data that is being evaluated.
* **Vocabulary fitting.** The one-hot vocabulary is fitted on the logs you request in that run. Replay (random log only) may therefore have one or two fewer feature columns than Parts B and C (random plus bts). This does not affect correctness.
* **The neural bandit's exploration is heuristic.** MC-dropout is not an exact posterior.
* **Do not report numbers from the 10k-row sample.** They are dominated by noise.

---

## 9. Reading the results

Things you should be able to explain from the outputs:

* **Part A.** Contextual algorithms can only beat context-free ones if the features carry signal. Running CTR early in a run is extremely noisy (few matches), so the figure starts at 50 matched rounds and the table reports the **second-half CTR**, which reflects the learned behavior rather than the exploration phase. Compare each algorithm to the `random` row, which is a sanity check: its CTR should match the log's average click rate.
* **Part B.** Compare the estimates on the bts test split with the ground-truth column. Large gaps between `test_ips` and `gt_value` for aggressive policies mean the policy leaves the support of the logging policy; check `ess_test` (effective sample size) to see why. Check whether the policy chosen by validation DR is also near the top by ground truth.
* **Part C.** In the error-versus-size plot, expect error to shrink as `n` grows, with DM flattening out (bias does not vanish with more data) and IPS being noisiest for policies far from the logger. In the clipping plot, as the clip level rises the bias should fall and the standard deviation should rise; the best RMSE usually sits in the middle. These are typical patterns, not guarantees, so check them against your own run.

---

## 10. Tests

```bash
python -m pytest tests -q
```

The tests use synthetic data where the true answer is known:

* IPS, SNIPS and DR recover the true policy value; DM with a bad reward model does not; DR with the same bad model still does.
* Clipping, effective sample size and the bootstrap CI behave as expected.
* Replay keeps about `1/K` of the rounds, rejects non-uniform logs, and the random policy matches the log average.
* LinUCB and linear Thompson sampling beat context-free algorithms when the best item depends on the context; context-free algorithms find the best fixed item; the neural bandit learns something.
* Every policy satisfies the `select_action` / `update` interface.

---

## 11. Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Could not find the Open Bandit Dataset` | Pass `--data-path`, set `OBD_PATH`, or run `pip install obp --no-deps` for the sample. |
| `ValueError: Replay needs logs from the uniform-random behavior policy` | Replay only works on the `random` logs. |
| Out of memory | Use `--campaign men`, `--no-affinity`, or `--max-rows`. |
| Results look random or all zeros | You are probably on the 10k-row sample (about 80 matched rounds). Use the full dataset. |
| `NaN` relative RMSE | The ground-truth value of that policy is 0 (sample data), so a relative error is undefined. |
| `ModuleNotFoundError: torch` | `pip install torch`. It is only needed for the neural bandit and Part B; use `--algos` to skip `neural`. |
| `ModuleNotFoundError: bandit_ope` | Run scripts from the project root: `python scripts/run_replay.py`. |

---

## 12. Extending the project

* **Slate / multi-slot:** use all three positions and model position effects (for example, with a position-based click model).
* **Sequential decisions (full RL):** let the user's session be a state and show how bandits fall short; this connects naturally to long-horizon RL work.
* **Better reward models:** replace the ridge model with gradient-boosted trees or a neural network and see how DM and DR change.
* **Other estimators:** switching estimator, MRDR, or shrinkage-based weights (see the `obp` library for reference implementations to cross-check your code).
* **Cross-check with `obp`:** install `obp` fully and compare your IPS/DR numbers with `obp.ope` on the same data.
* **Spark:** the preprocessing here is small, but the same pipeline on the full `all` campaign can be a reasonable place to try PySpark.

---

## 13. Describing this project on a resume

Fill in the bracketed numbers from your own runs, and only claim what you ran.

> **Contextual Bandits and Off-Policy Evaluation (Open Bandit Dataset)** — GitHub
> * Implemented epsilon-greedy, UCB, Thompson sampling, LinUCB, linear Thompson sampling and a dropout-based neural bandit from scratch; compared them with unbiased replay on [N] logged impressions, with hyperparameters tuned on a held-out split ([X]% CTR lift over random).
> * Built DM, IPS, SNIPS and doubly-robust estimators with weight clipping and cross-fitting; validated them against random-log ground truth and quantified their bias-variance trade-off across [sizes] samples.
> * Learned softmax policies from biased logs with IPS/SNIPS/DR objectives and selected the best on validation DR ([metric]).

Be ready to answer: why is IPS high-variance? why does DR work if either model is correct? what is the support condition? why is replay only valid for uniform logging? why not shuffle a time-ordered dataset when training a model?

---

## 14. References

* Auer, Cesa-Bianchi, Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem* (UCB1).
* Li, Chu, Langford, Schapire (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation* (LinUCB).
* Li, Chu, Langford, Wang (2011). *Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms* (replay).
* Chapelle, Li (2011). *An Empirical Evaluation of Thompson Sampling*.
* Dudík, Langford, Li (2011). *Doubly Robust Policy Evaluation and Learning*.
* Swaminathan, Joachims (2015). *The Self-Normalized Estimator for Counterfactual Learning* (SNIPS).
* Gal, Ghahramani (2016). *Dropout as a Bayesian Approximation* (MC-dropout).
* Saito, Aihara, Matsutani, Narita (2021). *Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation* (the dataset and the `obp` library).
