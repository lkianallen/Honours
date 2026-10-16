"""Which of the three training changes actually helps?

The changes are the rejection penalty, the queue-preference bonus and the adaptive
entropy floor (all in mappo.py). Every run resumes from the same converged
checkpoint, trains for the same number of extra rollouts, then gets scored on the
same held-out traces. Only the switches differ:

  control         none: the original objective, trained for the same extra rollouts
  rejection_only  rejection penalty on
  queue_only      queue-preference bonus on
  entropy_only    adaptive entropy floor on
  all_three       all of them (what finetune_routing.py does)

The control is what keeps this honest. Without it, "all three beat the checkpoint"
can't be told apart from "it simply trained for longer".

Each seed is one independent training run, and seeds are what the confidence
intervals and Welch's t-tests are built from. With only a few seeds the intervals
are wide and the tests have little power, so only big effects will show. Treat the
p-values as a rough guide: nothing is corrected for multiple comparisons.

Use the ORIGINAL converged checkpoint (training_run/checkpoint_rollout200.pt), not
one that's already been fine-tuned.

Evaluation reuses run_policy / run_heuristic from evaluate_baselines.py, so these
numbers are directly comparable with the baseline table.

Results go to ablation_run/. Each run is saved as it finishes, so an interrupted
job picks up where it left off if you rerun the same command.

Usage: python scripts/run_ablation.py [checkpoint] [--seeds 3] [--rollouts 100]
                                      [--eval-episodes 10] [--configs a,b,c]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

import argparse
import contextlib
import copy
import csv
import inspect
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from scipy import stats
except ImportError:
    sys.exit("This script needs scipy (confidence intervals, Welch's t-test): pip install scipy")

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer
from evaluate_baselines import (
    run_policy, run_heuristic,
    NUM_JURISDICTIONS, NODES_PER_JURISDICTION, NUM_NODES, MAX_ARRIVALS_PER_NODE, NODE_KWARGS,
)

NODE_IDS = list(range(1, NUM_NODES + 1))
TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
OUT_DIR = REPO_ROOT / "ablation_run"
RUNS_CSV = OUT_DIR / "ablation_runs.csv"
SUMMARY_CSV = OUT_DIR / "ablation_summary.csv"
PLOT_PATH = OUT_DIR / "ablation.png"

EVAL_SEED_BASE = 10_000   # eval traces are tagged from these seeds, identical for every run
EVAL_TORCH_SEED = 12345   # same sampling-noise start for every evaluation


def _default(name):
    return inspect.signature(MAPPOTrainer.__init__).parameters[name].default


# the "on" value of each switch is whatever MAPPOTrainer defaults to, so this can't drift
SWITCHES = {n: _default(n) for n in ("rejection_penalty_weight", "queue_preference_weight", "entropy_adapt_rate")}


def only(*names):
    return {n: (SWITCHES[n] if n in names else 0.0) for n in SWITCHES}


CONFIGS = {
    "control": only(),
    "rejection_only": only("rejection_penalty_weight"),
    "queue_only": only("queue_preference_weight"),
    "entropy_only": only("entropy_adapt_rate"),
    "all_three": only(*SWITCHES),
}

FIELDS = [
    "config", "seed", "rollouts", "eval_episodes", "checkpoint",
    "s_writes", "s_rejected", "s_violations", "s_reward",
    "a_writes", "a_rejected", "a_violations", "a_reward",
    "train_entropy", "train_queue_pref", "train_violation_rate", "seconds",
]
TEXT_FIELDS = {"config", "checkpoint"}


@contextlib.contextmanager
def quiet():
    """The simulator prints a line per tick; across all these runs that's tens of thousands of lines."""
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        yield


def finetune(trace, base_ckpt, switches, seed, rollouts):
    # Optimizer.load_state_dict can alias the tensors it's handed instead of copying them, so
    # without this every run would train on (and mutate) the shared checkpoint's Adam state and
    # inherit whatever the previous run left behind.
    base_ckpt = copy.deepcopy(base_ckpt)
    torch.manual_seed(seed)
    tag_rng = random.Random(seed)  # same tagging sequence for every config at a given seed

    def sampler():
        return trace.sample(node_ids=NODE_IDS, num_jurisdictions=NUM_JURISDICTIONS,
                            jurisdiction_strategy="realistic", seed=tag_rng.randrange(2**31))

    env = BlockchainShardingEnv(
        num_jurisdictions=NUM_JURISDICTIONS, nodes_per_jurisdiction=NODES_PER_JURISDICTION,
        max_ticks=len(trace.blocks), max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
        node_kwargs=NODE_KWARGS, trace_sampler_fn=sampler, seed=seed,
    )
    trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS,
                           ppo_epochs=4, **switches)
    trainer.actor.load_state_dict(base_ckpt["actor_state_dict"])
    trainer.critic.load_state_dict(base_ckpt["critic_state_dict"])
    if "actor_optim_state_dict" in base_ckpt:
        trainer.actor_optim.load_state_dict(base_ckpt["actor_optim_state_dict"])
    if "critic_optim_state_dict" in base_ckpt:
        trainer.critic_optim.load_state_dict(base_ckpt["critic_optim_state_dict"])

    history = []
    with quiet():
        for _ in range(rollouts):
            env.reset()
            trainer.collect_rollout(num_ticks=env.max_ticks)
            history.append(trainer.update())
    return trainer, history


def summarise(runs):
    return {
        "writes": float(np.mean([r["total_writes"] for r in runs])),
        "rejected": float(np.mean([r["total_rejected"] for r in runs])),
        "violations": float(np.mean([r["violation_rate"] * r["total_writes"] for r in runs])),
        "reward": float(np.mean([r["reward"] for r in runs])),
    }


def evaluate(trainer, eval_traces, deterministic):
    torch.manual_seed(EVAL_TORCH_SEED)
    with quiet():
        return summarise([run_policy(t, trainer, deterministic=deterministic) for t in eval_traces])


def one_run(config, seed, args, trace, base_ckpt, eval_traces):
    start = time.time()
    trainer, history = finetune(trace, base_ckpt, CONFIGS[config], seed, args.rollouts)
    sampled = evaluate(trainer, eval_traces, deterministic=False)
    argmax = evaluate(trainer, eval_traces, deterministic=True)
    tail = history[-10:]
    return {
        "config": config, "seed": seed, "rollouts": args.rollouts, "eval_episodes": args.eval_episodes,
        "checkpoint": str(args.checkpoint),
        "s_writes": sampled["writes"], "s_rejected": sampled["rejected"],
        "s_violations": sampled["violations"], "s_reward": sampled["reward"],
        "a_writes": argmax["writes"], "a_rejected": argmax["rejected"],
        "a_violations": argmax["violations"], "a_reward": argmax["reward"],
        "train_entropy": float(np.mean([h["entropy"] for h in tail])),
        "train_queue_pref": float(np.mean([h["queue_preference_rate"] for h in tail])),
        "train_violation_rate": float(np.mean([h["violation_rate"] for h in tail])),
        "seconds": time.time() - start,
    }


def load_rows():
    if not RUNS_CSV.exists():
        return []
    with open(RUNS_CSV) as f:
        return [{k: (v if k in TEXT_FIELDS else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def append_row(row):
    new = not RUNS_CSV.exists()
    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def ci95(values):
    if len(values) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if np.std(a) == 0 and np.std(b) == 0:  # no variance in either group: the t-test is 0/0
        return 1.0 if np.isclose(np.mean(a), np.mean(b)) else 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    # scipy warns when the groups are nearly identical and its own result is unreliable;
    # report n/a rather than a number it doesn't trust
    return float("nan") if any(issubclass(w.category, RuntimeWarning) for w in caught) else p


def build_summary(rows, configs):
    def per_seed(config, fn):
        return [fn(r) for r in rows if r["config"] == config]

    metrics = {
        "writes": lambda r: r["s_writes"],
        "rej_pct": lambda r: 100 * r["s_rejected"] / (r["s_writes"] + r["s_rejected"]),
        "violations": lambda r: r["s_violations"],
        "a_writes": lambda r: r["a_writes"],
        "entropy": lambda r: r["train_entropy"],
        "queue_pref": lambda r: 100 * r["train_queue_pref"],
    }
    summary = {}
    for c in configs:
        vals = {m: per_seed(c, fn) for m, fn in metrics.items()}
        summary[c] = {"n": len(vals["writes"]), "values": vals,
                      **{m: (float(np.mean(v)), ci95(v)) for m, v in vals.items()}}
    for c in configs:
        for m in ("writes", "violations"):
            summary[c]["p_" + m] = None if c == "control" else welch_p(summary[c]["values"][m], summary["control"]["values"][m])
    return summary


def fmt(pair, width=16):
    mean, ci = pair
    return f"{mean:>8.1f} ± {ci:<5.1f}".ljust(width) if not np.isnan(ci) else f"{mean:>8.1f}".ljust(width)


def fmt_p(p):
    return "   -   " if p is None else ("  n/a  " if np.isnan(p) else f"{p:7.3f}")


def print_summary(summary, refs, n_seeds):
    print("\nSampled policy, mean ± 95% CI across seeds (each seed = mean over the eval episodes)")
    print(f"{'config':<16}{'seeds':>5}  {'writes/ep':<16}{'rejected %':<16}{'violations/ep':<16}{'p(writes)':>10}{'p(viol)':>10}")
    print("-" * 94)
    for c, s in summary.items():
        print(f"{c:<16}{s['n']:>5}  {fmt(s['writes'])}{fmt(s['rej_pct'])}{fmt(s['violations'])}{fmt_p(s['p_writes']):>10}{fmt_p(s['p_violations']):>10}")
    print("(p-values are Welch's t-test against control)")

    print("\nTraining-time diagnostics (mean of the last 10 rollouts) and argmax throughput")
    print(f"{'config':<16}{'entropy':<16}{'queue-pref %':<16}{'argmax writes/ep':<18}")
    print("-" * 66)
    for c, s in summary.items():
        print(f"{c:<16}{fmt(s['entropy'])}{fmt(s['queue_pref'])}{fmt(s['a_writes'])}")

    print("\nReference on the same eval traces (no training; mean over eval episodes)")
    for name, r in refs.items():
        print(f"  {name:<22} writes/ep {r['writes']:8.1f}   rejected/ep {r['rejected']:8.1f}")
    print("  random_eligible and round_robin_eligible are the fair opponents; least_loaded needs a global")
    print("  view no agent has, so it's a ceiling.")
    if n_seeds < 5:
        crit = stats.t.ppf(0.975, max(n_seeds - 1, 1))
        print(f"\nWith {n_seeds} seeds each interval uses t = {crit:.2f}, so only large effects will clear the noise.")


def write_summary_csv(summary):
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seeds", "writes_mean", "writes_ci95", "rejected_pct_mean", "rejected_pct_ci95",
                    "violations_mean", "violations_ci95", "p_writes_vs_control", "p_violations_vs_control"])
        for c, s in summary.items():
            w.writerow([c, s["n"], *s["writes"], *s["rej_pct"], *s["violations"],
                        "" if s["p_writes"] is None else s["p_writes"],
                        "" if s["p_violations"] is None else s["p_violations"]])


def plot(summary, refs):
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [(axes[0], "writes", "Writes admitted per episode"),
                           (axes[1], "violations", "Violations per episode")]:
        means = [summary[n][key][0] for n in names]
        cis = [0 if np.isnan(summary[n][key][1]) else summary[n][key][1] for n in names]
        ax.bar(range(len(names)), means, yerr=cis, capsize=4, color="tab:blue", alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_title(title + " (sampled policy, 95% CI)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(refs["random_eligible"]["writes"], linestyle="--", color="gray", label="random_eligible")
    axes[0].axhline(refs["least_loaded_eligible"]["writes"], linestyle=":", color="gray", label="least_loaded (ceiling)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        default=REPO_ROOT / "training_run" / "checkpoint_rollout200.pt")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=100, help="extra rollouts per run")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset of: " + ", ".join(CONFIGS))
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. Choose from: {list(CONFIGS)}")
    if "control" not in configs:
        sys.exit("Include 'control': every comparison is made against it.")

    base_ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "finetune_rollout" in base_ckpt:
        print("WARNING: this checkpoint was already fine-tuned. The ablation should start from the "
              "original converged checkpoint, otherwise 'control' isn't a clean baseline.\n")

    OUT_DIR.mkdir(exist_ok=True)
    existing = load_rows()
    clash = [r for r in existing if (r["rollouts"], r["eval_episodes"], r["checkpoint"]) !=
             (args.rollouts, args.eval_episodes, str(args.checkpoint))]
    if clash:
        sys.exit(f"{RUNS_CSV} holds runs with different settings (rollouts / eval episodes / checkpoint). "
                 f"Delete it or move it aside before starting a new ablation.")
    done = {(r["config"], int(r["seed"])) for r in existing}

    trace = RetaggableTrace.from_files(TRACE_PATH)
    eval_traces = [trace.sample(node_ids=NODE_IDS, num_jurisdictions=NUM_JURISDICTIONS,
                                jurisdiction_strategy="realistic", seed=EVAL_SEED_BASE + i)
                   for i in range(args.eval_episodes)]
    print(f"Trace: {len(trace.blocks)} blocks | checkpoint: {args.checkpoint}")
    print(f"{len(configs)} configs x {args.seeds} seeds, {args.rollouts} extra rollouts each, "
          f"{args.eval_episodes} eval episodes\n")

    plan = [(c, s) for s in range(args.seeds) for c in configs]  # seed-major: finished seeds are complete
    todo = [p for p in plan if p not in done]
    print(f"{len(plan) - len(todo)} runs already done, {len(todo)} to go")
    started = time.time()
    for i, (config, seed) in enumerate(todo, 1):
        row = one_run(config, seed, args, trace, base_ckpt, eval_traces)
        append_row(row)
        elapsed = time.time() - started
        eta = elapsed / i * (len(todo) - i)
        print(f"[{i}/{len(todo)}] {config:<15} seed {seed}  writes {row['s_writes']:7.1f}  "
              f"violations {row['s_violations']:5.1f}  ({row['seconds']:.0f}s, ~{eta/60:.0f} min left)")

    rows = [r for r in load_rows() if r["config"] in configs and int(r["seed"]) < args.seeds]

    random.seed(EVAL_SEED_BASE)
    with quiet():
        refs = {}
        for name in ("random_eligible", "round_robin_eligible", "least_loaded_eligible"):
            refs[name] = summarise([run_heuristic(t, name) for t in eval_traces])

    summary = build_summary(rows, configs)
    print_summary(summary, refs, args.seeds)
    write_summary_csv(summary)
    plot(summary, refs)
    print(f"\nPer-run rows: {RUNS_CSV}\nSummary: {SUMMARY_CSV}\nPlot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
