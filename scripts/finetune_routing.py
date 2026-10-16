"""Resume training from a converged checkpoint with the rejection-penalty, queue-preference and adaptive-entropy changes in mappo.py switched on.

Resuming beats starting over: compliance took ~200 rollouts to learn from scratch, and a fresh run would have to relearn it while also absorbing two new signals. Starting from the converged weights, it only has to improve on top of what already works.

Output goes to training_run_finetune/, separate from the original training_run/, so before/after comparisons stay clean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

import csv
import os
from functools import partial

import torch
import matplotlib.pyplot as plt

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20

TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
RESUME_FROM = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "training_run" / "checkpoint_rollout200.pt"
NUM_FINETUNE_ROLLOUTS = 100   # on top of the original 200
CHECKPOINT_EVERY = 25
OUTPUT_DIR = REPO_ROOT / "training_run_finetune"

os.makedirs(OUTPUT_DIR, exist_ok=True)

trace = RetaggableTrace.from_files(TRACE_PATH)
node_ids = list(range(1, NUM_NODES + 1))
print(f"Loaded trace: {len(trace.blocks)} blocks")

sampler = partial(
    trace.sample, node_ids=node_ids, num_jurisdictions=NUM_JURISDICTIONS,
    jurisdiction_strategy="realistic", seed=None,
)

env = BlockchainShardingEnv(
    num_jurisdictions=NUM_JURISDICTIONS,
    nodes_per_jurisdiction=NODES_PER_JURISDICTION,
    max_ticks=len(trace.blocks),
    max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
    node_kwargs={"mu_max": 25.0, "p": 40},
    trace_sampler_fn=sampler,
    seed=1,
)

trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS, ppo_epochs=4)

checkpoint = torch.load(RESUME_FROM, map_location="cpu")
trainer.actor.load_state_dict(checkpoint["actor_state_dict"])
trainer.critic.load_state_dict(checkpoint["critic_state_dict"])
if "actor_optim_state_dict" in checkpoint:
    trainer.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
if "critic_optim_state_dict" in checkpoint:
    trainer.critic_optim.load_state_dict(checkpoint["critic_optim_state_dict"])
print(f"Resumed from rollout {checkpoint['rollout']} ({RESUME_FROM})")

log_path = OUTPUT_DIR / "finetune_log.csv"
log_fields = ["rollout", "mean_reward", "violation_rate", "queue_preference_rate", "total_writes",
              "total_rejected", "actor_loss", "critic_loss", "entropy", "entropy_coef"]

with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_fields)
    writer.writeheader()

    for rollout_num in range(1, NUM_FINETUNE_ROLLOUTS + 1):
        env.reset()
        trainer.collect_rollout(num_ticks=env.max_ticks)
        diagnostics = trainer.update()

        row = {"rollout": rollout_num, **{k: diagnostics[k] for k in log_fields if k != "rollout"}}
        writer.writerow(row)
        f.flush()

        print(
            f"rollout {rollout_num:3d} | mean_reward={diagnostics['mean_reward']:9.2f} | "
            f"violation_rate={diagnostics['violation_rate']*100:5.1f}% | "
            f"queue_pref={diagnostics['queue_preference_rate']*100:5.1f}% | "
            f"rejected={diagnostics['total_rejected']:5d} | entropy={diagnostics['entropy']:.3f} | "
            f"entropy_coef={diagnostics['entropy_coef']:.4f}"
        )

        if rollout_num % CHECKPOINT_EVERY == 0:
            ckpt_path = OUTPUT_DIR / f"checkpoint_finetune_rollout{rollout_num}.pt"
            torch.save(
                {
                    "rollout": checkpoint["rollout"] + rollout_num,  # running total, including the original run
                    "finetune_rollout": rollout_num,
                    "resumed_from": str(RESUME_FROM),
                    "actor_state_dict": trainer.actor.state_dict(),
                    "critic_state_dict": trainer.critic.state_dict(),
                    "actor_optim_state_dict": trainer.actor_optim.state_dict(),
                    "critic_optim_state_dict": trainer.critic_optim.state_dict(),
                },
                ckpt_path,
            )
            print(f"  -> checkpoint saved: {ckpt_path}")

print(f"\nFine-tune log written to {log_path}")

# convergence plot (same as training, plus queue-preference rate)
rollouts, mean_rewards, violation_rates, queue_prefs, entropies = [], [], [], [], []
with open(log_path) as f:
    for row in csv.DictReader(f):
        rollouts.append(int(row["rollout"]))
        mean_rewards.append(float(row["mean_reward"]))
        violation_rates.append(float(row["violation_rate"]) * 100)
        queue_prefs.append(float(row["queue_preference_rate"]) * 100)
        entropies.append(float(row["entropy"]))

fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)

axes[0].plot(rollouts, mean_rewards)
axes[0].set_ylabel("Mean reward (raw)")
axes[0].set_title("Fine-tuning convergence (resumed from converged checkpoint)")
axes[0].grid(alpha=0.3)

axes[1].plot(rollouts, violation_rates, color="tab:red")
axes[1].axhline(66.7, color="gray", linestyle="--", linewidth=1, label="random baseline (~67%)")
axes[1].set_ylabel("Violation rate (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(rollouts, queue_prefs, color="tab:blue")
axes[2].axhline(33.3, color="gray", linestyle="--", linewidth=1, label="chance baseline (~33%, 3 candidates)")
axes[2].set_ylabel("Queue-preference rate (%)")
axes[2].legend()
axes[2].grid(alpha=0.3)

axes[3].plot(rollouts, entropies, color="tab:green")
axes[3].set_ylabel("Policy entropy")
axes[3].set_xlabel("Fine-tune rollout")
axes[3].grid(alpha=0.3)

plt.tight_layout()
plot_path = OUTPUT_DIR / "finetune_convergence.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
