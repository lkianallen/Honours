"""Train the MAPPO policy on the real Sepolia trace. Writes a CSV log, periodic checkpoints and a convergence plot to training_run/."""

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
NODES_PER_JURISDICTION = 3   # N=9. For denser traces (~100 tx/block), consider N=30 (10 per jurisdiction) with K=10.
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20   # K: max arrivals per node per tick, sized for ~100 tx/block across 9 nodes

TRACE_PATH = REPO_ROOT / "sepolia_trace.json"  # from fetch_sepolia_trace.py
NUM_ROLLOUTS = 200                  # earlier runs needed ~150-200 rollouts to converge
CHECKPOINT_EVERY = 25
OUTPUT_DIR = REPO_ROOT / "training_run"

os.makedirs(OUTPUT_DIR, exist_ok=True)

trace = RetaggableTrace.from_files(TRACE_PATH)
node_ids = list(range(1, NUM_NODES + 1))
print(f"Loaded trace: {len(trace.blocks)} blocks, "
      f"{sum(len(b.get('transactions', [])) for b in trace.blocks)} total transactions")

sampler = partial(
    trace.sample,
    node_ids=node_ids,
    num_jurisdictions=NUM_JURISDICTIONS,
    jurisdiction_strategy="realistic",
    seed=None,  # None -> fresh random tagging every episode, by design
)

env = BlockchainShardingEnv(
    num_jurisdictions=NUM_JURISDICTIONS,
    nodes_per_jurisdiction=NODES_PER_JURISDICTION,
    max_ticks=len(trace.blocks),  # one tick per real block
    max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
    # mu_max/p raised from the defaults (10, 20), which rejected 63.6% of writes for
    # a full queue before jurisdiction was even checked. Hand-set, not derived from
    # Caliper (see the thesis dataset strategy).
    node_kwargs={"mu_max": 25.0, "p": 40},
    trace_sampler_fn=sampler,
    seed=1,
)

trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS, ppo_epochs=4)

log_path = os.path.join(OUTPUT_DIR, "training_log.csv")
log_fields = ["rollout", "mean_reward", "violation_rate", "total_writes", "total_rejected",
              "actor_loss", "critic_loss", "entropy"]

with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_fields)
    writer.writeheader()

    for rollout_num in range(1, NUM_ROLLOUTS + 1):
        env.reset()  # triggers a fresh jurisdiction/origin sample via trace_sampler_fn
        trainer.collect_rollout(num_ticks=env.max_ticks)
        diagnostics = trainer.update()

        row = {"rollout": rollout_num, **{k: diagnostics[k] for k in log_fields if k != "rollout"}}
        writer.writerow(row)
        f.flush()  # flush so you can tail -f the CSV while it runs

        print(
            f"rollout {rollout_num:3d} | "
            f"mean_reward={diagnostics['mean_reward']:9.2f} | "
            f"violation_rate={diagnostics['violation_rate']*100:5.1f}% | "
            f"rejected={diagnostics['total_rejected']:4d} | "
            f"actor_loss={diagnostics['actor_loss']:8.4f} | "
            f"critic_loss={diagnostics['critic_loss']:10.2f} | "
            f"entropy={diagnostics['entropy']:.4f}"
        )

        if rollout_num % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_rollout{rollout_num}.pt")
            torch.save(
                {
                    "rollout": rollout_num,
                    "actor_state_dict": trainer.actor.state_dict(),
                    "critic_state_dict": trainer.critic.state_dict(),
                    "actor_optim_state_dict": trainer.actor_optim.state_dict(),
                    "critic_optim_state_dict": trainer.critic_optim.state_dict(),
                },
                ckpt_path,
            )
            print(f"  -> checkpoint saved: {ckpt_path}")

print(f"\nTraining log written to {log_path}")

# --- convergence plot ---
rollouts, mean_rewards, violation_rates, entropies = [], [], [], []
with open(log_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rollouts.append(int(row["rollout"]))
        mean_rewards.append(float(row["mean_reward"]))
        violation_rates.append(float(row["violation_rate"]) * 100)
        entropies.append(float(row["entropy"]))

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

axes[0].plot(rollouts, mean_rewards)
axes[0].set_ylabel("Mean reward (raw)")
axes[0].set_title("Training convergence")
axes[0].grid(alpha=0.3)

axes[1].plot(rollouts, violation_rates, color="tab:red")
axes[1].axhline(66.7, color="gray", linestyle="--", linewidth=1, label="random baseline (~67%)")
axes[1].set_ylabel("Jurisdiction violation rate (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(rollouts, entropies, color="tab:green")
axes[2].axhline(2.197, color="gray", linestyle="--", linewidth=1, label="max entropy, N=9 (ln 9)")
axes[2].set_ylabel("Policy entropy")
axes[2].set_xlabel("Rollout")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "convergence.png")
plt.savefig(plot_path, dpi=150)
print(f"Convergence plot saved to {plot_path}")
