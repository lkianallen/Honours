"""Sweep w1/w2 (Section 3.2.4) on the real Sepolia trace.

80 rollouts per trial is a time-boxed compromise: full convergence takes ~150-200, so this is enough to rank settings against each other but not to fully converge any of them. Evaluation is averaged over 3 episodes to cut noise. Cost is roughly N_TRIALS x ROLLOUTS_PER_TRIAL, so run it in the background.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

import statistics
from functools import partial

import optuna

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20

TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
ROLLOUTS_PER_TRIAL = 80   # time-boxed, see docstring
EVAL_EPISODES = 3          # averaged to cut noise
N_TRIALS = 12              # fewer than a rigorous sweep would use; cost scales with N_TRIALS x ROLLOUTS_PER_TRIAL

trace = RetaggableTrace.from_files(TRACE_PATH)
node_ids = list(range(1, NUM_NODES + 1))
print(f"Loaded trace: {len(trace.blocks)} blocks for the sweep")


def make_env(w1: float, w2: float, seed: int) -> BlockchainShardingEnv:
    sampler = partial(
        trace.sample, node_ids=node_ids, num_jurisdictions=NUM_JURISDICTIONS,
        jurisdiction_strategy="realistic", seed=None,  # fresh tagging each episode
    )
    return BlockchainShardingEnv(
        num_jurisdictions=NUM_JURISDICTIONS,
        nodes_per_jurisdiction=NODES_PER_JURISDICTION,
        max_ticks=len(trace.blocks),
        max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
        node_kwargs={"mu_max": 25.0, "p": 40},  # same calibration as train_sepolia.py
        simulator_kwargs={"w1": w1, "w2": w2},
        trace_sampler_fn=sampler,
        seed=seed,
    )


def objective(trial: optuna.Trial) -> float:
    w1 = trial.suggest_float("w1", 0.0, 1.0)
    w2 = 1.0 - w1  # enforce w1 + w2 = 1 per Section 3.2.4

    env = make_env(w1, w2, seed=trial.number)
    trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS, ppo_epochs=4)

    for _ in range(ROLLOUTS_PER_TRIAL):
        env.reset()
        trainer.collect_rollout(num_ticks=env.max_ticks)
        trainer.update()

    # Evaluate the partly trained policy over a few fresh episodes and average the raw
    # (unscaled) reward, which is what Optuna should be optimising.
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        env.reset()
        trainer.collect_rollout(num_ticks=env.max_ticks)
        eval_rewards.append(sum(trainer.buffer.raw_rewards) / len(trainer.buffer.raw_rewards))
        trainer.buffer.clear()

    mean_eval_reward = statistics.mean(eval_rewards)
    trial.set_user_attr("eval_reward_std", statistics.stdev(eval_rewards) if len(eval_rewards) > 1 else 0.0)
    return mean_eval_reward


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print()
    print("Best trial:")
    print(f"  w1 = {study.best_params['w1']:.4f}, w2 = {1 - study.best_params['w1']:.4f}")
    print(f"  eval mean reward = {study.best_value:.2f}")
    print()
    print("All trials:")
    for t in study.trials:
        std = t.user_attrs.get("eval_reward_std", float("nan"))
        print(f"  trial {t.number}: w1={t.params['w1']:.4f} -> reward={t.value:.2f} (std={std:.2f})")
