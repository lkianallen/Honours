"""Does the trained policy actually favour the least-loaded compliant node, or just pick any of them?

Replays the real trace; wherever a transaction had more than one compliant option, checks whether the policy's argmax choice was the one with the shortest queue, and compares that to chance (1 / number of options).

Usage: python diagnose_load_awareness.py training_run/checkpoint_rolloutN.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

from functools import partial

import torch

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer
from sim.observations import build_local_state, build_tx_features

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20

CHECKPOINT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "training_run" / "checkpoint_rollout200.pt"
TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
NUM_EVAL_ROLLOUTS = 5


def compliant_candidates(env, tx):
    """Every node this transaction could legally go to."""
    return [n for n in env.simulator.nodes if env.simulator.is_compliant_placement(tx, n)]


def least_loaded(nodes):
    return min(nodes, key=lambda n: len(n.write_queue))


def main():
    trace = RetaggableTrace.from_files(TRACE_PATH)
    node_ids = list(range(1, NUM_NODES + 1))
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

    trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    trainer.actor.load_state_dict(checkpoint["actor_state_dict"])
    trainer.actor.eval()
    print(f"Loaded checkpoint from rollout {checkpoint['rollout']}")

    total_multi_compliant_decisions = 0
    chose_least_loaded = 0
    chance_baseline_sum = 0.0

    with torch.no_grad():
        for _ in range(NUM_EVAL_ROLLOUTS):
            env.reset()
            for _tick in range(env.max_ticks):
                for agent_id, arrivals in list(env._pending_by_agent.items()):
                    node_id = env._node_id_from_agent(agent_id)
                    local_state = torch.tensor(
                        build_local_state(env.simulator._node_by_id(node_id), NUM_JURISDICTIONS),
                        dtype=torch.float32,
                    )
                    for tx in arrivals:
                        candidates = compliant_candidates(env, tx)
                        if len(candidates) < 2:
                            continue  # only one option, so nothing to learn from this decision

                        x = torch.cat([
                            local_state,
                            torch.tensor(build_tx_features(tx, NUM_JURISDICTIONS), dtype=torch.float32),
                        ])
                        # argmax: we want the policy's most confident choice, not a random sample
                        logits = trainer.actor.forward(x)
                        chosen_node_id = int(torch.argmax(logits).item()) + 1
                        chosen_node = env.simulator._node_by_id(chosen_node_id)

                        best = least_loaded(candidates)
                        total_multi_compliant_decisions += 1
                        if chosen_node.id == best.id:
                            chose_least_loaded += 1
                        chance_baseline_sum += 1.0 / len(candidates)

                # step the env with a simple stand-in policy so queues evolve between checks
                actions = {}
                for agent_id, arrivals in env._pending_by_agent.items():
                    node_id = env._node_id_from_agent(agent_id)
                    actions[agent_id] = [node_id] * len(arrivals)
                env.step(actions)

    if total_multi_compliant_decisions == 0:
        print("No decisions had >1 compliant candidate — can't answer this question with this trace/config.")
        return

    observed_rate = chose_least_loaded / total_multi_compliant_decisions
    chance_rate = chance_baseline_sum / total_multi_compliant_decisions

    print(f"\nDecisions with >1 compliant candidate: {total_multi_compliant_decisions}")
    print(f"Policy chose the least-loaded compliant node: {observed_rate*100:.1f}% of the time")
    print(f"Chance baseline (uniform among candidates):   {chance_rate*100:.1f}% of the time")

    if observed_rate > chance_rate * 1.3:
        print("\n-> Meaningfully above chance: real load-awareness. Entropy plateau is likely benign.")
    elif observed_rate < chance_rate * 0.8:
        print("\n-> BELOW chance — policy is avoiding the least-loaded node. Worth investigating further.")
    else:
        print("\n-> Indistinguishable from chance: the policy has NOT learned load-aware routing among")
        print("   compliant options. This supports extending the same credit-assignment fix used for")
        print("   compliance to a load/latency-shaped local signal.")


if __name__ == "__main__":
    main()
