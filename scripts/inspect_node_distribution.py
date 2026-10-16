"""Is the policy spreading writes across the compliant nodes in each jurisdiction, or leaning on a favourite?

A load-reactive policy should land close to an even split (roughly 1/NODES_PER_JURISDICTION each), like random_eligible does by construction. A heavy, persistent skew toward one node suggests a fixed preference rather than real balancing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

from collections import defaultdict

import torch

from sim.data_sources import RetaggableTrace
from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer
from sim.observations import build_local_state, build_tx_features

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION
MAX_ARRIVALS_PER_NODE = 20
TRACE_PATH = REPO_ROOT / "sepolia_trace.json"
CHECKPOINT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "training_run" / "checkpoint_rollout200.pt"


def main():
    trace = RetaggableTrace.from_files(TRACE_PATH)
    node_ids = list(range(1, NUM_NODES + 1))
    trace_ticks = trace.sample(node_ids=node_ids, num_jurisdictions=NUM_JURISDICTIONS,
                                jurisdiction_strategy="realistic", seed=0)

    trainer = MAPPOTrainer(env=None, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    trainer.actor.load_state_dict(checkpoint["actor_state_dict"])
    trainer.actor.eval()

    env = BlockchainShardingEnv(
        num_jurisdictions=NUM_JURISDICTIONS, nodes_per_jurisdiction=NODES_PER_JURISDICTION,
        max_ticks=len(trace_ticks), max_arrivals_per_node=MAX_ARRIVALS_PER_NODE,
        node_kwargs={"mu_max": 25.0, "p": 40}, trace_ticks=trace_ticks, seed=1,
    )
    env.reset()

    choice_counts: dict[int, int] = defaultdict(int)

    with torch.no_grad():
        for _tick in range(len(trace_ticks)):
            actions_dict = {}
            for agent_id, arrivals in env._pending_by_agent.items():
                node_id = env._node_id_from_agent(agent_id)
                local_state = torch.tensor(build_local_state(env.simulator._node_by_id(node_id), NUM_JURISDICTIONS), dtype=torch.float32)
                chosen = []
                for tx in arrivals:
                    x = torch.cat([local_state, torch.tensor(build_tx_features(tx, NUM_JURISDICTIONS), dtype=torch.float32)])
                    action = int(torch.argmax(trainer.actor.forward(x)).item()) + 1
                    choice_counts[action] += 1
                    chosen.append(action)
                actions_dict[agent_id] = chosen

            _obs, _rewards, dones, _infos = env.step(actions_dict)
            if dones["__all__"]:
                break

    print(f"{'node_id':>8} | {'jurisdiction':>12} | {'times chosen':>13} | {'pct_of_total':>12}")
    print("-" * 52)
    total = sum(choice_counts.values())
    for node in env.simulator.nodes:
        count = choice_counts.get(node.id, 0)
        pct = (count / total * 100) if total else 0.0
        print(f"{node.id:>8} | {int(node.jurisdiction.decode()):>12} | {count:>13} | {pct:>11.1f}%")

    print()
    print(f"Uniform (fully load-reactive) would be ~{100/NODES_PER_JURISDICTION:.1f}% per node within a jurisdiction.")
    print("Heavy skew toward one node id per jurisdiction indicates a persistent, largely")
    print("state-independent preference rather than reactive load balancing.")


if __name__ == "__main__":
    main()
