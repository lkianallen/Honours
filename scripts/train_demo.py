"""Quick end-to-end check on synthetic arrivals. Not a result; it just proves the training loop runs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.env import BlockchainShardingEnv
from sim.mappo import MAPPOTrainer

NUM_JURISDICTIONS = 3
NODES_PER_JURISDICTION = 3
NUM_NODES = NUM_JURISDICTIONS * NODES_PER_JURISDICTION

env = BlockchainShardingEnv(
    num_jurisdictions=NUM_JURISDICTIONS,
    nodes_per_jurisdiction=NODES_PER_JURISDICTION,
    max_ticks=20,
    transactions_per_tick=10,
    max_arrivals_per_node=5,
    seed=1,
)

trainer = MAPPOTrainer(env, num_nodes=NUM_NODES, num_jurisdictions=NUM_JURISDICTIONS, ppo_epochs=4)

NUM_ROLLOUTS = 15
for rollout_num in range(NUM_ROLLOUTS):
    env.reset()
    trainer.collect_rollout(num_ticks=env.max_ticks)
    diagnostics = trainer.update()
    print(
        f"rollout {rollout_num+1:2d} | "
        f"mean_reward={diagnostics['mean_reward']:9.2f} | "
        f"actor_loss={diagnostics['actor_loss']:8.4f} | "
        f"critic_loss={diagnostics['critic_loss']:10.2f} | "
        f"entropy={diagnostics['entropy']:.4f}"
    )
