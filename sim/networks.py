"""Actor and critic networks for MAPPO (Section 3.2, following Yu et al. 2021).

Sizes per the spec: the actor is 2 x 64 ReLU with a softmax; the critic (centralised, training only) is 3 x 128 ReLU with a linear V(s) output. PPO clip epsilon is at most 0.2.

Two choices the spec doesn't spell out:
  1. The actor makes one decision per pending transaction, not one fixed-size output per tick. That fits env.py's variable-length action lists with no padding.
  2. Every agent shares one actor's weights. Each agent still only sees its own observation when it acts, so sharing doesn't centralise anything, and it keeps the parameter count flat as N grows.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Picks a target node for one transaction.

    Input is the agent's local state plus that transaction's features. MAPPOTrainer works out the exact size from the number of jurisdictions; the default here is just a placeholder. Output is a softmax over nodes. Only Assign Write Target is implemented; Trigger Data Migration and Throttle Rate aren't.
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, num_actions: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw action scores (logits), before the softmax."""
        return self.net(x)

    def distribution(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))

    def act(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; returns the action, its log-probability and the entropy."""
        dist = self.distribution(x)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-prob and entropy of actions already taken, under the current policy. PPO's update step needs this."""
        dist = self.distribution(x)
        return dist.log_prob(actions), dist.entropy()


class CentralizedCritic(nn.Module):
    """Estimates V(s) from the joint state: every agent's local state, concatenated.

    Training only; it's never used when agents act (that's the CTD in CTDE). Per-transaction details are left out on purpose, so the input size doesn't change with how many transactions arrive. As with the actor, the default local_state_dim is a placeholder; MAPPOTrainer passes the real value.
    """

    def __init__(self, num_agents: int, local_state_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        input_dim = num_agents * local_state_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, joint_state: torch.Tensor) -> torch.Tensor:
        return self.net(joint_state).squeeze(-1)
