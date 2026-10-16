"""MAPPO trainer for BlockchainShardingEnv.

GAE advantages plus PPO's clipped objective (epsilon <= 0.2, Section 3.2). One shared actor and one centralised critic serve all agents.

Only Assign Write Target is trained. Migration and Throttle aren't implemented (see networks.py).
"""

import math

import torch
import torch.nn as nn
import numpy as np

from .env import BlockchainShardingEnv
from .networks import Actor, CentralizedCritic
from .observations import build_local_state, build_tx_features


class RunningStats:
    """Running mean/variance (Welford's algorithm), used to rescale rewards before GAE."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return (self.M2 / self.n) ** 0.5


class RolloutBuffer:
    """Everything one rollout records: a row per routing decision, plus a joint state and reward per tick."""

    def __init__(self):
        self.actor_states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.tick_index: list[int] = []
        self.joint_states: list[torch.Tensor] = []
        self.rewards: list[float] = []          # rescaled; this is what GAE and the critic see
        self.raw_rewards: list[float] = []       # original values, for logging
        self.dones: list[bool] = []
        self.tick_compliance_stats: list[dict] = []
        self.decision_compliant: list[float] = []
        self.decision_rejected: list[float] = []       # 1.0 if the write was turned away (queue full)
        self.decision_least_loaded: list[float] = []   # 1.0 if the chosen node had the shortest queue of the compliant options
        self.decision_had_choice: list[float] = []     # 1.0 if there was more than one compliant option (else 'least loaded' is meaningless)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.tick_index)


def build_joint_state(env: BlockchainShardingEnv, num_jurisdictions: int) -> torch.Tensor:
    flat = []
    for node in env.simulator.nodes:
        flat.extend(build_local_state(node, num_jurisdictions))
    return torch.tensor(flat, dtype=torch.float32)


class MAPPOTrainer:
    def __init__(
        self,
        env: BlockchainShardingEnv,
        num_nodes: int,
        num_jurisdictions: int = 3,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        local_signal_weight: float = 5.0,
        rejection_penalty_weight: float = 2.0,
        queue_preference_weight: float = 1.5,
        target_entropy_fraction: float = 0.5,   # entropy floor, as a fraction of the maximum possible (ln num_nodes)
        entropy_adapt_rate: float = 0.02,
        device: str = "cpu",
    ):
        self.env = env
        self.num_nodes = num_nodes
        self.num_jurisdictions = num_jurisdictions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.local_signal_weight = local_signal_weight
        self.rejection_penalty_weight = rejection_penalty_weight
        self.queue_preference_weight = queue_preference_weight
        self.device = torch.device(device)

        # Entropy floor. The working theory (not yet confirmed): random routing spreads
        # load well because each choice is independent, and a confident, low-entropy
        # policy loses that. If entropy drops below target, entropy_coef gets nudged up
        # so the policy keeps some of that randomness.
        self.target_entropy = target_entropy_fraction * math.log(num_nodes)
        self.entropy_adapt_rate = entropy_adapt_rate
        self.base_entropy_coef = entropy_coef
        self.entropy_coef = entropy_coef  # current value; update() adjusts it each call

        local_state_dim = 3 + num_jurisdictions
        tx_feature_dim = 1 + 2 * num_jurisdictions
        actor_input_dim = local_state_dim + tx_feature_dim

        self.actor = Actor(input_dim=actor_input_dim, hidden_dim=64, num_actions=num_nodes).to(self.device)
        self.critic = CentralizedCritic(num_agents=num_nodes, local_state_dim=local_state_dim, hidden_dim=128).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = RolloutBuffer()
        self.reward_stats = RunningStats()

    def _node_local_state_tensor(self, node_id: int) -> torch.Tensor:
        node = self.env.simulator._node_by_id(node_id)
        return torch.tensor(build_local_state(node, self.num_jurisdictions), dtype=torch.float32)

    def collect_rollout(self, num_ticks: int):
        """Play num_ticks ticks (or until the episode ends) with the current actor and record everything the update needs. Call env.reset() first."""
        for tick_num in range(num_ticks):
            joint_state = build_joint_state(self.env, self.num_jurisdictions)

            actions_dict: dict[str, list[int]] = {}
            for agent_id, arrivals in self.env._pending_by_agent.items():
                node_id = self.env._node_id_from_agent(agent_id)
                local_state = self._node_local_state_tensor(node_id)

                agent_actions = []
                for tx in arrivals:
                    tx_features = torch.tensor(
                        build_tx_features(tx, self.num_jurisdictions),
                        dtype=torch.float32,
                    )
                    x = torch.cat([local_state, tx_features])
                    action, log_prob, _entropy = self.actor.act(x)

                    chosen_node_id = int(action.item()) + 1  # env node ids are 1-indexed
                    chosen_node = self.env.simulator._node_by_id(chosen_node_id)
                    compliant = 1.0 if self.env.simulator.is_compliant_placement(tx, chosen_node) else 0.0

                    # The queue-preference label uses the simulator's true state at the moment of
                    # choice (the CTDE training-time privilege, same as decision_compliant). The actor
                    # itself still never sees other nodes' queues; this only shapes the training signal.
                    compliant_candidates = [n for n in self.env.simulator.nodes if self.env.simulator.is_compliant_placement(tx, n)]
                    had_choice = 1.0 if len(compliant_candidates) > 1 else 0.0
                    if compliant_candidates:
                        least_loaded_id = min(compliant_candidates, key=lambda n: len(n.write_queue)).id
                        least_loaded = 1.0 if chosen_node_id == least_loaded_id else 0.0
                    else:
                        least_loaded = 0.0

                    self.buffer.actor_states.append(x)
                    self.buffer.actions.append(action)
                    self.buffer.log_probs.append(log_prob.detach())
                    self.buffer.tick_index.append(tick_num)
                    self.buffer.decision_compliant.append(compliant)
                    self.buffer.decision_least_loaded.append(least_loaded)
                    self.buffer.decision_had_choice.append(had_choice)

                    agent_actions.append(chosen_node_id)

                actions_dict[agent_id] = agent_actions

            decisions_this_tick = sum(len(v) for v in actions_dict.values())

            _next_obs, rewards, dones, _infos = self.env.step(actions_dict)

            # Placement results come back in the same agent order the buffer rows were built
            # in (nothing reorders the dict in between), so they line up one-to-one.
            placement_results = self.env.last_placement_results
            assert len(placement_results) == decisions_this_tick, (
                f"placement_results length {len(placement_results)} != decisions made {decisions_this_tick} "
                f"— ordering assumption broken, rejection signal would misalign."
            )
            self.buffer.decision_rejected.extend(0.0 if accepted else 1.0 for accepted in placement_results)

            shared_reward = next(iter(rewards.values()))
            done = dones["__all__"]

            self.reward_stats.update(shared_reward)
            scaled_reward = shared_reward / (self.reward_stats.std + 1e-8)
            scaled_reward = float(np.clip(scaled_reward, -10.0, 10.0))

            self.buffer.joint_states.append(joint_state)
            self.buffer.rewards.append(scaled_reward)
            self.buffer.raw_rewards.append(shared_reward)
            self.buffer.dones.append(done)
            self.buffer.tick_compliance_stats.append(dict(self.env.simulator.last_tick_stats))

            if done:
                break

    def _compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE over the per-tick joint rewards and critic values. Returns (advantages, returns)."""
        with torch.no_grad():
            joint_states = torch.stack(self.buffer.joint_states)
            values = self.critic(joint_states)

        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32)

        T = len(rewards)
        advantages = torch.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < T else 0.0
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> dict:
        """One PPO update from the buffer, repeated for ppo_epochs, then clear the buffer.

        Credit assignment is the tricky part. Rt is a single shared number per tick, but a tick can contain 100+ separate routing decisions, so handing every one of them the same advantage tells the policy nothing about which decision caused what. Three per-decision terms are added to the tick-level advantage to fix that:

          1. Compliance (+1 / -1). This is the fix that made learning work at all; without it the violation rate sat at the random baseline.
          2. Rejection penalty. A write bounced for a full queue used to show up only as one missing TPS count, smeared across the whole tick. Now the decision that caused it is penalised directly.
          3. Queue preference (+1 / -1), only where there was more than one compliant option: rewards picking the compliant node with the shortest queue.

        All three come from the simulator's true state at decision time (the CTDE training-time privilege), so the actor's own observation doesn't change. That also caps what term 3 can teach: the actor never sees other nodes' queues, so it can learn an average-case preference, not live load-balancing. That's a deliberate boundary of the Dec-POMDP setup, not an oversight.

        The entropy coefficient is adapted once per call: raised when policy entropy has fallen below the floor, relaxed back toward its base value when above. A simple proportional controller, not SAC-style learned temperature.
        """
        # adapt entropy_coef once per update (see docstring), before any gradient step
        with torch.no_grad():
            actor_states_snapshot = torch.stack(self.buffer.actor_states)
            current_entropy = self.actor.distribution(actor_states_snapshot).entropy().mean().item()
        entropy_gap = self.target_entropy - current_entropy
        self.entropy_coef = max(0.0, self.base_entropy_coef + self.entropy_adapt_rate * entropy_gap)

        advantages, returns = self._compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        tick_index = torch.tensor(self.buffer.tick_index)
        tick_level_advantages = advantages[tick_index]

        # per-decision terms (see docstring); kept at +/-1 so they sit on the same scale
        # as the already-normalised tick-level advantage
        decision_compliant = torch.tensor(self.buffer.decision_compliant, dtype=torch.float32)
        decision_rejected = torch.tensor(self.buffer.decision_rejected, dtype=torch.float32)
        decision_least_loaded = torch.tensor(self.buffer.decision_least_loaded, dtype=torch.float32)
        decision_had_choice = torch.tensor(self.buffer.decision_had_choice, dtype=torch.float32)

        compliance_signal = decision_compliant * 2 - 1
        rejection_signal = -decision_rejected
        queue_signal = (decision_least_loaded * 2 - 1) * decision_had_choice

        per_decision_advantages = (
            tick_level_advantages
            + self.local_signal_weight * compliance_signal
            + self.rejection_penalty_weight * rejection_signal
            + self.queue_preference_weight * queue_signal
        )

        old_log_probs = torch.stack(self.buffer.log_probs)
        actions = torch.stack(self.buffer.actions)
        actor_states = torch.stack(self.buffer.actor_states)
        joint_states = torch.stack(self.buffer.joint_states)

        last_actor_loss, last_critic_loss, last_entropy = 0.0, 0.0, 0.0

        for _ in range(self.ppo_epochs):
            new_log_probs, entropy = self.actor.evaluate(actor_states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * per_decision_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * per_decision_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            values = self.critic(joint_states)
            critic_loss = nn.functional.mse_loss(values, returns) * self.value_coef

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_entropy = entropy.mean().item()

        total_compliant = sum(s.get("compliant", 0) for s in self.buffer.tick_compliance_stats)
        total_violation = sum(s.get("violation", 0) for s in self.buffer.tick_compliance_stats)
        total_rejected = sum(s.get("rejected", 0) for s in self.buffer.tick_compliance_stats)
        total_writes = total_compliant + total_violation
        violation_rate = (total_violation / total_writes) if total_writes else 0.0

        decisions_with_choice = decision_had_choice.sum().item()
        queue_preference_rate = (
            (decision_least_loaded * decision_had_choice).sum().item() / decisions_with_choice
            if decisions_with_choice else 0.0
        )

        diagnostics = {
            "actor_loss": last_actor_loss,
            "critic_loss": last_critic_loss,
            "entropy": last_entropy,
            "entropy_coef": self.entropy_coef,
            "mean_reward": float(np.mean(self.buffer.raw_rewards)),
            "reward_scale_std": self.reward_stats.std,
            "num_ticks": len(self.buffer),
            "violation_rate": violation_rate,
            "total_writes": total_writes,
            "total_rejected": total_rejected,
            "queue_preference_rate": queue_preference_rate,
        }
        self.buffer.clear()
        return diagnostics
