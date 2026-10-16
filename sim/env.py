import random
import numpy as np
from .simulator import Simulator
from .shardnode import ShardNode
from .transaction import Transaction
from .observations import build_local_state, build_tx_features


class BlockchainShardingEnv:
    """Gym-style environment around the Simulator, with one agent per shard node.

    Each agent only routes the transactions that arrived at its own node, so execution is genuinely decentralised (Dec-POMDP) rather than one dispatcher making every call.

    A transaction's origin node is picked at random and has nothing to do with its jurisdiction. That's deliberate: it models a client submitting to whichever node is handy, which isn't necessarily a compliant one, and that's the situation the thesis is about.

    Observation encoding lives in observations.py and is shared with mappo.py. An agent's observation is its local state followed by K ingress slots (one per arriving transaction), zero-padded when fewer than K arrive. A zero payload means padding, since real payloads are never zero.

    step() takes {agent_id: [target node ids]}, each list trimmed to that agent's real arrival count, e.g. {"agent_1": [2, 1, 3], "agent_3": []}.
    """

    def __init__(
        self,
        num_jurisdictions: int = 3,
        nodes_per_jurisdiction: int | list[int] | dict[int, int] = 3,
        max_ticks: int = 50,
        transactions_per_tick: int = 5,
        max_arrivals_per_node: int = 5,  # K
        node_kwargs: dict | None = None,
        simulator_kwargs: dict | None = None,
        seed: int | None = None,
        trace_ticks: list[list[Transaction]] | None = None,
        trace_sampler_fn=None,  # Callable[[], list[list[Transaction]]] | None
    ):
        self.num_jurisdictions = num_jurisdictions
        self.nodes_per_jurisdiction = nodes_per_jurisdiction
        self.max_ticks = max_ticks
        self.transactions_per_tick = transactions_per_tick
        self.K = max_arrivals_per_node
        self.node_kwargs = node_kwargs or {}
        self.simulator_kwargs = simulator_kwargs or {}
        self._rng = random.Random(seed)

        # trace_ticks: a ready-made list of per-tick transaction batches; replaces the synthetic generator.
        # trace_sampler_fn: called on every reset() to build a freshly tagged trace from the same
        # underlying data (see RetaggableTrace), so the policy can't memorise one fixed jurisdiction
        # labelling. Wins over trace_ticks if both are given.
        self._trace_ticks = trace_ticks
        self._trace_sampler_fn = trace_sampler_fn
        self._trace_cursor = 0

        self.simulator: Simulator | None = None
        self._pending_by_agent: dict[str, list[Transaction]] = {}
        self.last_placement_results: list[bool] = []

    @staticmethod
    def _agent_id(node_id: int) -> str:
        return f"agent_{node_id}"

    @staticmethod
    def _node_id_from_agent(agent_id: str) -> int:
        return int(agent_id.split("_")[1])

    def _resolve_shard_counts(self) -> list[int]:
        """Turn nodes_per_jurisdiction (an int, list or dict) into one node count per jurisdiction."""
        npj = self.nodes_per_jurisdiction

        if isinstance(npj, int):
            return [npj] * self.num_jurisdictions

        if isinstance(npj, dict):
            missing = [j for j in range(1, self.num_jurisdictions + 1) if j not in npj]
            if missing:
                raise ValueError(f"nodes_per_jurisdiction dict is missing jurisdiction id(s): {missing}")
            return [npj[j] for j in range(1, self.num_jurisdictions + 1)]

        if isinstance(npj, (list, tuple)):
            if len(npj) != self.num_jurisdictions:
                raise ValueError(f"nodes_per_jurisdiction has {len(npj)} entries but num_jurisdictions={self.num_jurisdictions}.")
            return list(npj)

        raise TypeError(f"nodes_per_jurisdiction must be int, list[int], or dict[int, int], got {type(npj)}")

    def _build_nodes(self) -> list[ShardNode]:
        shard_counts = self._resolve_shard_counts()
        nodes = []
        node_id = 1
        for jurisdiction_id, count in enumerate(shard_counts, start=1):
            for _ in range(count):
                nodes.append(ShardNode(id=node_id, jurisdiction=bytes(str(jurisdiction_id), "utf-8"), **self.node_kwargs))
                node_id += 1
        return nodes

    def _generate_arrivals(self) -> dict[str, list[Transaction]]:
        if self._trace_ticks is not None:
            return self._next_trace_tick()
        return self._generate_synthetic_arrivals()

    def _next_trace_tick(self) -> dict[str, list[Transaction]]:
        by_agent: dict[str, list[Transaction]] = {self._agent_id(node.id): [] for node in self.simulator.nodes}
        valid_node_ids = {node.id for node in self.simulator.nodes}

        if self._trace_cursor < len(self._trace_ticks):
            for tx in self._trace_ticks[self._trace_cursor]:
                if tx.origin_node_id not in valid_node_ids:
                    raise ValueError(f"Trace transaction {tx.id} has origin_node_id={tx.origin_node_id}, no matching node.")
                agent_id = self._agent_id(tx.origin_node_id)
                if len(by_agent[agent_id]) < self.K:
                    by_agent[agent_id].append(tx)
                # anything past K arrivals at one node in a tick is dropped, not carried over
            self._trace_cursor += 1

        return by_agent

    def _generate_synthetic_arrivals(self) -> dict[str, list[Transaction]]:
        """Random placeholder workload, used when no real trace is supplied."""
        by_agent: dict[str, list[Transaction]] = {self._agent_id(node.id): [] for node in self.simulator.nodes}
        node_ids = [node.id for node in self.simulator.nodes]

        for _ in range(self.transactions_per_tick):
            origin_id = self._rng.choice(node_ids)
            agent_id = self._agent_id(origin_id)
            if len(by_agent[agent_id]) >= self.K:
                continue

            payload_size = self._rng.randint(1, 50)
            jurisdiction = bytes(str(self._rng.randint(1, self.num_jurisdictions)), "utf-8")
            by_agent[agent_id].append(Transaction(payload_size, jurisdiction, origin_node_id=origin_id))

        return by_agent

    def _build_obs(self) -> dict[str, np.ndarray]:
        """Each agent's observation: its local state plus K ingress slots, zero-padded. Built with the same functions mappo.py uses."""
        obs = {}
        for node in self.simulator.nodes:
            agent_id = self._agent_id(node.id)
            local_state = build_local_state(node, self.num_jurisdictions)

            tx_feature_dim = 1 + 2 * self.num_jurisdictions
            ingress = []
            arrivals = self._pending_by_agent.get(agent_id, [])
            for tx in arrivals[: self.K]:
                ingress.extend(build_tx_features(tx, self.num_jurisdictions))
            ingress.extend([0.0] * (tx_feature_dim * (self.K - len(arrivals[: self.K]))))

            obs[agent_id] = np.array(local_state + ingress, dtype=np.float32)
        return obs

    def reset(self):
        nodes = self._build_nodes()
        self.simulator = Simulator(nodes, **self.simulator_kwargs)
        self._trace_cursor = 0
        if self._trace_sampler_fn is not None:
            self._trace_ticks = self._trace_sampler_fn()
        self._pending_by_agent = self._generate_arrivals()
        return self._build_obs()

    def step(self, actions: dict[str, list[int]]):
        if self.simulator is None:
            raise RuntimeError("Call reset() before step().")

        all_transactions: list[Transaction] = []
        all_targets: list[int] = []

        for agent_id, arrivals in self._pending_by_agent.items():
            agent_actions = actions.get(agent_id, [])
            if len(agent_actions) != len(arrivals):
                raise ValueError(
                    f"{agent_id}: expected {len(arrivals)} action(s), got {len(agent_actions)}. "
                    f"Trim any K-slot padding before calling step()."
                )
            all_transactions.extend(arrivals)
            all_targets.extend(agent_actions)

        # Results come back in the same agent order collect_rollout used to build its
        # buffer, so the two can be matched up directly.
        if all_transactions:
            self.last_placement_results = self.simulator.apply_routing_actions(all_transactions, all_targets)
        else:
            self.last_placement_results = []

        reward = self.simulator.tick()

        self._pending_by_agent = self._generate_arrivals()
        next_obs = self._build_obs()

        # Done = the batch we just generated for the next step is empty. Checking the
        # cursor position instead looks right but drops the trace's last block every episode.
        trace_exhausted = self._trace_ticks is not None and all(len(v) == 0 for v in self._pending_by_agent.values())
        done = self.simulator.ticks >= self.max_ticks or trace_exhausted
        agent_ids = list(self._pending_by_agent.keys())

        # Rt is one shared scalar, so every agent gets the same reward (normal for cooperative CTDE).
        rewards = {agent_id: reward for agent_id in agent_ids}
        dones = {agent_id: done for agent_id in agent_ids}
        dones["__all__"] = done
        infos = {agent_id: {"tick": self.simulator.ticks} for agent_id in agent_ids}

        return next_obs, rewards, dones, infos
