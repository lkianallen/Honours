import random
from .transaction import Transaction
from .shardnode import ShardNode


class Simulator:
    """Runs the tick loop over all shard nodes: places writes, processes queues, resolves reads, and scores each tick with the reward from Section 3.2.4."""

    nodes: list[ShardNode]
    ticks: int

    gamma_network: float     # Eq. 5: how much a backed-up destination adds to read latency
    delta_crossShard: float  # Eq. 5: fixed cost of any cross-shard read, in ms

    w1: float
    w2: float
    lambda_violation: float

    block_registry: dict[int, int]  # which node each block was written to; reads use this to find their target

    def __init__(
        self,
        nodes: list[ShardNode],
        gamma_network: float = 5.0,
        delta_crossShard: float = 10.0,
        w1: float = 0.5,
        w2: float = 0.5,
        lambda_violation: float = -1000.0,
        verbose: bool = False,
    ):
        self.nodes = nodes
        self.ticks = 0
        self.gamma_network = gamma_network
        self.delta_crossShard = delta_crossShard
        self.w1 = w1
        self.w2 = w2
        self.lambda_violation = lambda_violation
        self.block_registry = {}
        self.verbose = verbose

        # Per-tick counters. Placement happens before tick() runs, so these get filled
        # in first and are reset at the END of tick(). Resetting at the start would
        # wipe them before anything reads them.
        self._tick_writes_compliant = 0
        self._tick_writes_violation = 0
        self._tick_writes_rejected = 0
        self._tick_reads_cross_shard = 0
        self._header_printed = False
        self.last_tick_stats: dict = {}  # snapshot of the finished tick; read this from outside, not the live counters

        # d_geo(A, B): base delay between jurisdictions. Chapter 3 gives no numbers for
        # it, so these are placeholders (0ms within a jurisdiction, 80ms across) until
        # real inter-region RTT figures are plugged in.
        self._jurisdictions = sorted({node.jurisdiction for node in nodes})
        self._d_geo = {
            (a, b): 0.0 if a == b else 80.0
            for a in self._jurisdictions
            for b in self._jurisdictions
        }

    def _node_by_id(self, node_id: int) -> ShardNode:
        return self.nodes[node_id - 1]

    def route(self, transaction: Transaction) -> int:
        """Baseline router, not the agent's policy: a random node that's in the right jurisdiction and not full."""
        eligible = [
            node for node in self.nodes
            if node.jurisdiction == transaction.jurisdiction and not node.is_saturated()
        ]
        if eligible:
            return random.choice(eligible).id
        raise ValueError("No shard node found for the given transaction jurisdiction")

    def assign_transactions(self, transactions: list[Transaction]) -> None:
        """Place a batch using route() above."""
        for transaction in transactions:
            self._place_transaction(transaction, self.route(transaction))

    def apply_routing_action(self, transaction: Transaction, node_id: int) -> bool:
        """Where the agent's routing decision (Assign Write Target) lands. There's no jurisdiction check here on purpose, so the agent can get it wrong and be penalised for it. Only a full queue can turn a write away."""
        return self._place_transaction(transaction, node_id)

    def apply_routing_actions(self, transactions: list[Transaction], node_ids: list[int]) -> list[bool]:
        """Same, for a batch. Returns accepted/rejected for each transaction, in input order, so training can penalise the specific decisions that got bounced."""
        if len(transactions) != len(node_ids):
            raise ValueError("transactions and node_ids must be the same length")
        return [self.apply_routing_action(transaction, node_id) for transaction, node_id in zip(transactions, node_ids)]

    def is_compliant_placement(self, transaction: Transaction, node: ShardNode) -> bool:
        """Is this node an allowed home for this transaction? Checks Mmeta (allowed_jurisdictions). The reward, the logging and the training signal all go through this one function, so they can't disagree."""
        node_jurisdiction_id = int(node.jurisdiction.decode())
        return node_jurisdiction_id in transaction.allowed_jurisdictions

    def _place_transaction(self, transaction: Transaction, node_id: int) -> bool:
        node = self._node_by_id(node_id)

        if transaction.is_read:
            node.receive_read(transaction)
            if self.verbose:
                print(f"payload size {transaction.payload_size} jurisdiction {transaction.jurisdiction} read routed to node {node.id}")
            return True

        accepted = node.receive_write(transaction)
        if not accepted:
            self._tick_writes_rejected += 1
            if self.verbose:
                print(f"Node {node.id}: write REJECTED, queue saturated (p={node.p})")
            return False

        self.block_registry[transaction.id] = node.id
        compliant = self.is_compliant_placement(transaction, node)
        if compliant:
            self._tick_writes_compliant += 1
        else:
            self._tick_writes_violation += 1

        if self.verbose:
            print(
                f"payload size {transaction.payload_size} jurisdiction {transaction.jurisdiction} "
                f"write routed to node {node.id}{'' if compliant else '  <-- JURISDICTION VIOLATION'}"
            )
        return True

    def _cross_shard_latency(self, origin_node: ShardNode, dest_node: ShardNode, dest_mu_t: float) -> float:
        """Latency of a cross-shard read (Eq. 5): base distance, plus how backed-up the destination is, plus a fixed overhead.

            L(A->B) = d_geo(A,B) + gamma_network * (Qlen_B / mu_B) + delta_crossShard
        """
        d_geo = self._d_geo[(origin_node.jurisdiction, dest_node.jurisdiction)]
        Qlen_B = len(dest_node.write_queue)
        congestion_term = self.gamma_network * (Qlen_B / dest_mu_t) if dest_mu_t > 0 else 0.0
        return d_geo + congestion_term + self.delta_crossShard

    def _process_reads(self, tick_stats: dict[int, dict]) -> dict[int, list[float]]:
        """Try to resolve every waiting read. Same-shard reads are free, cross-shard reads pay Eq. 5, and reads whose block hasn't been written yet just keep waiting."""
        latencies_by_node: dict[int, list[float]] = {node.id: [] for node in self.nodes}

        for node in self.nodes:
            still_pending = []
            for tx in node.read_queue:
                home_node_id = self.block_registry.get(tx.ref_block_id)
                if home_node_id is None:
                    still_pending.append(tx)  # target block isn't written yet; try again next tick
                    continue
                if home_node_id == node.id:
                    continue  # same shard, so no latency cost

                home_node = self._node_by_id(home_node_id)
                dest_mu_t = tick_stats.get(home_node.id, {}).get("mu_t", 0)
                latency = self._cross_shard_latency(node, home_node, dest_mu_t)
                latencies_by_node[node.id].append(latency)
                self._tick_reads_cross_shard += 1
                if self.verbose:
                    print(f"Node {node.id}: cross-shard read of block {tx.ref_block_id} -> node {home_node.id}, latency {latency:.2f}ms")

            node.read_queue = type(node.read_queue)(still_pending)

        return latencies_by_node

    def _compute_jurisdiction_penalty(self, node: ShardNode, tx: Transaction) -> float:
        """Penalty for one placement: 0 if the node is allowed, lambda_violation if not (Section 3.2.4)."""
        return 0.0 if self.is_compliant_placement(tx, node) else self.lambda_violation

    def compute_reward(self, tick_stats: dict[int, dict], latencies_by_node: dict[int, list[float]]) -> float:
        """Reward for one tick (Section 3.2.4): throughput minus latency, minus a hit for every violation.

            Rt = sum_i(w1*TPS_i - w2*Latency_i) - sum_i C_jurisdiction
        """
        total = 0.0
        for node in self.nodes:
            stats = tick_stats[node.id]
            tps_i = stats["tps"]
            node_latencies = latencies_by_node.get(node.id, [])
            latency_i = sum(node_latencies) / len(node_latencies) if node_latencies else 0.0

            total += self.w1 * tps_i - self.w2 * latency_i
            for tx in stats["processed"]:
                total += self._compute_jurisdiction_penalty(node, tx)

        return total

    def tick(self) -> float:
        if self.verbose:
            print(f"             Tick {self.ticks + 1}:")
            print("-----------------------------------")

        tick_stats: dict[int, dict] = {}
        for node in self.nodes:
            tick_stats[node.id] = node.process(self.ticks)

        latencies_by_node = self._process_reads(tick_stats)
        reward = self.compute_reward(tick_stats, latencies_by_node)

        if self.verbose:
            print(f"Tick reward Rt = {reward:.2f}")
        else:
            self._print_tick_row(reward)

        # save the numbers before they're reset below
        self.last_tick_stats = {
            "compliant": self._tick_writes_compliant,
            "violation": self._tick_writes_violation,
            "rejected": self._tick_writes_rejected,
            "cross_shard_reads": self._tick_reads_cross_shard,
            "reward": reward,
        }

        self._tick_writes_compliant = 0
        self._tick_writes_violation = 0
        self._tick_writes_rejected = 0
        self._tick_reads_cross_shard = 0

        self.ticks += 1
        return reward

    def _print_tick_row(self, reward: float) -> None:
        """One summary line per tick. Heads-up: the counts are taken when writes are placed, but the reward's violation penalty is applied when they're processed, so under load the two can differ."""
        if not self._header_printed:
            print(f"{'tick':>5} | {'compliant':>9} | {'violation':>9} | {'rejected':>8} | {'viol%':>6} | {'x-shard rd':>10} | {'reward':>12}")
            print("-" * 76)
            self._header_printed = True

        total_writes = self._tick_writes_compliant + self._tick_writes_violation
        viol_pct = (self._tick_writes_violation / total_writes * 100) if total_writes else 0.0

        print(
            f"{self.ticks + 1:>5} | {self._tick_writes_compliant:>9} | {self._tick_writes_violation:>9} | "
            f"{self._tick_writes_rejected:>8} | {viol_pct:>5.1f}% | {self._tick_reads_cross_shard:>10} | {reward:>12.2f}"
        )

    def print_results(self) -> None:
        print(f"Simulation completed in {self.ticks} ticks.")
        for node in self.nodes:
            print(f"Node {node.id}: Ucpu {node.Ucpu:.3f}, Udisk {node.Udisk:.3f}, Qlen {len(node.write_queue)}")
