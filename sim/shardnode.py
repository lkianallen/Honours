import math
from collections import deque
from .transaction import Transaction


class ShardNode:
    """One validator cluster (a shard) and its per-tick update rules from Section 3.3.3.

    Utilisation is kept as a 0-1 fraction rather than raw capacity, because Eq. 2 and Eq. 4 only make sense for Ucpu in [0, 1].
    """

    id: int
    jurisdiction: bytes

    write_queue: deque[Transaction]
    read_queue: deque[Transaction]

    Ucpu: float   # CPU load as a 0-1 fraction (Eq. 4)
    Udisk: float  # disk fill as a 0-1 fraction; only ever grows (Eq. 3)

    Cap_cpu: float   # scale that turns queue/arrival counts into a CPU fraction
    Cap_disk: float  # disk capacity, same units as payload_size

    mu_max: float     # most writes this node can process per tick when idle (Eq. 2)
    beta: float       # how fast capacity drops as CPU load rises (Eq. 2)
    alpha_cpu: float  # CPU average smoothing; higher reacts more slowly (Eq. 4)

    p: int  # queue length at which the node stops accepting writes (Section 3.3.3(1))

    _pending_arrivals: int  # writes accepted so far this tick (lambda_i(t))

    def __init__(
        self,
        id: int,
        jurisdiction: bytes,
        Cap_cpu: float = 100.0,
        Cap_disk: float = 10_000.0,
        mu_max: float = 10.0,
        beta: float = 1.0,
        alpha_cpu: float = 0.9,
        p: int = 20,
    ):
        self.id = id
        self.jurisdiction = jurisdiction
        self.write_queue = deque()
        self.read_queue = deque()
        self.Ucpu = 0.0
        self.Udisk = 0.0
        self.Cap_cpu = Cap_cpu
        self.Cap_disk = Cap_disk
        self.mu_max = mu_max
        self.beta = beta
        self.alpha_cpu = alpha_cpu
        self.p = p
        self._pending_arrivals = 0

    def is_saturated(self) -> bool:
        """True once the write queue has hit p, after which new writes get turned away (Section 3.3.3(1))."""
        return len(self.write_queue) >= self.p

    def receive_write(self, transaction: Transaction) -> bool:
        """Try to queue a write. Returns False (and drops it) if the node is saturated."""
        if self.is_saturated():
            return False
        self.write_queue.append(transaction)
        self._pending_arrivals += 1
        return True

    def receive_read(self, transaction: Transaction) -> None:
        self.read_queue.append(transaction)

    def _compute_mu(self, Qlen_t: int, lambda_t: int) -> int:
        """How many writes this node can get through this tick (Eq. 2). Capped by what's actually queued, and shrinks as CPU load rises.

            mu = min(Qlen + lambda, floor(mu_max * (1 - exp(-beta * (1 - Ucpu)))))
        """
        capacity_bound = math.floor(
            self.mu_max * (1 - math.exp(-self.beta * (1 - self.Ucpu)))
        )
        return int(min(Qlen_t + lambda_t, capacity_bound))

    def process(self, tick: int) -> dict:
        """Run one tick: work out capacity (Eq. 2), pop that many writes off the queue, then update disk (Eq. 3) and CPU (Eq. 4)."""
        Qlen_t = len(self.write_queue)
        lambda_t = self._pending_arrivals

        mu_t = self._compute_mu(Qlen_t, lambda_t)
        num_to_process = min(mu_t, Qlen_t)

        processed: list[Transaction] = []
        for _ in range(num_to_process):
            processed.append(self.write_queue.popleft())

        if processed:
            written_bytes = sum(tx.payload_size for tx in processed)
            self.Udisk = min(1.0, self.Udisk + written_bytes / self.Cap_disk)

        self.Ucpu = self.alpha_cpu * self.Ucpu + (1 - self.alpha_cpu) * (
            (lambda_t + Qlen_t) / self.Cap_cpu
        )
        self.Ucpu = max(0.0, min(1.0, self.Ucpu))

        self._pending_arrivals = 0

        return {
            "tick": tick,
            "node_id": self.id,
            "processed": processed,
            "tps": len(processed),
            "mu_t": mu_t,
            "Qlen_t": Qlen_t,
            "Ucpu": self.Ucpu,
            "Udisk": self.Udisk,
        }
