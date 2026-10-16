"""How node and transaction state get encoded into network inputs (Section 3.2).

Both env.py (its observation) and mappo.py (what actually trains the actor) use these functions, so the two can't drift apart. They did once, which is why this lives in its own file.
"""

import math


def one_hot(index_1based: int, num_classes: int) -> list[float]:
    vec = [0.0] * num_classes
    vec[index_1based - 1] = 1.0
    return vec


def multi_hot(indices_1based: list[int], num_classes: int) -> list[float]:
    vec = [0.0] * num_classes
    for idx in indices_1based:
        vec[idx - 1] = 1.0
    return vec


def build_local_state(node, num_jurisdictions: int) -> list[float]:
    """A node's own state: [Ucpu, Udisk, Qlen/p] plus a one-hot jurisdiction. Length 3 + num_jurisdictions."""
    jurisdiction_id = int(node.jurisdiction.decode())
    return [
        node.Ucpu,
        node.Udisk,
        len(node.write_queue) / node.p,
    ] + one_hot(jurisdiction_id, num_jurisdictions)


def build_tx_features(tx, num_jurisdictions: int) -> list[float]:
    """A transaction's features: log1p(payload), one-hot jurisdiction, and the Mmeta multi-hot. Length 1 + 2 * num_jurisdictions. (log1p because payloads range from a few bytes to thousands.)"""
    jurisdiction_id = int(tx.jurisdiction.decode())
    return (
        [math.log1p(tx.payload_size)]
        + one_hot(jurisdiction_id, num_jurisdictions)
        + multi_hot(tx.allowed_jurisdictions, num_jurisdictions)
    )
