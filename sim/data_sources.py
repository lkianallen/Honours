"""Loads real Sepolia transactions and turns them into simulator input (Section 3.4).

Pull blocks with fetch_sepolia_trace.py, which saves them as JSON, then load them here with load_transactions_from_alchemy(). This module only reads files; it never touches the network.
"""

import json
import random
from pathlib import Path

from .transaction import Transaction


def _hex_to_int(value: str) -> int:
    return int(value, 16)


def _input_length_bytes(input_hex: str) -> int:
    """Size of a transaction's input data in bytes, used as its payload size."""
    if not input_hex or input_hex in ("0x", "0X"):
        return 0
    return (len(input_hex) - 2) // 2  # strip '0x', 2 hex chars per byte


def load_alchemy_blocks(paths) -> list[dict]:
    """Read saved eth_getBlockByNumber results (one file, a JSON array, or a list of files) and sort them by timestamp. Accepts str or Path."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    blocks: list[dict] = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            blocks.extend(data)
        else:
            blocks.append(data)

    blocks.sort(key=lambda b: _hex_to_int(b["timestamp"]))
    return blocks


def assign_jurisdiction(strategy: str, num_jurisdictions: int, rng: random.Random) -> bytes:
    """Pick a jurisdiction tag for a transaction (Section 3.4). "realistic" is roughly 60/30/10 (with geometric falloff when there aren't 3 jurisdictions); "equal" is uniform."""
    if strategy == "equal":
        j = rng.randint(1, num_jurisdictions)
        return bytes(str(j), "utf-8")

    if strategy == "realistic":
        if num_jurisdictions == 3:
            weights = [0.6, 0.3, 0.1]
        else:
            raw = [0.5 ** i for i in range(num_jurisdictions)]
            total = sum(raw)
            weights = [w / total for w in raw]
        j = rng.choices(range(1, num_jurisdictions + 1), weights=weights, k=1)[0]
        return bytes(str(j), "utf-8")

    raise ValueError(f"Unknown jurisdiction strategy: {strategy!r} (expected 'realistic' or 'equal')")


class RetaggableTrace:
    """A real trace whose jurisdiction and origin tags can be re-rolled every episode.

    Real chain data has no legal jurisdiction, so those tags are synthetic. If they were fixed once, the policy could just memorise that one labelling. Keeping the real parts (payload, gas, hash, block structure) and re-sampling only the tags gives it fresh variety from a single data pull.
    """

    def __init__(self, blocks: list[dict]):
        self.blocks = blocks

    def sample(
        self,
        node_ids: list[int],
        num_jurisdictions: int = 3,
        jurisdiction_strategy: str = "realistic",
        seed: int | None = None,
    ) -> list[list[Transaction]]:
        """A freshly tagged trace_ticks, ready to hand to BlockchainShardingEnv."""
        transactions = build_transactions_from_blocks(self.blocks, node_ids, num_jurisdictions, jurisdiction_strategy, seed)
        return group_transactions_into_ticks(transactions, self.blocks)

    @classmethod
    def from_files(cls, paths: str | list[str]) -> "RetaggableTrace":
        return cls(load_alchemy_blocks(paths))


def build_transactions_from_blocks(
    blocks: list[dict],
    node_ids: list[int],
    num_jurisdictions: int = 3,
    jurisdiction_strategy: str = "realistic",
    seed: int | None = None,
) -> list[Transaction]:
    """Flatten blocks into Transactions in timestamp order. Jurisdiction tags are synthetic (see RetaggableTrace) and the origin node is picked at random."""
    rng = random.Random(seed)
    transactions = []

    for block in blocks:
        for tx in block.get("transactions", []):
            payload_size = _input_length_bytes(tx.get("input", "0x"))
            gas = _hex_to_int(tx["gas"]) if "gas" in tx else None
            jurisdiction = assign_jurisdiction(jurisdiction_strategy, num_jurisdictions, rng)
            origin_node_id = rng.choice(node_ids)

            transactions.append(
                Transaction(
                    payload_size=max(payload_size, 1),
                    jurisdiction=jurisdiction,
                    origin_node_id=origin_node_id,
                    gas=gas,
                    source_tx_hash=tx.get("hash"),
                )
            )

    return transactions


def group_transactions_into_ticks(
    transactions: list[Transaction],
    blocks: list[dict],
    mode: str = "block_per_tick",
) -> list[list[Transaction]]:
    """Split the flat transaction list back into one batch per tick. In block_per_tick mode (the only one so far) each block is one tick, which keeps each block's real burst size and mix but throws away the real gaps between blocks."""
    if mode != "block_per_tick":
        raise NotImplementedError(f"mode={mode!r} not implemented.")

    ticks: list[list[Transaction]] = []
    tx_index = 0
    for block in blocks:
        num_tx_in_block = len(block.get("transactions", []))
        ticks.append(transactions[tx_index: tx_index + num_tx_in_block])
        tx_index += num_tx_in_block

    return ticks


def load_transactions_from_alchemy(
    paths: str | list[str],
    node_ids: list[int],
    num_jurisdictions: int = 3,
    jurisdiction_strategy: str = "realistic",
    seed: int | None = None,
) -> list[list[Transaction]]:
    """Shortcut: load, tag and group in one call."""
    blocks = load_alchemy_blocks(paths)
    transactions = build_transactions_from_blocks(blocks, node_ids, num_jurisdictions, jurisdiction_strategy, seed)
    return group_transactions_into_ticks(transactions, blocks)
