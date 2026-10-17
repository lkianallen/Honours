"""Small simulator demo, no RL: random writes and reads on a 9-node setup.

Writes are routed with the jurisdiction-filtered fallback router, then reads
that point at those writes are added so you can watch cross-shard latency (Eq. 5)
kick in. The Sepolia trace pipeline never generates reads, so this is the only
place that code path gets exercised.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from sim.simulator import Simulator
from sim.shardnode import ShardNode
from sim.transaction import Transaction


def generate_seeded_transactions(num_transactions, jurisdiction):
    """Writes where roughly half carry the given jurisdiction and the rest are random."""
    transactions = []
    for _ in range(num_transactions):
        payload_size = random.randint(1, 50)  # random size for now
        if random.random() <= 0.5:
            transaction = Transaction(payload_size, jurisdiction)
        else:
            transaction = Transaction(payload_size, random.choice([b'1', b'2', b'3']))
        transactions.append(transaction)
    return transactions


def generate_random_transactions(num_transactions):
    transactions = []
    for _ in range(num_transactions):
        payload_size = random.randint(1, 50)  # random size for now
        jurisdiction = random.choice([b'1', b'2', b'3'])
        transaction = Transaction(payload_size, jurisdiction)
        transactions.append(transaction)
    return transactions


def generate_seeded_reads(written_transactions, num_reads, created_tick=0):
    """Reads that point at blocks from an already-routed batch of writes.

    Call this after assign_transactions() so block_registry knows those ids.
    A read's own jurisdiction is random, so it's cross-shard whenever it
    differs from the jurisdiction of the block it references.
    """
    reads = []
    for _ in range(num_reads):
        ref_tx = random.choice(written_transactions)
        payload_size = random.randint(1, 10)  # a query, not a block write
        jurisdiction = random.choice([b'1', b'2', b'3'])
        read = Transaction(
            payload_size,
            jurisdiction,
            is_read=True,
            ref_block_id=ref_tx.id,
            created_tick=created_tick,
        )
        reads.append(read)
    return reads


def initialise_shardnodes(num_nodes_per_jurisdiction):
    shard_nodes = []
    for i in range(1, 4):  # 3 jurisdictions, so N = 9 with 3 nodes each
        for j in range(num_nodes_per_jurisdiction):
            shard_node = ShardNode(
                id=(i - 1) * num_nodes_per_jurisdiction + j + 1,
                jurisdiction=bytes(str(i), 'utf-8'),
                Cap_cpu=100.0,
                Cap_disk=10_000.0,
            )
            shard_nodes.append(shard_node)
            print(f"Init S{shard_node.id} - {shard_node.jurisdiction}")
    return shard_nodes


random.seed(42)
shard_nodes = initialise_shardnodes(num_nodes_per_jurisdiction=3)
simulator = Simulator(shard_nodes)
transaction_list = generate_random_transactions(20)
 

simulator.assign_transactions(transaction_list)  # writes land in block_registry

read_list = generate_seeded_reads(transaction_list, num_reads=10)
simulator.assign_transactions(read_list)  # reads point at the writes above

for _ in range(10):
    simulator.tick()
simulator.print_results()
