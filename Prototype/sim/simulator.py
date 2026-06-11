

from sim.transaction import Transaction

from sim.shardnode import ShardNode


class Simulator:
    nodes: list[ShardNode]
    ticks: int

    def __init__(self, nodes: list[ShardNode]):
        self.nodes = nodes
        self.ticks = 0

    def tick(self):
        # Process transactions in each shard node for the current tick
        for node in self.nodes:
            result = node.process()
            print(result)
        self.ticks += 1

    def route(self, transaction: Transaction) -> ShardNode:
        # Route the transaction to the appropriate shard node based on its jurisdiction
        for node in self.nodes:
            if node.jurisdiction == transaction.jurisdiction and not node.isfull():
                return node
        raise ValueError("No shard node found for the given transaction jurisdiction")