import random
from .transaction import Transaction
from .shardnode import ShardNode


class Simulator:
    nodes: list[ShardNode]
    ticks: int

    def __init__(self, nodes: list[ShardNode]):
        self.nodes = nodes
        self.ticks = 0

    def tick(self):
        # Process transactions in each shard node for the current tick
        print(f"             Tick {self.ticks + 1}:")
        print("-----------------------------------")

        for node in self.nodes:
            result = node.process()
        self.ticks += 1

    def route(self, transaction: Transaction) -> int:
        # should also have 'home' shard variable, i.e. where transaction was initially bound for
        eligible = []
        for node in self.nodes:
            if node.jurisdiction == transaction.jurisdiction and not node.isfull():
                eligible.append(node)
        if eligible:
            selected_node = random.choice(eligible)
            return selected_node.id
        raise ValueError("No shard node found for the given transaction jurisdiction")
    
    def assign_transactions(self, transactions: list[Transaction]):
        for transaction in transactions:
            nodeID = self.route(transaction)
            node = self.nodes[nodeID - 1] 
            node.write_queue.append(transaction)
            print(f"payload size {transaction.payload_size} jurisdiction {transaction.jurisdiction} routed to node {node.id}")

    def print_results(self):
        print(f"Simulation completed in {self.ticks} ticks.")
        for node in self.nodes:
            print(f"Node {node.id}: CPU {node.cpu}, Disk {node.disk}")