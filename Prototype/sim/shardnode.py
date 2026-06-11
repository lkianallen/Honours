import numpy as np
from sim.transaction import Transaction

class ShardNode:
    id: int
    jurisdiction: bytes
    queue: list[Transaction]
    cpu: int
    disk: int
    #add round Robin scheduling to the shard node in later version

    def __init__(self, id: int, jurisdiction: bytes, cpu: int, disk: int):
        self.id = id
        self.jurisdiction = jurisdiction
        self.queue = []
        self.cpu = cpu
        self.disk = disk

    def process(self) -> str:
        # Process transactions in the queue based on the node's CPU and disk capacity
        if self.queue and self.cpu > 0 and self.disk > 0:
            transaction = self.queue[0]
            if transaction.jurisdiction != self.jurisdiction:
                # Read transactions consume CPU but not disk
                self.cpu -= transaction.payload_size
                self.queue.pop(0)
                return f"Node {self.id}: read"
            else:
                # Write transactions consume both CPU and disk
                self.cpu -= transaction.payload_size
                self.disk -= transaction.payload_size
                self.queue.pop(0)
                return f"Node {self.id}: write"


    def isfull(self) -> bool:
        # Check if the node is full based on its CPU and disk capacity
        return self.cpu <= 0 or self.disk <= 0
