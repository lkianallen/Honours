import numpy as np
from collections import deque
from .transaction import Transaction

class ShardNode:
    id: int
    jurisdiction: bytes
    read_queue: deque[Transaction]
    write_queue: deque[Transaction]
    cpu: int
    disk: int
    cpu_max: int
    disk_max: int
    #add round Robin scheduling to the shard node in later version

    def __init__(self, id: int, jurisdiction: bytes, cpu: int, disk: int):
        self.id = id
        self.jurisdiction = jurisdiction
        self.read_queue = deque()
        self.write_queue = deque()
        self.cpu = cpu
        self.disk = disk
        self.cpu_max = cpu
        self.disk_max = disk

    def process(self):
        # Process transactions in the queue based on the node's CPU and disk capacity
        '''
        if self.read_queue: 
            transaction = self.read_queue[0]
            if transaction.jurisdiction != self.jurisdiction:
                # cross-shard read transactions consume CPU and introduce more latency 
                self.cpu -= transaction.payload_size
                self.read_queue.popleft()
                print(f"Node {self.id}: cross-shard read - latency penalty") 
                # actual latency penalty can be modeled in the simulator tick function based on the transaction payload size and a fixed latency
            else:
                # Write transactions consume both CPU and disk                
                self.cpu -= transaction.payload_size
                self.read_queue.popleft()
                print(f"Node {self.id}: processed read transaction")
        '''
        if self.write_queue and self.cpu > 0 + self.write_queue[0].payload_size and self.disk > 0:
            transaction = self.write_queue[0]
            if transaction.jurisdiction != self.jurisdiction:
                self.cpu -= transaction.payload_size
                self.disk -= transaction.payload_size
                self.write_queue.popleft()
                print(f"Node {self.id}: cross-shard write - latency penalty")
                # latency penalty
            else:
                # Write transactions consume both CPU and disk
                self.cpu -= transaction.payload_size
                self.disk -= transaction.payload_size
                self.write_queue.popleft()
                print(f"Node {self.id}: processed write transaction")
        
        self.cpu = self.cpu * 1.2 + 5  # Simulate CPU recovery over time
        if self.cpu > self.cpu_max:
            self.cpu = self.cpu_max


    def isfull(self) -> bool:
        # Check if the node is full based on its CPU and disk capacity
        return self.cpu <= 0 or self.disk <= 0
