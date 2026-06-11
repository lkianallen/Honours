from sim.simulator import Simulator
from sim.shardnode import ShardNode
from sim.transaction import Transaction



print("Starting simulator...")
shard_nodes = [
    ShardNode(id=1, jurisdiction=b'1', cpu=100, disk=100),
    ShardNode(id=2, jurisdiction=b'1', cpu=100, disk=100),
    ShardNode(id=3, jurisdiction=b'1', cpu=100, disk=100),
    ShardNode(id=4, jurisdiction=b'2', cpu=100, disk=100),
    ShardNode(id=5, jurisdiction=b'2', cpu=100, disk=100),
    ShardNode(id=6, jurisdiction=b'2', cpu=100, disk=100),
    ShardNode(id=7, jurisdiction=b'3', cpu=100, disk=100),
    ShardNode(id=8, jurisdiction=b'3', cpu=100, disk=100),
    ShardNode(id=9, jurisdiction=b'3', cpu=100, disk=100)
]
simulator = Simulator(shard_nodes)

transaction_list = [
    (10, b'1'),
    (10, b'2'),
    (10, b'3'),
]
for payload_size, jurisdiction in transaction_list:
    #for i in range(3):
        transaction = Transaction(payload_size, jurisdiction)
        node = simulator.route(transaction)
        node.queue.append(transaction)
        print(f"payload size {payload_size} jurisdiction {jurisdiction} routed to node {node.id}")
simulator.tick()