import random
from sim.simulator import Simulator
from sim.shardnode import ShardNode
from sim.transaction import Transaction
from sim.visualiser import Visualiser

# generates a list of transactions favoured towards given jurisdiction
def generate_seeded_transactions(num_transactions, jurisdiction):
    transactions = []
    for _ in range(num_transactions):
        payload_size = random.randint(1, 50)  # Random size for now
        if random.random() <= 0.5:
            transaction = Transaction(payload_size, jurisdiction)
        else:             
            transaction = Transaction(payload_size, random.choice([b'1', b'2', b'3']))
        transactions.append(transaction)
    return transactions

def generate_random_transactions(num_transactions):
    transactions = []
    for _ in range(num_transactions):
        payload_size = random.randint(1, 50)  # Random size for now
        jurisdiction = random.choice([b'1', b'2', b'3'])
        transaction = Transaction(payload_size, jurisdiction)
        transactions.append(transaction)
    return transactions

def initialise_shardnodes(num_nodes_per_jurisdiction):
    shard_nodes = []
    for i in range(1, 4):  # Assuming 3 jurisdictions
        for j in range(num_nodes_per_jurisdiction):
            shard_node = ShardNode(id=(i-1)*num_nodes_per_jurisdiction + j + 1, jurisdiction=bytes(str(i), 'utf-8'), cpu=100, disk=1000)
            shard_nodes.append(shard_node)
            print(f"Initialised Shard Node {shard_node.id} with jurisdiction {shard_node.jurisdiction}")
    return shard_nodes



random.seed(42)
shard_nodes = initialise_shardnodes(num_nodes_per_jurisdiction=3)
simulator = Simulator(shard_nodes)
transaction_list = generate_random_transactions(20)
'''
visualiser = Visualiser(simulator)
visualiser.run(transaction_list)
 '''
''' TRANSACTION DEBUGGING
print("Generated Transactions:")
print ("-----------------------------")
print (f"{'Payload Size':<15} {'Jurisdiction':<15}")   
'''
# SIMULATOR TESTING
simulator.assign_transactions(transaction_list)
for _ in range(10):  # Simulate 10 ticks
    simulator.tick()
simulator.print_results()
