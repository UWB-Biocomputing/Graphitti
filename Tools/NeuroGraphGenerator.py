# NeuroGraphGenerator.py
# Omar Elgazzar

# Python script used to create the GraphML files
# used in the Neural Simulation

# Credit to Jardi A.M. Jordan for initial code

# IMPORTANT: Once you generate the GraphML file, you will need
# to manually change the keys "attr.type" member to the data type you desire
# For example for key 'x' you will need to change attr.type from "long" to "double"

# It is currently unknown whether the networkx library has a way to change the data
# type automatically, but from the research conducted this can't be done in this script
# and thus must be done manually.

import networkx as nx
G = nx.DiGraph()
height = 2 # Number of neurons in grid (number of neurons = height^2)

# ActiveNList = [0] Is this the same as excitatory? Assuming so:

InhibitoryNList = [1] # Specify which nodes are inhibitory neurons

G.add_nodes_from([i for i in range(height * height)])
for id, node in G.nodes(data=True):
        node['x'] = id % height
        node['y'] = id // height
        if (id in InhibitoryNList):
            node['type'] = 1 # INH in the simulator is an enumerator for 1
        else:
            node['type'] = 2 # EXC in the simulator is an enumerator for 2
nx.write_graphml(G, 'example.graphml', named_key_ids=True)
