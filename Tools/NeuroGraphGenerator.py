# NeuroGraphGenerator.py
# Omar Elgazzar

# Python script used to create the GraphML files
# used in the Neural Simulation

# Credit to Jardi A.M. Jordan for initial code

# IMPORTANT: Once you generate the GraphML file, you will need
# to manually change the keys "attr.type" member
# for the key 'active'. You will need to change attr.type from "long" to "boolean"

# It is currently unknown whether the networkx library has a way to change the data
# type automatically, but from the research conducted this can't be done in this script
# and thus must be done manually.

import networkx as nx
G = nx.DiGraph()
height = 2 # Number of neurons in grid (number of neurons = height^2)

ActiveNList = [0] # Specify which nodes are endogenously active

InhibitoryNList = [1] # Specify which nodes are inhibitory neurons

G.add_nodes_from([i for i in range(height * height)])
for id, node in G.nodes(data=True):
        node['x'] = float(id % height)
        node['y'] = float(id // height)
        if (id in ActiveNList):
            node['active'] = 1
        else:
            node['active'] = 0
        if (id in InhibitoryNList):
            node['type'] = "INH"
        else:
            node['type'] = "EXC"
nx.write_graphml(G, 'example.graphml', named_key_ids=True)
