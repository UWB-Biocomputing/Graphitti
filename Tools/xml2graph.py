import networkx as nx
G = nx.DiGraph()
height = 100
G.add_nodes_from([i for i in range(height * height)])
for id, node in G.nodes(data=True):
        node['x'] = id % height
        node['y'] = id // height
nx.write_graphml(G, 'test_graph.graphml', named_key_ids=True)