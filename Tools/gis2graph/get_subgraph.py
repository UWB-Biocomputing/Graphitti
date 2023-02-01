import networkx as nx

def main():
    """Extracts the subgraph for the Seattle PD PSAP from the King County GraphML file."""

    graphml_file = "graph_files/King_county_NG911.graphml"
    edge_filter = "977" # this is the id of the SEATTLE PD vertex in the King_County graph
    subgraph_file = "graph_files/spd.graphml"
    G =nx.read_graphml(graphml_file)

    # extract the subgraph for the given vertex
    G_spd = nx.DiGraph()
    G_spd.add_edges_from(G.out_edges(edge_filter, data=True))
    G_spd.add_edges_from(G.in_edges(edge_filter, data=True))
    for node in G_spd.nodes():
        G_spd.nodes[node].update(G.nodes[node])

    # This will give us a continuos numbering starting from zero
    G_spd = nx.convert_node_labels_to_integers(G_spd, first_label=0, ordering='default')
    nx.write_graphml(G_spd, subgraph_file, named_key_ids=True)

if __name__ == '__main__':
    main()