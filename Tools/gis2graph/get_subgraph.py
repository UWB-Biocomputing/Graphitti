import networkx as nx
from random import randint

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

    # Setup a list of profiles for the PSAPs in the subgraph
    # The # of agents for SEATTLE PD were calculated from the average calls/hr and average call
    # duration using the Erlang C formula:
    # 57.25 calls/hr
    # 204 seconds of average duration
    # 10 seconds delay of callers tolerance
    #
    # The # of trunks was calculated using the Erlang units for the maximum hourly rate:
    # Max hourly rate: 137 calls/hr
    # Average duration: 204 seconds + 10 seconds pos-processing
    # Erlang = 137 * (214 / 3600) = 8.14
    # 
    # Then we use the number or Erlang units with the Erlang B equation to calculate the required
    # number of Trunks.
    # We used a blocking probability of 0.01 (1%)
    profiles = {'SEATTLE PD': {'agents': 6, 'trunks': 16},
            "King County Sheriff's Office": {'agents': 4, 'trunks': 12},
            'UW PD': {'agents': 2, 'trunks': 4},
            'PORT PD': {'agents': 2, 'trunks': 4},
            'SEATTLE PD - LAKE WASHINGTON': {'agents': 2, 'trunks': 4},
            'BOTHELL': {'agents': 2, 'trunks': 4},
            'Washington State Patrol - Bellevue': {'agents': 2, 'trunks': 4},
            'VALLEY COM': {'agents': 2, 'trunks': 4},
            "King County Sheriff's Office - Marine Patrol": {'agents': 2, 'trunks': 4},
            "King County Sheriff's Office - Marine Patrol for Shoreline": {'agents': 2, 'trunks': 4},
            "King County Sheriff's Office - Marine Patrol for Burien": {'agents': 2, 'trunks': 4}
    }

    # TODO: For now, we are assigning the number of responders at random
    min_responders = 3
    max_responders = 15

    # Loop over the nodes and assign them the number of agents, trunks, and responders
    for id, node in G_spd.nodes(data=True):
        if node['type'] == 'PSAP':
            node['agents'] = profiles[node['name']]['agents']
            node['trunks'] = profiles[node['name']]['trunks']
        elif node['type'] != 'CALR':
            node['agents'] = randint(min_responders, max_responders)
            node['trunks'] = int(node['agents'] * 1.25) # 25% more lines

    # This will give us a continuos numbering starting from zero
    G_spd = nx.convert_node_labels_to_integers(G_spd, first_label=0, ordering='default')
    nx.write_graphml(G_spd, subgraph_file, named_key_ids=True)

if __name__ == '__main__':
    main()