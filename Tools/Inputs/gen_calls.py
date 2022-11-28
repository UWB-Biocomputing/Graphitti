import pandas as pd
import networkx as nx
import numpy as np

def main():
    """Generates an XML file with a list of calls."""

    # call log from the Seattle PD PSAP for September 2020. 
    # From this log we are using start_time and talk_time, then we add the (x,y)
    # coordinates by uniformily distributing the calls amongst a set of caller
    # region sections extracted from a GIS dataset.
    call_log = "SPD_call_log.csv"
    # Seattle PD PSAP graph
    graph_file = "../gis2graph/graph_files/spd.graphml"
    # call ratio per emergency type in percentage.
    # TODO: This values are not based on data. Need to either find out from SPD or
    #       from relevant research data.
    ems = 50
    law = 30
    fire = 20

    # Id of caller region these calls correspond to
    SPD_caller_region_id = '194'

    # Read files
    call_log = pd.read_csv(call_log)
    G = nx.read_graphml(graph_file)

    # use eval() to convert the string into python a list
    SPD_grid = eval(G.nodes[SPD_caller_region_id]["segments"]) # node 194 is Seattle PD's Caller Region
    call_log["start_time"] = pd.to_datetime(call_log["start_time"], format="%m/%d/%Y %H:%M:%S")

    # Sort the calls so they are in order
    sorted = call_log.sort_values("start_time")

    # At this point the call log only has the start_time and talk_time.
    # We will generate the coordinates by uniformily distributing the calls along
    # the caller region grid.
    # The grid is a set of square denoted by their bottom left and upper right corners.
    # We get the grid from the graphml file it looks like this:
    # SPD_grid = [[(-122.40326150882478, 47.59789698297564),
    #              (-122.37990573296226, 47.62125275883817)],
    #             [(-122.47332883641236, 47.55118543125059),
    #              (-122.44997306054984, 47.574541207113114)], ... ]
    grid_size = len(SPD_grid)
    np.random.seed(12346)   # Seed to make random numbers predictable

    # Distribute the calls amongst the square using a uniform distribution
    sorted['grid_idx'] = np.random.randint(0, len(SPD_grid), sorted.shape[0])
    sorted['x'] = sorted.apply(lambda x: (np.random.uniform(SPD_grid[x['grid_idx']][0][0], SPD_grid[x['grid_idx']][1][0])), axis=1)
    sorted['y'] = sorted.apply(lambda x: (np.random.uniform(SPD_grid[x['grid_idx']][0][1], SPD_grid[x['grid_idx']][1][1])), axis=1)

    # We will also uniformily distribute the calls between 3 emergency types: EMS, Fire, and Law.
    # Assigned a number from 0 to 99 using a uniform distribution, which will be
    # used as a threshold for the emergency type.
    sorted['type_prob'] = np.random.randint(0, 100, sorted.shape[0])
    # 20% of the values should be under 20, 30% between 20 and 49, 50% between 50 and 99
    sorted['type'] = sorted.apply(lambda call: 'Fire' if call['type_prob'] < fire else 'Law' if call['type_prob'] < ems else 'EMS', axis=1)
    sorted = sorted.assign(vertex_id = SPD_caller_region_id)
    sorted = sorted.assign(vertex =  G.nodes[SPD_caller_region_id]['name'])

    # Clean up
    sorted = sorted.drop(['grid_idx', 'type_prob'], axis=1)

    # Save dataframe
    sorted.to_xml("SPD_calls.xml", index=False, row_name="event")


if __name__ == '__main__':
    main()