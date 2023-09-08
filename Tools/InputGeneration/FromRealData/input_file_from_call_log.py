import pandas as pd
import networkx as nx
import numpy as np
import lxml.etree as et

def main():
    """Generates an XML file with a list of calls."""

    # call log from the Seattle PD PSAP for September 2020. 
    # From this log we are using start_time and talk_time, then we add the (x,y)
    # coordinates by uniformily distributing the calls amongst a set of caller
    # region sections extracted from a GIS dataset.
    call_log = "SPD_call_log.csv"
    # Seattle PD PSAP graph
    graph_file = "../../gis2graph/graph_files/spd.graphml"
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
    call_log["time"] = pd.to_datetime(call_log["time"], format="%m/%d/%Y %H:%M:%S")

    # Sort the calls so they are in order
    sorted = call_log.sort_values(["vertex","time"])
    # Convert start_time to seconds since the first call in the list
    first_time = sorted["time"].iloc[0]
    sorted['time'] = sorted.apply(lambda call: (call['time'] - first_time).seconds + \
                                                     (call['time'] - first_time).days * 24 * 3600, axis=1)

    # Conver duration to seconds
    sorted['duration'] = pd.to_timedelta(sorted['duration'])
    sorted['duration'] = sorted.apply(lambda call: call['duration'].seconds, axis=1)

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

    # Add patience time. This is the time a caller is willing to wait before
    # abandoning the queue.
    # We will be modeling patience as an exponentially distributed random variable.
    # The average waiting time is 1/abandonment rate and the abandonment rate is
    # estimated using Mandelbaum and Zeltyn's formula:
    #     Abandonment rate = fraction of abandonment / average wait time
    # 
    # Patience time calculated from September 2020 data:
    #   Fraction of abandonment = 0.0942
    #   Avg Wait Time = 4.65 seconds
    #   Abandonment rate = 0.0942/4.65 = 0.020258/second
    #   Avg. Patience = 1/0.020258 = 49.36 Seconds
    avg_patience = 49.36
    sorted['patience'] = np.random.exponential(scale=avg_patience, size=sorted.shape[0]).astype(np.int64)

    # this is the root element
    inputs = et.Element('simulator_inputs')

    # Construct an element tree to be writen to a file in XML format
    # data is the root element
    data = et.SubElement(inputs, 'data', {"description": "SPD_calls_sept2020", 
                                          "clock_tick_size": "1",
                                          "clock_tick_unit": "sec"})

    # Insert one event element per row
    # Make sure there are no time duplicates
    prev_time = -1
    vertex = et.SubElement(data, 'vertex', {'id': sorted.iloc[0]['vertex_id'], 'name': sorted.iloc[0]['vertex']})
    for idx, row in sorted.iterrows():
        d = row.to_dict()
        if d['duration'] == 0:
            continue    # not including zero duration calls

        # If vertex_id is different create a new vertex node
        if vertex.attrib['id'] != d['vertex_id']:
            vertex = et.SubElement(data, 'vertex', {'id': d['vertex_id'], 'name': d['vertex']})

        # remove vertex id and name from the dictionary to avoid redundancy
        del d['vertex']

        # ensure that we don't have duplicate times
        if d['time'] <= prev_time:
            d['time'] = prev_time + 1
        prev_time = d['time']

        # convert everything to strings
        for k, v in d.items():
            d[k] = str(v)

        # We could add attributes to the event node
        event = et.SubElement(vertex, 'event', d)
        # for k, v in d.items():
        #     attr = et.SubElement(event, k)
        #     attr.text = str(v)

    tree = et.ElementTree(inputs)
    tree_out = tree.write("SPD_calls.xml",
                          xml_declaration=True,
                          encoding="UTF-8",
                          pretty_print=True)


if __name__ == '__main__':
    main()