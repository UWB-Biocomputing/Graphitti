# Jasleen Kaur Saini
# Zaina Shaikh
# Graph Generation
# graph_generation.py

# Purpose: This script generates synthetic star graphs representing Public Safety Answering Points (PSAPs), responders, and callers.
# The generated graphs can be used for simulations and analysis in emergency response systems. 

import matplotlib.pyplot as plt
import networkx as nx
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import math

# Function to partition the bounding box into exact non-overlapping PSAP regions.
def generate_psap_regions(bbox, num_psaps):
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    num_cols = int(math.ceil(math.sqrt(num_psaps)))
    num_rows = int(math.ceil(num_psaps / num_cols))
    
    while num_rows * num_cols > num_psaps:
        num_cols -= 1
        if num_cols <= 0:
            num_cols = 1
            break
        num_rows = int(math.ceil(num_psaps / num_cols))

    region_width = width / num_cols
    region_height = height / num_rows

    regions = []
    for i in range(num_rows):
        for j in range(num_cols):
            if len(regions) >= num_psaps:
                break
            region_xmin = x_min + j * region_width
            region_ymin = y_min + i * region_height
            region_xmax = region_xmin + region_width
            region_ymax = region_ymin + region_height
            regions.append((region_xmin, region_ymin, region_width, region_height))
    
    return regions

# Function to partition a PSAP region into exact non-overlapping caller regions.
def generate_caller_regions(psap_region, num_callers):
    x_min, y_min, width, height = psap_region
    num_cols = int(math.ceil(math.sqrt(num_callers)))
    num_rows = int(math.ceil(num_callers / num_cols))
    
    while num_rows * num_cols > num_callers:
        num_cols -= 1
        if num_cols <= 0:
            num_cols = 1
            break
        num_rows = int(math.ceil(num_callers / num_cols))

    caller_width = width / num_cols
    caller_height = height / num_rows

    caller_regions = []
    for i in range(num_rows):
        for j in range(num_cols):
            if len(caller_regions) >= num_callers:
                break
            caller_xmin = x_min + j * caller_width
            caller_ymin = y_min + i * caller_height
            caller_xmax = caller_xmin + caller_width
            caller_ymax = caller_ymin + caller_height
            caller_regions.append((caller_xmin, caller_ymin, caller_width, caller_height))
    
    return caller_regions

# Function to combine two regions into a larger region.
def combine_regions(region1, region2):
    combined_region_xmin = min(region1[0], region2[0])
    combined_region_ymin = min(region1[1], region2[1])
    combined_region_xmax = max(region1[0] + region1[2], region2[0] + region2[2])
    combined_region_ymax = max(region1[1] + region1[3], region2[1] + region2[3])
    
    combined_region = (combined_region_xmin, combined_region_ymin, combined_region_xmax - combined_region_xmin, combined_region_ymax - combined_region_ymin)
    
    return combined_region

# Function to handle special case when the number of PSAPs is a prime number.
def generate_special_psap_regions(bbox, num_psaps):
    num_regions = num_psaps + 1
    regions = generate_psap_regions(bbox, num_regions)
    
    empty_region = regions[-1]
    closest_region = min(regions[:-1], key=lambda r: (abs(r[0] - empty_region[0]) + abs(r[1] - empty_region[1])))

    combined_region = combine_regions(empty_region, closest_region)
    adjusted_regions = [r for r in regions[:-1] if r != closest_region] + [combined_region]

    return adjusted_regions

# Function to handle special case when the number of callers is a prime number.
def generate_special_caller_regions(psap_region, num_callers):
    num_regions = num_callers + 1
    regions = generate_caller_regions(psap_region, num_regions)
    
    empty_region = regions[-1]
    closest_region = min(regions[:-1], key=lambda r: (abs(r[0] - empty_region[0]) + abs(r[1] - empty_region[1])))

    combined_region = combine_regions(empty_region, closest_region)
    adjusted_regions = [r for r in regions[:-1] if r != closest_region] + [combined_region]

    return adjusted_regions

# Function to generate a star graph centered around a PSAP with responders and callers.
def generate_star_graph(psap_id, num_responders, num_callers, psap_pos, psap_region):
    G = nx.Graph()
    psap_x, psap_y = psap_pos
    G.add_node(psap_id, type='PSAP', pos=(psap_x, psap_y))

    responder_types = ['EMS', 'FIRE', 'LAW']
    # Generate responders of each type
    for responder_type in responder_types:
        for i in range(num_responders):
            responder_id = f'{psap_id}_RESP_{responder_type}_{i}'
            responder_x = random.uniform(psap_region[0], psap_region[0] + psap_region[2])
            responder_y = random.uniform(psap_region[1], psap_region[1] + psap_region[3])
            G.add_node(responder_id, type='RESP', responder_type=responder_type, pos=(responder_x, responder_y))
            G.add_edge(psap_id, responder_id)

    # Add callers
    if is_prime(num_callers):
        caller_regions = generate_special_caller_regions(psap_region, num_callers)
    else:
        caller_regions = generate_caller_regions(psap_region, num_callers)
        
    for i, caller_region in enumerate(caller_regions):
        num_callers_in_region = num_callers // len(caller_regions)
        for j in range(num_callers_in_region):
            caller_id = f'{psap_id}_CALL_{i * num_callers_in_region + j}'
            caller_x = random.uniform(caller_region[0], caller_region[0] + caller_region[2])
            caller_y = random.uniform(caller_region[1], caller_region[1] + caller_region[3])
            G.add_node(caller_id, type='CALL', pos=(caller_x, caller_y))
            G.add_edge(psap_id, caller_id)

    return G

# Function to check if a number is prime.
def is_prime(n):
    # Exception: 2 is a prime number, however, allows for even segments. 
    if n == 2:
        return False
    elif n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

# Function to generate a set of star graphs for multiple PSAPs.
def generate_graph_set(num_psaps, num_responders_per_psap, num_callers_per_psap, bounding_box):
    G = nx.Graph()
    xmin, ymin, xmax, ymax = bounding_box

    if is_prime(num_psaps):
        psap_regions = generate_special_psap_regions(bounding_box, num_psaps)
    else:
        psap_regions = generate_psap_regions(bounding_box, num_psaps)

    for i, region in enumerate(psap_regions):
        psap_x = random.uniform(region[0], region[0] + region[2])
        psap_y = random.uniform(region[1], region[1] + region[3])
        psap_pos = (psap_x, psap_y)

        star_graph = generate_star_graph(f'PSAP_{i}', num_responders_per_psap, num_callers_per_psap, psap_pos, region)
        G = nx.compose(G, star_graph)

    G.graph['global_bounding_box'] = f"{xmin}, {ymin}, {xmax}, {ymax}"
    G.graph['psap_regions'] = psap_regions
    G.graph['caller_regions'] = [generate_caller_regions(region, num_callers_per_psap) if not is_prime(num_callers_per_psap) else generate_special_caller_regions(region, num_callers_per_psap) for region in psap_regions]

    return G

# Function to convert graph attributes to string format for GraphML.
def convert_graphml_attrs(G):
    for node, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, tuple):
                G.nodes[node][key] = f'{value[0]},{value[1]}'

# Function to create the GraphML representation of the graph.
def create_graphml(G):
    def add_node(node_id, data):
        node_elem = ET.Element('node', id=node_id)
        for key, value in data.items():
            data_elem = ET.SubElement(node_elem, 'data', key=key)
            data_elem.text = str(value)
        return node_elem

    def add_edge(source, target):
        edge_elem = ET.Element('edge', source=source, target=target)
        return edge_elem

    root = ET.Element('graphml', xmlns="http://graphml.graphdrawing.org/xmlns")
    graph_elem = ET.SubElement(root, 'graph', id='G', edgedefault='undirected')

    global_bbox = G.graph.get('global_bounding_box')
    if global_bbox:
        bbox_elem = ET.SubElement(graph_elem, 'data', key="global_bounding_box")
        bbox_elem.text = global_bbox

    psap_regions = G.graph.get('psap_regions', [])
    if psap_regions:
        regions_elem = ET.SubElement(graph_elem, 'data', key="psap_regions")
        regions_elem.text = ' | '.join([f"{x_min}, {y_min}, {x_max}, {y_max}" for (x_min, y_min, x_max, y_max) in psap_regions])

    caller_regions = G.graph.get('caller_regions', [])
    if caller_regions:
        all_caller_regions = [region for sublist in caller_regions for region in sublist]
        caller_elem = ET.SubElement(graph_elem, 'data', key="caller_regions")
        caller_elem.text = ' | '.join([f"{x_min}, {y_min}, {x_max}, {y_max}" for (x_min, y_min, x_max, y_max) in all_caller_regions])

    for node, data in G.nodes(data=True):
        node_elem = add_node(node, data)
        graph_elem.append(node_elem)

    for source, target in G.edges():
        edge_elem = add_edge(source, target)
        graph_elem.append(edge_elem)

    xml_str = ET.tostring(root, encoding='utf-8', method='xml')
    parsed_xml = minidom.parseString(xml_str)
    pretty_xml_str = parsed_xml.toprettyxml(indent="  ")

    return pretty_xml_str

# Function to save the GraphML representation to a file.
def save_graphml(G, filename):
    graphml_str = create_graphml(G)
    with open(filename, 'w') as f:
        f.write(graphml_str)


# Main function to execute the graph generation and save the GraphML file.
# Contains testable parameters. 
# Simplified Case: 1 centeral PSAP, 1 caller and 3 responders (1 set)
# Bounding box, PSAP region and caller region are the same (simplified analysis ) 
def main():
    num_psaps = 9
    num_callers_per_psap = 7
    # Responders generated in sets of 3 (EMS, FIRE, LAW)
    num_responders_per_psap = 1
    # Control variable 
    bounding_box = (0, 0, 100, 100)
    
    G = generate_graph_set(num_psaps, num_responders_per_psap, num_callers_per_psap, bounding_box)

    # Save the GraphML file using the first format
    output_file = "synth_graph.graphml"
    save_graphml(G, output_file)
    print("The GraphML has been formatted and saved to 'synth_output.graphml'.")

# Execute the main function.
if __name__ == "__main__":
    main()


     
 